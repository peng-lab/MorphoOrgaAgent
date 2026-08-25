"""Cellpose and SAM3 helpers for organoid instance segmentation."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SAM3_REPO = Path("/home/haicu/hanyi.zhang/sam3")
DEFAULT_TEXT_PROMPT = "cell cluster"


def run_cellpose_single_image(
    image_path: Path,
    output_dir: Path,
) -> tuple[np.ndarray, Path]:
    """Run Cellpose on one image and persist its coarse label mask."""
    import cv2
    import torch
    from cellpose import models

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type="cyto2")
    masks, *_ = model.eval(image_bgr, diameter=None, channels=[0, 0])
    label_mask = np.asarray(masks, dtype=np.int32)

    base_name = image_path.stem
    mask_path = output_dir / f"{base_name}_label_mask.npy"
    np.save(mask_path, label_mask)
    save_label_mask_debug_images(base_name, output_dir, image_bgr, label_mask)
    return label_mask, mask_path


def save_label_mask_debug_images(
    base_name: str,
    output_dir: Path,
    image_bgr: np.ndarray,
    label_mask: np.ndarray,
) -> dict[str, Path]:
    """Save a label mask plus colored and overlaid diagnostic images."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    colored = colorize_label_mask(label_mask)

    pure_mask_path = output_dir / f"{base_name}_pure_mask.png"
    overlay_path = output_dir / f"{base_name}_overlay.png"
    eval_png_path = output_dir / f"{base_name}.png"

    cv2.imwrite(str(eval_png_path), label_mask.astype(np.uint16, copy=False))
    cv2.imwrite(str(pure_mask_path), colored)

    overlay = image_bgr.copy()
    if int(label_mask.max(initial=0)) > 0:
        cv2.addWeighted(colored, 0.4, overlay, 0.6, 0, overlay)
        for obj_id in range(1, int(label_mask.max()) + 1):
            obj_mask = (label_mask == obj_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                obj_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "eval_png": eval_png_path,
        "pure_mask": pure_mask_path,
        "overlay": overlay_path,
    }


def colorize_label_mask(label_mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Convert an integer label mask to a deterministic color image."""
    num_objects = int(label_mask.max(initial=0))
    color_mask = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    if num_objects == 0:
        return color_mask

    rng = np.random.default_rng(seed)
    colors = rng.integers(50, 255, size=(num_objects + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[label_mask.astype(np.int32, copy=False)]


def ensure_sam3_importable(sam3_repo: Path = SAM3_REPO) -> None:
    """Add the configured local SAM3 checkout to the import path."""
    repo_string = str(sam3_repo)
    if sam3_repo.exists() and repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def configure_cuda_math(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def sam3_inference_context(device: str):
    if device != "cuda":
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def label_mask_to_xywh_boxes(
    mask: np.ndarray,
    min_area: int = 1,
) -> list[list[float]]:
    """Return one pixel-space ``[x, y, width, height]`` box per label."""
    boxes: list[list[float]] = []
    for label_id in sorted(int(label) for label in np.unique(mask) if label != 0):
        ys, xs = np.where(mask == label_id)
        if xs.size < min_area:
            continue

        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    return boxes


def xywh_to_normalized_cxcywh(
    box_xywh: Iterable[float],
    width: int,
    height: int,
) -> list[float]:
    """Convert a pixel-space XYWH box to normalized center-format coordinates."""
    x, y, box_width, box_height = [float(value) for value in box_xywh]
    return [
        (x + box_width / 2.0) / float(width),
        (y + box_height / 2.0) / float(height),
        box_width / float(width),
        box_height / float(height),
    ]


def build_sam3_processor(confidence_threshold: float, device: str):
    """Build the SAM3 image processor lazily."""
    ensure_sam3_importable()
    configure_cuda_math(device)

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cpu":
        raise RuntimeError(
            "This SAM3 checkout cannot build on CPU because its image model "
            "creates CUDA tensors internally. Run on a CUDA node."
        )

    sam3_root = Path(sam3.__file__).resolve().parent
    bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    with sam3_inference_context(device):
        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
        )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def get_sam3_masks_from_state(
    inference_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Read masks and scores from a SAM3 inference state as NumPy arrays."""
    import torch

    masks = inference_state.get("masks")
    scores = inference_state.get("scores")
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool), np.zeros((0,), dtype=np.float32)

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().float().cpu().numpy()
    else:
        masks_np = np.asarray(masks, dtype=np.float32)

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim != 3:
        raise ValueError(f"Expected SAM3 masks with shape [N,H,W], got {masks_np.shape}")

    if scores is None:
        scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
    elif isinstance(scores, torch.Tensor):
        scores_np = scores.detach().float().cpu().numpy().astype(np.float32)
    else:
        scores_np = np.asarray(scores, dtype=np.float32)

    return masks_np.astype(bool), scores_np.reshape(-1)


def sam3_masks_to_label_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Combine SAM3 boolean masks into one integer instance-label mask."""
    height, width = image_shape
    label_mask = np.zeros((height, width), dtype=np.int32)
    if masks.size == 0:
        return label_mask

    next_label = 1
    for mask_index in np.argsort(scores):
        mask = masks[int(mask_index)]
        if mask.shape != (height, width):
            raise ValueError(f"SAM3 mask shape {mask.shape} does not match {(height, width)}")
        if not np.any(mask):
            continue
        label_mask[mask] = next_label
        next_label += 1
    return label_mask


def run_sam3_refinement_single_image(
    image_path: Path,
    cellpose_mask: np.ndarray,
    output_dir: Path,
    confidence_threshold: float,
    min_cellpose_area: int,
    device: str,
    text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> tuple[np.ndarray, Path]:
    """Refine a Cellpose label mask with SAM3 and persist the result."""
    import cv2
    import torch
    from PIL import Image

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    width, height = image_pil.size

    boxes_xywh = label_mask_to_xywh_boxes(cellpose_mask, min_area=min_cellpose_area)
    if not boxes_xywh:
        label_mask = np.zeros((height, width), dtype=np.int32)
    else:
        processor = build_sam3_processor(
            confidence_threshold=confidence_threshold,
            device=device,
        )
        with torch.inference_mode(), sam3_inference_context(device):
            inference_state = processor.set_image(image_pil)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                prompt=text_prompt,
                state=inference_state,
            )
            for box_xywh in boxes_xywh:
                inference_state = processor.add_geometric_prompt(
                    state=inference_state,
                    box=xywh_to_normalized_cxcywh(
                        box_xywh,
                        width=width,
                        height=height,
                    ),
                    label=True,
                )

            sam3_masks, sam3_scores = get_sam3_masks_from_state(inference_state)
            label_mask = sam3_masks_to_label_mask(
                sam3_masks,
                sam3_scores,
                image_shape=(height, width),
            )

    base_name = image_path.stem
    mask_path = output_dir / f"{base_name}.npy"
    np.save(mask_path, label_mask.astype(np.int32, copy=False))
    save_label_mask_debug_images(base_name, output_dir, image_bgr, label_mask)
    return label_mask, mask_path

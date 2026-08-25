# organoid_agent/tools/visualization_tools.py

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional

# Set style for high-quality scientific plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})


# =========================================================================
# Phase 1: Overlay & Quality Control Visualizations
# =========================================================================

def plot_mask_overlay(image: np.ndarray, mask: np.ndarray, save_path: str, alpha: float = 0.4) -> None:
    """
    Generates a semi-transparent colored mask overlay over the original organoid image.
    Each unique object ID gets a distinct color, with its ID labeled near its upper-right corner.
    """
    # Ensure image is 3-channel BGR/RGB
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_img = image.copy()

    object_ids = [int(x) for x in np.unique(mask) if x > 0]
    
    # Generate distinct random colors for each object
    np.random.seed(42)
    colors = {obj_id: np.random.randint(0, 255, size=3).tolist() for obj_id in object_ids}
    
    overlay = vis_img.copy()
    
    for obj_id in object_ids:
        obj_mask = (mask == obj_id)
        color = colors[obj_id]
        overlay[obj_mask] = color
        
        # Find contours to anchor the text label near the top/rightmost boundary
        contours, _ = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if contours:
            # Get the top-rightmost point of the contour
            pts = contours[0].reshape(-1, 2)
            top_right_idx = np.argmin(pts[:, 1] - pts[:, 0])  # minimizes y - x
            label_x, label_y = pts[top_right_idx]
            
            # Put label with a subtle drop shadow
            cv2.putText(vis_img, f"#{obj_id}", (label_x + 2, label_y - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_img, f"#{obj_id}", (label_x, label_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend original image and the colored overlay
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


def plot_bbox_overlay(image: np.ndarray, mask: np.ndarray, save_path: str) -> None:
    """
    Draws precise bounding boxes around each organoid object with its ID tagged at the top-right corner.
    """
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_img = image.copy()

    object_ids = [int(x) for x in np.unique(mask) if x > 0]
    
    for obj_id in object_ids:
        obj_mask = (mask == obj_id)
        y_indices, x_indices = np.where(obj_mask)
        
        if len(x_indices) == 0:
            continue
            
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        
        # Draw bounding box (vibrant green)
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Position label on top-right of the box
        label = f"#{obj_id}"
        cv2.putText(vis_img, label, (xmax + 2, ymin - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_img, label, (xmax, ymin - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


# =========================================================================
# Phase 2: Morphological Metric Heatmap Overlays & Spatial Metrology
# =========================================================================

def plot_metric_heatmap_overlay(image: np.ndarray, mask: np.ndarray, metric_values: List[float], metric_name: str, save_path: str, alpha: float = 0.5) -> None:
    """
    Colors each organoid dynamically using a continuous color map (JET/VIRIDIS style) based on its quantitative metric value.
    Allows rapid visual identification of structural outliers.
    """
    object_ids = sorted([int(x) for x in np.unique(mask) if x > 0])
    if not object_ids or len(metric_values) != len(object_ids):
        return

    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_img = image.copy()

    # Normalize metrics to [0, 1] range for color mapping
    min_val, max_val = min(metric_values), max(metric_values)
    val_range = max_val - min_val if max_val != min_val else 1.0

    # Initialize a clean canvas for generating colormap pixels
    heatmap_canvas = np.zeros_like(vis_img, dtype=np.uint8)
    cmap = plt.get_cmap('viridis')

    for idx, obj_id in enumerate(object_ids):
        norm_val = (metric_values[idx] - min_val) / val_range
        # Get RGB channels from the matplotlib map tuple (0.0 - 1.0) -> convert to 0-255
        rgba_color = cmap(norm_val)
        rgb_color = [int(c * 255) for c in rgba_color[:3]]
        
        heatmap_canvas[mask == obj_id] = rgb_color

    # Merge heatmap values onto original brightfields
    blended = cv2.addWeighted(heatmap_canvas, alpha, vis_img, 1 - alpha, 0)

    # Plot using Matplotlib to add a native colorbar scale legend
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(blended)
    ax.axis('off')
    ax.set_title(f"Organoid Surface Mapping: {metric_name}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label=metric_name, shrink=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_spatial_centroid_mapping(mask: np.ndarray, x_coords: List[float], y_coords: List[float], hue_metric_values: List[float], metric_name: str, save_path: str) -> None:
    """
    Plots an abstract spatial scatter map tracking localized organoid placement density distributions.
    Bubble sizes mirror size dimensions while colors illustrate the requested metric intensity.
    """
    h_max, w_max = mask.shape[:2]
    
    df = pd.DataFrame({
        'Centroid X': x_coords,
        'Centroid Y': y_coords,
        metric_name: hue_metric_values
    })

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Render spatial bubble distribution map
    scatter = ax.scatter(
        x=df['Centroid X'], 
        y=df['Centroid Y'], 
        c=df[metric_name],
        cmap='plasma', 
        s=120, 
        edgecolors='black', 
        alpha=0.85
    )
    
    ax.set_xlim(0, w_max)
    ax.set_ylim(h_max, 0)  # Invert Y axis to match matrix pixel spatial layouts cleanly
    ax.set_xlabel("Image Pixel X Dimension")
    ax.set_ylabel("Image Pixel Y Dimension")
    ax.set_title(f"Spatial Topology Density Profile ({metric_name})")
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label(metric_name)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================================
# Phase 3: Comprehensive Statistical Distributions & Correlations
# =========================================================================

def plot_metric_distribution(metric_name: str, metric_values: List[float], save_path: str) -> None:
    """
    Builds a multi-perspective descriptive data profile subplot containing:
    1. A unified Violin Plot overlaid with a Box Plot for structural median visualization.
    2. A jittered strip chart revealing absolute single object data points.
    """
    if not metric_values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Violin + Internal Box Plot
    sns.violinplot(y=metric_values, ax=axes[0], color="#9b59b6", alpha=0.4, inner=None)
    sns.boxplot(y=metric_values, ax=axes[0], width=0.15, color="#34495e", boxprops=dict(alpha=0.8))
    axes[0].set_title(f"Violin & Box Quantile Spread")
    axes[0].set_ylabel(metric_name)

    # Subplot 2: Bar / Point Stripplot spread
    sns.stripplot(y=metric_values, ax=axes[1], color="#e74c3c", size=6, jitter=0.25, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Individual Point Micro Distribution")
    axes[1].set_ylabel(metric_name)
    
    plt.suptitle(f"Statistical Profile Report for: {metric_name}", weight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_correlation_heatmap(metrics_dataframe: pd.DataFrame, save_path: str) -> None:
    """
    Generates a Pearson correlation coefficient heatmap matrix across all processed features.
    Enables exploratory profiling of morphological trade-offs.
    """
    if metrics_dataframe.empty:
        return

    # Filter out non-numeric descriptive column trackers if present
    numeric_df = metrics_dataframe.select_dtypes(include=[np.number])
    
    # Drop trivial indices or position elements to focus squarely on cross-correlations
    cols_to_drop = [col for col in ['id', 'organoid_idx', 'x', 'y'] if col in numeric_df.columns]
    if cols_to_drop:
        numeric_df = numeric_df.drop(columns=cols_to_drop)

    corr_matrix = numeric_df.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate upper-triangle mask to clear mirror redundancies
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap='RdBu_r', 
        vmin=-1.0, vmax=1.0, 
        annot=True, fmt=".2f", 
        square=True, linewidths=.5, 
        cbar_kws={"shrink": .7}, ax=ax
    )
    
    ax.set_title("Morphological Metric Covariance Matrix", weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
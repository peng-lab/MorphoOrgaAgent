"""Central state container for the organoid analysis multi-agent pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class OrganoidState:
    """Single Source of Truth shared by all agents and modules in the pipeline."""

    # ====== 1. Raw inputs and file tracking ======
    image_path: str
    output_dir: str  # Global directory for archiving results, plots, and markdown reports
    raw_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ====== 2. Structured intent written by the TU Agent ======
    target_objects: List[str] = field(default_factory=lambda: ["organoid"])
    required_metrics: List[str] = field(default_factory=list)
    required_visualizations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    report_instruction: str = ""  # Dynamic prompt passed down as a hint to the Report Agent
    requires_segmentation: bool = False

    # ====== 3. Computational results and artifact paths ======
    segmentation_masks: Optional[np.ndarray] = None  # In-memory mask array for immediate compute
    mask_path: str = ""                             # Path where the mask is persisted (.npy or .png)
    
    # Nested dictionary formatted as: {"organoid_1": {"area": 120, "roundness": 0.8}, ...}
    # Native JSON serializable for effortless multi-agent transport and LLM prompt embedding
    metric_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    csv_path: str = ""                              # Path to the backup CSV table for human inspection

    generated_plots: Dict[str, str] = field(default_factory=dict)  # Mapping of chart keys to saved file paths
    final_report: str = ""

    # ====== 4. System audit logs ======
    logs: List[str] = field(default_factory=list)

    def append_log(self, agent_name: str, message: str) -> None:
        """Append a namespaced audit log entry with timestamping."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] [{agent_name}] {message}")
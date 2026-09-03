"""Backward-compatible imports for visualization and evaluation helpers.

New code can import from :mod:`scene_visualization` or
:mod:`evaluation_utils` directly. This facade keeps the original
``lab_utils.viz_eval`` import path working for existing notebooks and scripts.
"""

from .evaluation_utils import (
    BoxIoUFunction,
    evaluate_level_results,
    evaluate_simple_iou,
)
from .scene_visualization import visualize_3d_scene_bbox_results


__all__ = [
    "BoxIoUFunction",
    "evaluate_level_results",
    "evaluate_simple_iou",
    "visualize_3d_scene_bbox_results",
]

"""Compatibility facade for the level-specific visualization helpers.

The implementations now live in :mod:`level_a_viz`, :mod:`level_c_viz`, and
:mod:`level_e_viz` so they are easier to locate. Existing imports from
``lab_utils.level_specific_viz`` continue to work unchanged.
"""

from .level_a_viz import query_and_visualize_semantic_grid, visualize_level_a_example
from .level_c_viz import visualize_level_c_example
from .level_e_viz import visualize_level_e_example
from .visualization_utils import visualize_2d_detections


__all__ = [
    "query_and_visualize_semantic_grid",
    "visualize_2d_detections",
    "visualize_level_a_example",
    "visualize_level_c_example",
    "visualize_level_e_example",
]

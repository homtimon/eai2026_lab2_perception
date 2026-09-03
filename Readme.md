# 3D Open-Vocabulary Semantic Mapping Lab

## Overview

This lab introduces 3D open-vocabulary semantic mapping using state-of-the-art vision foundation models with RGB-D sensor data input. Such semantic maps can be utilized for various downstream tasks, for example make a robot navigate to or manipulate an object of a specified type without requiring detailed information about the environment and/or objects therein. You'll work with real-world indoor scene data from the ARKitScenes dataset, learning to bridge computer vision and 3D spatial understanding.

## What You Will Learn
Through this lab, you will get familiar with:

- **Vision Foundation Models**: Working with OwlV2 (open-vocabulary object detection), SAM (segmentation), and CLIP (vision-language understanding)
- **3D Computer Vision**: Converting 2D detections to 3D representations using camera geometry, depth data, and coordinate transformations
- **Sensor Fusion**: Combining RGB images, depth maps, and camera poses to build 3D scene understanding
- **Multi-View Processing**: Aggregating information across multiple viewpoints to create robust object representations
- **Open-Vocabulary Understanding**: Moving beyond fixed object categories to query scenes with arbitrary text prompts

## Lab Structure
There are three levels of progressive difficulty: **Level E → Level C → Level A** (solve them in order).

- **Level E**: Open-vocabulary 3D object detection using OWLv2
- **Level C**: Enhanced OWLv2 detection with SAM segmentation refinement
- **Level A**: Dense semantic mapping enabling text-based 3D queries

The mapping approaches use the key ideas of ConceptGraphs (Level C) and ConceptFusion (Level A) that were discussed in the lecture.

To get an impression of what the results are roughly supposed to look like you can have a look at the images in the ExampleImages folder.

**Important**: Each group member should understand the code and be able to explain the processing pipeline. The grade for Level C requires completing Level E, and Level A requires completing all three levels.

## Implementation Details

**Jupyter Notebooks**: Each level has its own notebook (`lab2_E.ipynb`, `lab2_C.ipynb`, `lab2_A.ipynb`).

**Coding Tasks**: Look for TODO sections in the notebooks where you'll implement components of the pipeline.

**Utilities**: The `lab_utils/` folder contains helper functions for data loading and visualization - you don't need to modify these, but they will be helpful to understand the full system.

- `data_utils.py`, `ground_truth.py`: RGB-D alignment and annotations
- `geometry_utils_3d.py`, `tsdf_utils.py`: depth geometry and 3D reconstruction
- `model_loaders.py`, `detection_utils.py`, `batch_processing_utils.py`: model and frame processing
- `scene_visualization.py`, `evaluation_utils.py`: 3D display and scoring
- `level_a_viz.py`, `level_c_viz.py`, `level_e_viz.py`: level-specific examples
- `visualization_utils.py`: reusable 2D detection plots

The older `viz_eval.py` and `level_specific_viz.py` paths remain as compatibility
wrappers. You don't need to modify the utilities, but feel free to explore them
to understand the full system.

**Evaluation**: Levels E and C include automatic evaluation against ground truth with minimum IoU requirements. Level A is qualitative: students should show three natural-language queries (two concrete scene objects and one open-vocabulary or abstract prompt), with the concrete-object responses forming plausible localized regions, and explain one ambiguous or failed query (using an additional prompt if needed).

The official thresholds are fixed and checked by each notebook: Level E requires
`mIoU > 0.17`, and Level C requires `mIoU > 0.34`. Students may tune the
documented detector, SAM, and fusion parameters, but should leave the grading
threshold unchanged.

Each notebook routes evaluation through its active metric implementation, so
the displayed score exercises the submitted code. Official grading uses the
fixed level thresholds.

**References** (optional reading):
- OWLv2 (Google DeepMind): https://arxiv.org/pdf/2306.09683
- SAM (Meta AI): https://arxiv.org/pdf/2304.02643
- CLIP (OpenAI): https://arxiv.org/pdf/2103.00020
- ConceptFusion: https://arxiv.org/pdf/2302.07241
- ConceptGraphs: https://arxiv.org/pdf/2309.16650

## Installation:

Each notebook contains installation commands in the first cells. Make sure to:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv_lab2
   source venv_lab2/bin/activate  # or venv_lab2\Scripts\activate on Windows
   ```

2. Run the installation cells in each notebook

3. **Important**: After installing rerun visualization packages, close and reopen VSCode for the 3D viewer to work properly.

## Troubleshooting

**GPU Memory Issues**: If the kernel dies, reduce `max_frames` in the configuration or restart the kernel before running. (Sometimes closing and reopening vscode refreshes the cache and frees memory.)

**Visualization Issues**: If rerun 3D viewer doesn't appear:
- Close and reopen VSCode after package installation
- Allow third-party widget sources when prompted
- Manually add `jsdelivr.com` and `unpkg.com` in VSCode settings under "Jupyter: Widget Script Sources"

**Best Practice**: Restart the kernel before running the full pipeline to clear GPU memory.

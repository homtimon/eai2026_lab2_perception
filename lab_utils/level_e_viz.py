"""Visualization helpers specific to Level E open-vocabulary detection."""

from typing import Dict

import matplotlib.pyplot as plt

from .data_utils import get_frame_list, load_camera_poses, validate_and_align_frame_data
from .model_loaders import load_owlv2_model
from .visualization_utils import visualize_2d_detections


def visualize_level_e_example(config,
                              frame_index: int = 50,
                              show_depth_analysis: bool = True) -> Dict:
    """Educational visualization for Level E: show OWLv2 detection on one frame."""
    print("=" * 60)
    print("LEVEL E EXAMPLE VISUALIZATION")
    print("=" * 60)
    print(f"Demonstrating OWLv2 object detection on frame {frame_index}")
    print(f"Target classes: {config.OBJECT_CLASSES}")

    print("\n1. Loading OWLv2 model...")
    processor, model, device = load_owlv2_model()

    print("\n2. Loading frame data...")
    camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_E_CONFIG['frame_skip'])

    if frame_index >= len(frames_metadata):
        frame_index = len(frames_metadata) // 2
        print(f"Adjusted frame index to {frame_index}")

    aligned_frames = validate_and_align_frame_data(
        frames_metadata,
        camera_poses,
        config.RGB_PATH,
        config.DEPTH_PATH,
        config.INTRINSICS_PATH
    )

    if not aligned_frames or frame_index >= len(aligned_frames):
        print("No valid frames found!")
        return {}

    frame_data = aligned_frames[frame_index]
    frame_name = frame_data['frame_name']

    print(f"Selected frame: {frame_name}")

    from __main__ import detect_objects_in_frame

    print("\n3. Running OWLv2 object detection...")
    detections_2d, image = detect_objects_in_frame(
        frame_data['rgb_path'],
        config.OBJECT_CLASSES,
        processor,
        model,
        device,
        threshold=config.LEVEL_E_CONFIG['detection_threshold']
    )

    print(f"Found {len(detections_2d)} detections above threshold {config.LEVEL_E_CONFIG['detection_threshold']}")

    for i, det in enumerate(detections_2d):
        bbox = det['bbox']
        print(f"  {i+1}. {det['label']}: {det['score']:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    if not detections_2d:
        print("No detections found. Try:")
        print("- Lowering DETECTION_THRESHOLD")
        print("- Trying a different frame_index")

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No Detections Found - Frame {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        return {
            'frame_name': frame_name,
            'detections_2d': [],
            'has_detections': False
        }

    print("\n4. Visualizing 2D detection results...")
    visualize_2d_detections(image, detections_2d, show_plot=True)

    results = {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'detections_2d': detections_2d,
        'has_detections': len(detections_2d) > 0,
        'image_size': image.size,
        'detection_classes': list(set(d['label'] for d in detections_2d))
    }

    print("\n" + "="*40)
    print("EXAMPLE VISUALIZATION SUMMARY")
    print("="*40)
    print(f"Frame: {frame_name}")
    print(f"Detections: {len(detections_2d)}")
    print(f"Classes found: {results['detection_classes']}")

    return results


__all__ = ["visualize_level_e_example"]

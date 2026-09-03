"""Visualization helpers specific to Level C detection and segmentation."""

from typing import Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .data_utils import get_frame_list, load_camera_poses, validate_and_align_frame_data
from .model_loaders import load_owlv2_model, load_sam_model


def visualize_level_c_example(config, frame_index: int = 65) -> Dict:
    """Educational visualization for Level C: show OWLv2 and SAM on one frame."""
    print("=" * 60)
    print("LEVEL C EXAMPLE VISUALIZATION")
    print("=" * 60)
    print(f"Demonstrating OWLv2 + SAM segmentation on frame {frame_index}")
    print(f"Target classes: {config.OBJECT_CLASSES}")

    print("\n1. Loading models...")
    owl_processor, owl_model, device = load_owlv2_model()
    sam_model, sam_processor, _ = load_sam_model(
        model_size=config.LEVEL_C_CONFIG['sam_model_size'],
        device=device
    )

    print("\n2. Loading frame data...")
    camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_C_CONFIG['frame_skip'])

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

    from __main__ import detect_objects_in_frame, segment_with_sam_bbox

    print("\n3. Running OWLv2 object detection...")
    detections_2d, image = detect_objects_in_frame(
        frame_data['rgb_path'],
        config.OBJECT_CLASSES,
        owl_processor,
        owl_model,
        device,
        threshold=config.LEVEL_C_CONFIG['detection_threshold']
    )

    print(f"Found {len(detections_2d)} detections above threshold {config.LEVEL_C_CONFIG['detection_threshold']}")

    if not detections_2d:
        print("No detections found. Try lowering detection_threshold or different frame_index")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No Detections Found - Frame {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return {'frame_name': frame_name, 'detections_2d': [], 'segmentation_results': []}

    print("\n4. Running SAM segmentation...")
    segmentation_results = []

    for detection in detections_2d:
        mask = segment_with_sam_bbox(
            image,
            detection['bbox'],
            sam_model,
            sam_processor,
            device,
            mask_quality_threshold=config.LEVEL_C_CONFIG['sam_mask_quality_threshold']
        )

        if mask is not None:
            segmentation_results.append({
                'detection': detection,
                'mask': mask,
                'mask_area': np.sum(mask)
            })

    print(f"SAM segmentation: {len(segmentation_results)}/{len(detections_2d)} successful")

    print("\n5. Visualizing results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.imshow(image)
    ax1.set_title("OWLv2 Object Detections", fontsize=14, fontweight='bold')

    for det in detections_2d:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f"{det['label']}: {det['score']:.2f}",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
                 fontsize=10, color='white', weight='bold')
    ax1.axis('off')

    if segmentation_results:
        overlay = np.array(image).copy()
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

        for i, result in enumerate(segmentation_results):
            mask = result['mask']
            color = colors[i % len(colors)]
            overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4

        ax2.imshow(overlay)
        ax2.set_title("SAM Segmentation Masks", fontsize=14, fontweight='bold')
    else:
        ax2.imshow(image)
        ax2.set_title("No SAM Segmentations", fontsize=14, fontweight='bold')

    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    results = {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'detections_2d': detections_2d,
        'segmentation_results': segmentation_results,
        'stats': {
            'owl_detections': len(detections_2d),
            'successful_segmentations': len(segmentation_results),
            'success_rate': len(segmentation_results) / len(detections_2d) if detections_2d else 0
        }
    }

    print("\n" + "="*40)
    print("EXAMPLE VISUALIZATION SUMMARY")
    print("="*40)
    print(f"Frame: {frame_name}")
    print(f"OWLv2 detections: {results['stats']['owl_detections']}")
    print(f"SAM segmentations: {results['stats']['successful_segmentations']}")
    print(f"Success rate: {results['stats']['success_rate']:.1%}")

    return results


__all__ = ["visualize_level_c_example"]

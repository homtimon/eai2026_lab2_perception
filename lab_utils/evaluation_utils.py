"""Evaluation helpers for 3D detection results."""

from typing import Callable, Dict, List, Optional

import numpy as np


# A notebook can provide its own IoU implementation. This keeps the evaluator
# useful for student iteration while keeping the graded algorithm in the lab.
BoxIoUFunction = Callable[[np.ndarray, np.ndarray], float]


def _axis_aligned_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Backward-compatible evaluator support for callers without a callback.

    Notebook pipelines can provide their active metric through ``box_iou_fn``.
    This small fallback keeps older notebooks and ad-hoc evaluation calls
    runnable while they migrate to that interface.
    """

    try:
        corners1 = np.asarray(corners1)
        corners2 = np.asarray(corners2)
        min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
        min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)

        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        if np.any(intersection_min >= intersection_max):
            return 0.0

        intersection_volume = np.prod(intersection_max - intersection_min)
        volume1 = np.prod(max1 - min1)
        volume2 = np.prod(max2 - min2)
        union_volume = volume1 + volume2 - intersection_volume
        return float(intersection_volume / union_volume) if union_volume > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_simple_iou(detections: List[Dict],
                        ground_truth: List[Dict],
                        confidence_threshold: float = 0.3,
                        box_iou_fn: Optional[BoxIoUFunction] = None) -> Dict:
    """Compute mean IoU between detections and ground-truth boxes."""
    # Prefer a caller-provided metric so evaluation follows the active
    # notebook implementation; retain the fallback for older callers.
    box_iou_fn = box_iou_fn or _axis_aligned_iou
    detections = [d for d in detections if d['score'] >= confidence_threshold]

    if not detections or not ground_truth:
        return {
            'num_detections': len(detections),
            'num_ground_truth': len(ground_truth),
            'mean_iou': 0.0
        }

    best_ious = []

    for gt in ground_truth:
        gt_corners = np.array(gt['corners'])

        best_iou = 0.0
        for det in detections:
            if det['label'] != gt['label']:
                continue

            det_corners = np.array(det['bbox_3d_world'])
            iou = box_iou_fn(gt_corners, det_corners)
            best_iou = max(best_iou, float(iou))

        best_ious.append(best_iou)

    return {
        'num_detections': len(detections),
        'num_ground_truth': len(ground_truth),
        'mean_iou': np.mean(best_ious),
        'median_iou': np.median(best_ious),
        'matched_gt': sum(1 for iou in best_ious if iou > 0.1)
    }


def evaluate_level_results(detections: List[Dict],
                           gt_annotations: List[Dict],
                           level_name: str,
                           confidence_threshold: float = 0.3,
                           required_miou: float = 0.14,
                           box_iou_fn: Optional[BoxIoUFunction] = None) -> Dict:
    """Evaluate detections and print the level's pass/fail summary."""
    print("\n" + "="*50)
    print(f"{level_name.upper()} EVALUATION")
    print("="*50)

    if not detections or not gt_annotations:
        print("Cannot perform evaluation - missing detections or ground truth")
        return {
            'level': level_name,
            'num_detections': len(detections) if detections else 0,
            'num_ground_truth': len(gt_annotations) if gt_annotations else 0,
            'mean_iou': 0.0,
            'passed': False,
            'error': 'Missing data'
        }

    eval_results = evaluate_simple_iou(
        detections,
        gt_annotations,
        confidence_threshold,
        box_iou_fn=box_iou_fn,
    )

    print(f"Detections: {eval_results['num_detections']}")
    print(f"Ground Truth: {eval_results['num_ground_truth']}")
    print(f"Mean IoU: {eval_results['mean_iou']:.3f}")
    print(f"Median IoU: {eval_results['median_iou']:.3f}")
    print(f"GT Matched (IoU>0.1): {eval_results['matched_gt']}/{eval_results['num_ground_truth']}")

    passed = eval_results['mean_iou'] > required_miou

    if passed:
        print(f"✓ PASSED: mIoU {eval_results['mean_iou']:.3f} > {required_miou} requirement")
    else:
        print(f"✗ NOT PASSED: mIoU {eval_results['mean_iou']:.3f} < {required_miou} requirement")

    eval_results.update({
        'level': level_name,
        'passed': passed,
        'required_miou': required_miou,
        'confidence_threshold': confidence_threshold
    })

    return eval_results


__all__ = [
    "BoxIoUFunction",
    "evaluate_simple_iou",
    "evaluate_level_results",
]

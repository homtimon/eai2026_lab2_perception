"""3D scene visualization helpers shared by the lab notebooks."""

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from typing import Dict, List


def visualize_3d_scene_bbox_results(point_cloud: o3d.geometry.PointCloud = None,
                                    detections_3d: List[Dict] = None,
                                    raw_detections_3d: List[Dict] = None,
                                    gt_annotations: List[Dict] = None,
                                    gt_mesh: o3d.geometry.PointCloud = None,
                                    show_ground_truth: bool = True,
                                    show_gt_mesh: bool = False,
                                    show_object_pointclouds: bool = False,
                                    show_raw_detections: bool = False,
                                    title: str = "3D Scene Visualization",
                                    config=None) -> None:
    """Render point clouds, detections, and ground-truth boxes in Rerun."""
    width = config.RERUN_WIDTH if config else 1600
    height = config.RERUN_HEIGHT if config else 800

    rr.init("3d_scene_analysis")
    print(f"Visualizing: {title}")

    if point_cloud is not None and len(point_cloud.points) > 0:
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

        if colors is not None and len(colors) > 0:
            colors_uint8 = (colors * 255).astype(np.uint8)
            rr.log("world/environment/tsdf_pointcloud",
                   rr.Points3D(points, colors=colors_uint8, radii=0.008))
        else:
            z_values = points[:, 2]
            z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())

            height_colors = np.zeros((len(points), 3), dtype=np.uint8)
            height_colors[:, 0] = (z_normalized * 255).astype(np.uint8)
            height_colors[:, 1] = ((1 - np.abs(z_normalized - 0.5) * 2) * 255).astype(np.uint8)
            height_colors[:, 2] = ((1 - z_normalized) * 255).astype(np.uint8)

            rr.log("world/environment/tsdf_pointcloud",
                   rr.Points3D(points, colors=height_colors, radii=0.008))

    if show_raw_detections and raw_detections_3d and len(raw_detections_3d) > 0:
        raw_detection_colors = {
            'chair': [255, 100, 100],
            'table': [100, 255, 100],
            'sofa': [100, 100, 255],
            'bed': [255, 200, 100],
            'shelf': [255, 150, 200],
            'pillow': [200, 100, 255],
            'window': [100, 255, 255],
            'basketball': [255, 255, 100]
        }

        raw_centers = []
        raw_sizes = []
        raw_colors = []
        raw_labels = []

        for i, detection in enumerate(raw_detections_3d):
            bbox_3d = np.array(detection['bbox_3d_world'])
            label = detection['label']
            score = detection['score']
            color = raw_detection_colors.get(label, [200, 200, 200])

            min_bounds = np.min(bbox_3d, axis=0)
            max_bounds = np.max(bbox_3d, axis=0)
            raw_centers.append((min_bounds + max_bounds) / 2)
            raw_sizes.append(max_bounds - min_bounds)
            raw_colors.append(color)
            raw_labels.append(f"Raw_{label}_{i} ({score:.2f})")

        if raw_centers:
            rr.log(
                "world/detections/raw_bboxes",
                rr.Boxes3D(
                    centers=raw_centers,
                    sizes=raw_sizes,
                    colors=raw_colors,
                    labels=raw_labels
                )
            )

    if detections_3d and len(detections_3d) > 0:
        detection_class_colors = {
            'chair': [255, 0, 0],
            'table': [0, 255, 0],
            'sofa': [0, 0, 255],
            'bed': [255, 165, 0],
            'stool': [128, 0, 128],
            'cabinet': [165, 42, 42],
            'shelf': [255, 192, 203],
            'pillow': [128, 0, 128],
            'window': [0, 255, 255],
            'basketball': [255, 105, 180]
        }

        detection_centers = []
        detection_sizes = []
        detection_colors = []
        detection_labels = []

        for i, detection in enumerate(detections_3d):
            bbox_3d = np.array(detection['bbox_3d_world'])
            label = detection['label']
            score = detection['score']
            color = detection_class_colors.get(label, [255, 255, 255])

            min_bounds = np.min(bbox_3d, axis=0)
            max_bounds = np.max(bbox_3d, axis=0)
            detection_centers.append((min_bounds + max_bounds) / 2)
            detection_sizes.append(max_bounds - min_bounds)
            detection_colors.append(color)

            label_text = f"Merged_{label} ({score:.2f})"
            if 'merge_info' in detection:
                merge_count = detection['merge_info']['num_detections_merged']
                label_text += f" [x{merge_count}]"
            elif 'num_observations' in detection:
                label_text += f" [{detection['num_observations']} obs]"
            detection_labels.append(label_text)

            if show_object_pointclouds and 'pointcloud' in detection:
                pc_points = detection['pointcloud']
                pc_colors = detection.get('colors')

                if pc_colors is not None:
                    if pc_colors.dtype != np.uint8:
                        pc_colors = (pc_colors * 255).astype(np.uint8)
                else:
                    pc_colors = np.array([color] * len(pc_points), dtype=np.uint8)

                rr.log(
                    f"world/detections/object_{label}_{i}/pointcloud",
                    rr.Points3D(pc_points, colors=pc_colors, radii=0.01)
                )

        if detection_centers:
            rr.log(
                "world/detections/merged_bboxes",
                rr.Boxes3D(
                    centers=detection_centers,
                    sizes=detection_sizes,
                    colors=detection_colors,
                    labels=detection_labels
                )
            )

    if show_ground_truth and (gt_annotations or gt_mesh):
        gt_class_colors = {
            'chair': [255, 100, 100],
            'table': [100, 255, 100],
            'sofa': [100, 100, 255],
            'bed': [255, 200, 100],
            'cabinet': [200, 150, 100],
            'shelf': [255, 150, 200],
            'door': [150, 150, 150],
        }

        if show_gt_mesh and gt_mesh is not None and len(gt_mesh.points) > 0:
            points = np.asarray(gt_mesh.points)

            if gt_mesh.has_colors():
                colors = np.asarray(gt_mesh.colors)
                colors_uint8 = (colors * 255).astype(np.uint8)
                rr.log("ground_truth/mesh", rr.Points3D(points, colors=colors_uint8, radii=0.014))
            else:
                gt_mesh_color = np.full((len(points), 3), [255, 215, 0], dtype=np.uint8)
                rr.log("ground_truth/mesh", rr.Points3D(points, colors=gt_mesh_color, radii=0.014))

        if gt_annotations:
            annotations_by_class = {}
            for ann in gt_annotations:
                class_name = ann['label']
                if class_name not in annotations_by_class:
                    annotations_by_class[class_name] = []
                annotations_by_class[class_name].append(ann)

            for class_name, class_annotations in annotations_by_class.items():
                class_centers = []
                class_sizes = []
                class_color = gt_class_colors.get(class_name, [200, 200, 200])
                class_colors = []
                class_labels = []

                for i, ann in enumerate(class_annotations):
                    corners = np.array(ann['corners'])
                    min_bounds = np.min(corners, axis=0)
                    max_bounds = np.max(corners, axis=0)
                    class_centers.append((min_bounds + max_bounds) / 2)
                    class_sizes.append(max_bounds - min_bounds)
                    class_colors.append(class_color)
                    class_labels.append(f"GT_{class_name}_{i}")

                if class_centers:
                    rr.log(
                        f"ground_truth/bboxes/{class_name}",
                        rr.Boxes3D(
                            centers=class_centers,
                            sizes=class_sizes,
                            colors=class_colors,
                            labels=class_labels
                        )
                    )

    frame_size = 1.0
    rr.log(
        "world/coordinate_frame",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[frame_size, 0, 0], [0, frame_size, 0], [0, 0, frame_size]],
            colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]],
            labels=["X (1m)", "Y (1m)", "Z (1m)"]
        )
    )

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint)

    rr.notebook_show(width=width, height=height)
    print("3D visualization complete")


__all__ = ["visualize_3d_scene_bbox_results"]

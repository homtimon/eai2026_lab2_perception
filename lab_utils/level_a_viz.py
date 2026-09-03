"""Visualization helpers specific to Level A semantic mapping."""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import torch

from .data_utils import get_frame_list
from .model_loaders import load_clip_model, load_sam_model


def visualize_level_a_example(config, frame_index: int = 29) -> Dict:
    """Level A visualization: SAM proposals with CLIP semantic similarity overlays."""
    print("=" * 50)
    print("LEVEL A: SEMANTIC SEGMENT ANALYSIS")
    print("=" * 50)
    print(f"Processing frame {frame_index} with text queries: 'pillow' vs 'sofa'")

    print("Loading models...")
    sam_model, sam_processor, device = load_sam_model(model_size='base')
    clip_model, clip_processor, _ = load_clip_model(device=device)

    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_A_CONFIG['frame_skip'])
    if frame_index >= len(frames_metadata):
        frame_index = len(frames_metadata) // 2
        print(f"Adjusted frame index to {frame_index}")

    frame_name = frames_metadata[frame_index]['filename']
    rgb_path = os.path.join(config.RGB_PATH, frame_name)

    if not os.path.exists(rgb_path):
        print(f"Frame not found: {rgb_path}")
        return {}

    image = Image.open(rgb_path).convert("RGB")
    print(f"Frame: {frame_name}, size: {image.size}")

    try:
        from __main__ import generate_sam_proposals, extract_clip_features_from_segment
    except ImportError:
        print("ERROR: Could not import required functions from notebook.")
        print("Please ensure generate_sam_proposals and extract_clip_features_from_segment are defined in the notebook.")
        return {}

    proposals = generate_sam_proposals(
        image,
        sam_model,
        sam_processor,
        device,
        grid_size=config.LEVEL_A_CONFIG['grid_size'],
        mask_quality_threshold=config.LEVEL_A_CONFIG['sam_mask_quality_threshold']
    )

    print(f"Generated {len(proposals)} proposals above mask_quality threshold")

    if not proposals:
        print("No proposals generated - try lowering sam_mask_quality_threshold")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No SAM Proposals - {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return {'frame_name': frame_name, 'proposals': [], 'has_proposals': False}

    print("Extracting CLIP features...")
    proposal_features = []
    for proposal in proposals:
        features = extract_clip_features_from_segment(
            image,
            proposal['mask'],
            clip_model,
            clip_processor,
            device,
            padding_ratio=config.LEVEL_A_CONFIG['padding_ratio_image_crops']
        )
        if features is not None:
            proposal_features.append({
                'proposal': proposal,
                'features': features
            })

    print(f"Successfully extracted features for {len(proposal_features)}/{len(proposals)} proposals")

    if not proposal_features:
        print("No CLIP features extracted - check segment quality")
        return {'frame_name': frame_name, 'proposals': proposals, 'proposal_features': []}

    text_queries = ['pillow', 'sofa']

    print("Computing text embeddings...")
    text_embeddings = {}
    for query in text_queries:
        inputs = clip_processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_embeddings[query] = text_features.cpu().numpy().squeeze()

    similarities = {}
    for query in text_queries:
        query_similarities = []
        for pf in proposal_features:
            similarity = np.dot(text_embeddings[query], pf['features'])
            query_similarities.append(similarity)
        similarities[query] = np.array(query_similarities)

    image_gray = image.convert('L')
    image_gray_rgb = np.stack([np.array(image_gray)] * 3, axis=-1)

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    viridis_cmap = plt.get_cmap('viridis')

    for idx, query in enumerate(text_queries):
        ax = ax1 if idx == 0 else ax2
        overlay = image_gray_rgb.astype(np.float32).copy()

        query_sims = similarities[query]
        if len(query_sims) > 0:
            sim_min, sim_max = query_sims.min(), query_sims.max()
            if sim_max > sim_min:
                sim_normalized = (query_sims - sim_min) / (sim_max - sim_min)
            else:
                sim_normalized = np.ones_like(query_sims) * 0.5

            for i, pf in enumerate(proposal_features):
                mask = pf['proposal']['mask']
                similarity_score = sim_normalized[i]
                color_rgba = viridis_cmap(similarity_score)
                color_rgb = np.array(color_rgba[:3]) * 255
                alpha = 0.7
                overlay[mask] = overlay[mask] * (1 - alpha) + color_rgb * alpha

        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"Semantic Response: '{query}'\n"
                     f"Similarity range: [{query_sims.min():.3f}, {query_sims.max():.3f}]",
                     fontsize=14, fontweight='bold')
        ax.axis('off')

    cax = fig.add_subplot(gs[1, :])
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Semantic Similarity (Normalized)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    cbar.ax.text(0.1, -0.8, 'Low\n(Dark Purple)', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.9, -0.8, 'High\n(Yellow)', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\nSemantic Similarity Analysis:")
    for query in text_queries:
        sims = similarities[query]
        print(f"  '{query}': mean={sims.mean():.3f}, max={sims.max():.3f}, std={sims.std():.3f}")

    return {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'proposals': proposals,
        'proposal_features': proposal_features,
        'similarities': similarities,
        'text_queries': text_queries,
        'stats': {
            'total_proposals': len(proposals),
            'successful_features': len(proposal_features),
            'queries_tested': len(text_queries)
        }
    }


def query_and_visualize_semantic_grid(level_a_results: Dict,
                                      text_query: str,
                                      environment_pcd: o3d.geometry.PointCloud = None,
                                      config=None) -> None:
    """Query the semantic voxel grid with text and visualize results."""
    print(f"\nQuerying semantic grid with: '{text_query}'")

    voxel_grid = level_a_results['voxel_grid']
    clip_model = level_a_results['clip_model']
    clip_processor = level_a_results['clip_processor']
    device = level_a_results['device']

    voxel_centers, similarities = voxel_grid.query_text(
        text_query,
        clip_model,
        clip_processor,
        device
    )

    if len(voxel_centers) == 0:
        print("No voxels to visualize!")
        return

    print("Query results:")
    print(f"  Voxels with data: {len(voxel_centers)}")
    print(f"  Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    print(f"  Mean similarity: {similarities.mean():.3f}")

    sim_normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-6)

    rr.init("level_a_semantic_mapping")

    width = config.RERUN_WIDTH if config else 1600
    height = config.RERUN_HEIGHT if config else 800

    rr.log("world", rr.Clear(recursive=True))

    if environment_pcd is not None and len(environment_pcd.points) > 0:
        points = np.asarray(environment_pcd.points)
        colors = np.full((len(points), 3), [200, 200, 200], dtype=np.uint8)
        rr.log("world/environment",
               rr.Points3D(points, colors=colors, radii=0.005))

    colormap = plt.get_cmap('viridis')

    voxel_colors = []
    voxel_radii = []

    for sim in sim_normalized:
        color_rgba = colormap(sim)
        color_rgb = (np.array(color_rgba[:3]) * 255).astype(np.uint8)
        voxel_colors.append(color_rgb)
        radius = 0.02 + sim * 0.03
        voxel_radii.append(radius)

    voxel_colors = np.array(voxel_colors)
    voxel_radii = np.array(voxel_radii)

    rr.log("world/semantic_voxels",
           rr.Points3D(voxel_centers, colors=voxel_colors, radii=voxel_radii))

    rr.log("world/query_text",
           rr.TextDocument(f"Query: '{text_query}'\n"
                           f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]\n"
                           f"Occupied voxels: {len(voxel_centers)}"))

    rr.log("world/coordinate_frame",
           rr.Arrows3D(
               origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
               labels=["X", "Y", "Z"]
           ))

    print("\nViridis colormap:")
    print("  Dark purple/blue: Low similarity")
    print("  Green: Medium similarity")
    print("  Yellow: High similarity")

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint)

    rr.notebook_show(width=width, height=height)


__all__ = ["visualize_level_a_example", "query_and_visualize_semantic_grid"]

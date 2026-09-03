"""
2D Object Detection Utilities

This module contains functions for running OWLv2 open-vocabulary detection
and post-processing its raw outputs into a simple detection format.
"""

import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
import torch


def process_owlv2_outputs(outputs, processor, target_sizes, text_queries, threshold=0.1):
    """Process raw OWLv2 model outputs into detection format."""
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        detections.append({
            'bbox': box.tolist(),
            'score': float(score),
            'label': text_queries[label],
            'label_id': int(label)
        })
    
    return detections


def detect_objects_in_frame(image_path: str, 
                           text_queries: List[str],
                           processor, 
                           model, 
                           device: str,
                           threshold: float = 0.1) -> Tuple[List[Dict], Image.Image]:
    """Run OWLv2 object detection."""
    
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([[orig_height, orig_width]]).to(device)
    
    detections = process_owlv2_outputs(outputs, processor, target_sizes, text_queries, threshold)
    
    # Clip to image bounds
    for detection in detections:
        bbox = detection['bbox']
        bbox[0] = max(0, min(bbox[0], orig_width))   # x1
        bbox[1] = max(0, min(bbox[1], orig_height))  # y1
        bbox[2] = max(0, min(bbox[2], orig_width))   # x2
        bbox[3] = max(0, min(bbox[3], orig_height))  # y2
        detection['bbox'] = bbox
    
    return detections, image

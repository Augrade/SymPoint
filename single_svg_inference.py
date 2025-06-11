#!/usr/bin/env python3
"""
Single SVG file inference script for SymPoint model.
Reads an SVG file, preprocesses it, performs semantic and instance segmentation predictions,
and maps the instance predictions back to the original SVG paths.

Key features:
- Performs semantic and instance segmentation on SVG files
- Maps instance segmentation masks to original SVG path elements
- Groups SVG paths by detected instances
- Outputs a grouped SVG file with color-coded instances
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parse_svg import parse_svg
from svgnet.model.svgnet import SVGNet
# Remove unused imports


def parse_svg_file(svg_path):
    """Parse SVG file using the existing parser."""
    print(f"Parsing SVG file: {svg_path}")
    
    # Parse SVG to get command data
    data = parse_svg(svg_path)
    
    if data is None:
        raise ValueError(f"Failed to parse SVG file: {svg_path}")
    
    return data


def preprocess_svg_data(svg_data, norm_type='mean', min_points=2048):
    """Convert parsed SVG data to model input format."""
    
    # Extract data
    commands = np.array(svg_data['commands'])
    svg_args = svg_data['args']  # Keep as list first to check structure
    lengths = np.array(svg_data['lengths'])
    semantic_ids = np.array(svg_data['semanticIds'])
    instance_ids = np.array(svg_data['instanceIds'])
    
    # Check if args is a list of lists and flatten if needed
    if isinstance(svg_args[0], list):
        svg_args = [item for sublist in svg_args for item in sublist]
    svg_args = np.array(svg_args, dtype=np.float32)
    
    # Number of elements
    num_elements = len(commands)
    
    # Create coordinate array (4 points per element, 3D coordinates)
    coords = np.zeros((num_elements * 4, 3), dtype=np.float32)
    for i in range(num_elements):
        # Extract 4 points (x,y pairs) for this element
        for j in range(4):
            coords[i * 4 + j, 0] = svg_args[i * 8 + j * 2]      # x
            coords[i * 4 + j, 1] = svg_args[i * 8 + j * 2 + 1]  # y
            coords[i * 4 + j, 2] = 0                         # z (always 0)
    
    # Normalize coordinates
    coords[:, :2] = coords[:, :2] / 140.0
    
    # Apply normalization
    if norm_type == 'mean':
        coords = coords - coords.mean(axis=0)
    elif norm_type == 'min':
        coords = coords - coords.min(axis=0)
    
    # Create features array (6D: arc_angle, normalized_length, command_one_hot)
    features = np.zeros((num_elements * 4, 6), dtype=np.float32)
    
    for i in range(num_elements):
        # Arc angle feature
        for j in range(4):
            idx = i * 4 + j
            x, y = coords[idx, 0], coords[idx, 1]
            features[idx, 0] = np.arctan2(y, x) / np.pi
        
        # Normalized length (same for all 4 points of an element)
        norm_length = np.clip(lengths[i], 0, 140) / 140.0
        features[i * 4:(i + 1) * 4, 1] = norm_length
        
        # Command type one-hot encoding
        cmd_type = commands[i]
        features[i * 4:(i + 1) * 4, 2 + cmd_type] = 1.0
    
    # Combine coordinates and features
    point_features = np.concatenate([coords, features], axis=1)
    
    # Expand semantic and instance IDs (4 points per element)
    semantic_labels = np.repeat(semantic_ids, 4)
    instance_labels = np.repeat(instance_ids, 4)
    
    # Convert background semantic ID from 36 to 35
    semantic_labels[semantic_labels == 36] = 35
    
    # Set instance ID to -1 for background/stuff classes
    instance_labels[semantic_labels >= 35] = -1
    
    # Pad to minimum points if necessary
    num_points = point_features.shape[0]
    if num_points < min_points:
        pad_size = min_points - num_points
        point_features = np.pad(point_features, ((0, pad_size), (0, 0)), mode='constant')
        semantic_labels = np.pad(semantic_labels, (0, pad_size), mode='constant', constant_values=35)
        instance_labels = np.pad(instance_labels, (0, pad_size), mode='constant', constant_values=-1)
    
    # Sample if too many points
    if num_points > min_points:
        # Use random sampling to match training behavior
        sample_idx = np.random.choice(num_points, min_points, replace=False)
        
        point_features = point_features[sample_idx]
        semantic_labels = semantic_labels[sample_idx]
        instance_labels = instance_labels[sample_idx]
        num_points = min_points
    
    # Update num_points to reflect actual tensor size after padding/sampling
    actual_num_points = point_features.shape[0]
    
    return {
        'points': torch.from_numpy(point_features).float(),
        'semantic_labels': torch.from_numpy(semantic_labels).long(),
        'instance_labels': torch.from_numpy(instance_labels).long(),
        'num_points': actual_num_points,  # Use actual tensor size
        'original_data': svg_data
    }


def load_model(checkpoint_path, config_dict, device='cuda'):
    """Load the trained model."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Convert config dict to object with attributes
    class Config:
        def __init__(self, d):
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
    
    # Create config object from model section
    model_config = config_dict.get('model', {})
    cfg = Config(model_config)
    
    # Ensure required attributes exist
    if not hasattr(cfg, 'in_channels'):
        cfg.in_channels = 9
    if not hasattr(cfg, 'semantic_classes'):
        cfg.semantic_classes = 35
    
    # Create model
    model = SVGNet(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict from checkpoint
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded successfully (epoch: {checkpoint.get('epoch', 'unknown')})")
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        if device == 'cuda':
            print("Warning: CUDA requested but not available, using CPU")
    
    model.eval()
    
    return model


def perform_inference(model, data, device='cuda'):
    """Run inference on preprocessed data."""
    print("Running inference...")
    
    # Determine device
    if device == 'cuda' and torch.cuda.is_available():
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device('cpu')
    
    print(f"Using device: {device_obj}")
    
    # Prepare batch in the format expected by the model
    coords = data['points'][:, :3].to(device_obj)  # [N, 3]
    feats = data['points'][:, 3:].to(device_obj)   # [N, 6]
    semantic_labels = data['semantic_labels'].to(device_obj)  # [N]
    instance_labels = data['instance_labels'].to(device_obj)  # [N]
    
    print(f"Data shapes - coords: {coords.shape}, feats: {feats.shape}")
    
    # Validate data ranges to prevent CUDA index errors
    print(f"Coords range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"Features range: [{feats.min():.3f}, {feats.max():.3f}]")
    print(f"Semantic labels range: [{semantic_labels.min()}, {semantic_labels.max()}]")
    print(f"Instance labels range: [{instance_labels.min()}, {instance_labels.max()}]")
    
    # Clamp values to safe ranges
    coords = torch.clamp(coords, min=-10.0, max=10.0)
    feats = torch.clamp(feats, min=-10.0, max=10.0)
    semantic_labels = torch.clamp(semantic_labels, min=0, max=35)
    instance_labels = torch.clamp(instance_labels, min=-1, max=1000)
    
    # Stack semantic and instance labels - model expects [N, 2] format
    labels = torch.stack([semantic_labels, instance_labels], dim=-1)  # [N, 2]
    
    # Create offset for single batch - this indicates where the batch ends
    actual_points = coords.shape[0]  # This is 2048 after padding/sampling
    offsets = torch.IntTensor([actual_points]).to(device_obj)
    
    # Lengths tensor - create zeros for all points (not used in inference)
    lengths = torch.zeros(actual_points).to(device_obj)
    
    print(f"Batch info - offsets: {offsets}, actual_points: {actual_points}")
    print(f"Labels shape: {labels.shape}")
    
    # The model expects: (coords, feats, labels, offsets, lengths)
    # But the forward method unpacks as: coords, feats, semantic_labels, offsets, lengths
    # So we need to pass labels in the semantic_labels position
    batch = (coords, feats, labels, offsets, lengths)
    
    try:
        with torch.no_grad():
            # Clear CUDA cache before inference
            if device_obj.type == 'cuda':
                torch.cuda.empty_cache()
            
            print("Starting model forward pass...")
            outputs = model(batch, return_loss=False)
            print("Model forward pass completed successfully")
            
    except RuntimeError as e:
        error_msg = str(e)
        print(f"RuntimeError during model forward: {error_msg}")
        
        if "out of bounds" in error_msg or "index" in error_msg:
            print("Index error detected - data ranges may be invalid")
        elif "overflow" in error_msg:
            print("Overflow error detected - tensor size may be too large")
            print(f"Current tensor sizes - coords: {coords.shape}, feats: {feats.shape}, labels: {labels.shape}")
        elif "memory" in error_msg.lower():
            print("Memory error detected - try reducing number of points")
            
        raise RuntimeError(f"Model inference failed: {error_msg}")
    except Exception as e:
        print(f"Unexpected error during inference: {type(e).__name__}: {str(e)}")
        raise e
    
    print(f"Output keys: {list(outputs.keys())}")
    
    # Extract predictions - model returns different format
    semantic_scores = outputs['semantic_scores']  # [num_points, num_classes]
    semantic_labels = outputs['semantic_labels']  # [num_points]
    instances_data = outputs['instances'] if 'instances' in outputs else None
    
    # Get semantic predictions (already have per-point scores)
    semantic_pred = semantic_labels.cpu().numpy()
    
    # Get instance predictions if available
    instances = []
    if instances_data is not None:
        # Check if instances_data is a tensor or list
        if isinstance(instances_data, list):
            # If it's already a list of instances, process each one
            for inst in instances_data:
                # Convert numpy arrays to lists for JSON serialization
                processed_inst = {}
                for key, value in inst.items():
                    if isinstance(value, np.ndarray):
                        processed_inst[key] = value.tolist()
                    elif torch.is_tensor(value):
                        processed_inst[key] = value.cpu().numpy().tolist()
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        processed_inst[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        processed_inst[key] = float(value)
                    else:
                        processed_inst[key] = value
                instances.append(processed_inst)
        elif torch.is_tensor(instances_data):
            # Process instance data - extract individual instances
            for inst_id in torch.unique(instances_data):
                if inst_id < 0:  # Skip background/invalid instances
                    continue
                mask = (instances_data == inst_id)
                if mask.sum() > 0:
                    # Get the most common semantic label for this instance
                    inst_semantic_labels = semantic_labels[mask]
                    label = inst_semantic_labels.mode()[0].item()
                    
                    instance = {
                        'label': int(label),
                        'mask': mask.cpu().numpy().tolist(),
                        'num_points': int(mask.sum())
                    }
                    instances.append(instance)
    
    return {
        'semantic_predictions': semantic_pred.tolist(),
        'instances': instances,
        'num_instances': len(instances),
        'coords': coords.cpu().numpy(),  # Keep original coords for visualization (numpy array for viz)
        'semantic_labels': semantic_labels.cpu().numpy()  # Keep as numpy for visualization
    }


def map_instances_to_svg_paths(results, svg_data):
    """Map instance segmentation masks to original SVG path indices.
    
    Returns a dictionary mapping instance_id to list of SVG element indices.
    """
    # Get the original number of elements (before point expansion)
    num_elements = len(svg_data['commands'])
    
    # Each element generates 4 points in preprocessing
    # So we need to map point indices back to element indices
    instance_to_elements = {}
    
    for idx, inst in enumerate(results['instances']):
        # Handle both 'mask' and 'masks' keys
        mask_data = inst.get('mask') or inst.get('masks')
        if mask_data is not None:
            if isinstance(mask_data, list):
                mask = np.array(mask_data, dtype=bool)
            else:
                mask = mask_data.astype(bool)
            
            # Find which elements this instance covers
            element_indices = set()
            
            # Get indices where mask is True
            point_indices = np.where(mask)[0]
            
            for point_idx in point_indices:
                # Each element has 4 points, so element_idx = point_idx // 4
                element_idx = point_idx // 4
                if element_idx < num_elements:  # Ensure we're within valid range
                    element_indices.add(element_idx)
                else:
                    # This point is in the padded region
                    continue
            
            if element_indices:
                # Convert to Python integers to avoid JSON serialization issues
                instance_to_elements[idx] = [int(x) for x in sorted(list(element_indices))]
    
    return instance_to_elements


def group_svg_paths_by_instance(svg_file_path, instance_to_elements, output_svg_path=None):
    """Group SVG paths by instance and optionally save as a new SVG with groups.
    
    Args:
        svg_file_path: Path to original SVG file
        instance_to_elements: Dictionary mapping instance_id to list of element indices
        output_svg_path: Optional path to save the grouped SVG
    
    Returns:
        Dictionary with instance information including grouped paths
    """
    import xml.etree.ElementTree as ET
    
    # Parse SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    
    # Get SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    if root.tag.startswith('{'):
        ns['svg'] = root.tag.split('}')[0][1:]
    
    # Create a list to store all drawable elements in order
    all_elements = []
    element_to_original = {}  # Map from our index to original element
    
    # Collect all drawable elements (paths, circles, ellipses, etc.)
    idx = 0
    for elem in root.iter():
        if elem.tag.endswith(('path', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'rect')):
            all_elements.append(elem)
            element_to_original[idx] = elem
            idx += 1
    
    # Group paths by instance
    instance_groups = {}
    
    for instance_id, element_indices in instance_to_elements.items():
        instance_groups[instance_id] = {
            'elements': [],
            'element_indices': element_indices,
            'paths': []
        }
        
        for elem_idx in element_indices:
            if elem_idx < len(all_elements):
                elem = all_elements[elem_idx]
                instance_groups[instance_id]['elements'].append(elem)
                
                # Extract path data or shape info
                if elem.tag.endswith('path'):
                    instance_groups[instance_id]['paths'].append({
                        'type': 'path',
                        'd': elem.get('d', ''),
                        'element': elem
                    })
                elif elem.tag.endswith('circle'):
                    instance_groups[instance_id]['paths'].append({
                        'type': 'circle',
                        'cx': elem.get('cx', '0'),
                        'cy': elem.get('cy', '0'),
                        'r': elem.get('r', '0'),
                        'element': elem
                    })
                elif elem.tag.endswith('ellipse'):
                    instance_groups[instance_id]['paths'].append({
                        'type': 'ellipse',
                        'cx': elem.get('cx', '0'),
                        'cy': elem.get('cy', '0'),
                        'rx': elem.get('rx', '0'),
                        'ry': elem.get('ry', '0'),
                        'element': elem
                    })
                elif elem.tag.endswith('rect'):
                    instance_groups[instance_id]['paths'].append({
                        'type': 'rect',
                        'x': elem.get('x', '0'),
                        'y': elem.get('y', '0'),
                        'width': elem.get('width', '0'),
                        'height': elem.get('height', '0'),
                        'element': elem
                    })
                elif elem.tag.endswith('line'):
                    instance_groups[instance_id]['paths'].append({
                        'type': 'line',
                        'x1': elem.get('x1', '0'),
                        'y1': elem.get('y1', '0'),
                        'x2': elem.get('x2', '0'),
                        'y2': elem.get('y2', '0'),
                        'element': elem
                    })
    
    # Optionally save grouped SVG
    if output_svg_path:
        # Create a new SVG with groups
        new_root = ET.Element('svg')
        # Copy attributes from original root
        for attr, value in root.attrib.items():
            new_root.set(attr, value)
        
        # Add namespace if needed
        if ns['svg'] != 'http://www.w3.org/2000/svg':
            new_root.set('xmlns', ns['svg'])
        
        # Create groups for each instance
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                  '#FFA500', '#800080', '#FFC0CB', '#A52A2A']
        
        for idx, (instance_id, group_data) in enumerate(instance_groups.items()):
            g = ET.SubElement(new_root, 'g')
            g.set('id', f'instance_{instance_id}')
            g.set('stroke', colors[idx % len(colors)])
            g.set('fill', 'none')
            g.set('stroke-width', '2')
            
            # Add all elements of this instance to the group
            for elem in group_data['elements']:
                # Clone the element
                new_elem = ET.SubElement(g, elem.tag)
                for attr, value in elem.attrib.items():
                    if attr not in ['stroke', 'fill', 'stroke-width']:  # Override style
                        new_elem.set(attr, value)
        
        # Write the new SVG
        tree = ET.ElementTree(new_root)
        tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)
        print(f"Grouped SVG saved to: {output_svg_path}")
    
    return instance_groups


def visualize_results(results, svg_file_path, output_path=None):
    """Visualize the inference results with colored point clouds"""
    
    # Class names mapping
    class_names = {
        0: 'single door', 1: 'double door', 2: 'sliding door', 3: 'folding door',
        4: 'revolving door', 5: 'rolling door', 6: 'window', 7: 'bay window',
        8: 'blind window', 9: 'opening symbol', 10: 'sofa', 11: 'bed',
        12: 'chair', 13: 'table', 14: 'TV cabinet', 15: 'Wardrobe',
        16: 'cabinet', 17: 'gas stove', 18: 'sink', 19: 'refrigerator',
        20: 'airconditioner', 21: 'bath', 22: 'bath tub', 23: 'washing machine',
        24: 'squat toilet', 25: 'urinal', 26: 'toilet', 27: 'stairs',
        28: 'elevator', 29: 'escalator', 30: 'row chairs', 31: 'parking spot',
        32: 'wall', 33: 'curtain wall', 34: 'railing', 35: 'background'
    }
    
    # Generate colors for each class
    np.random.seed(42)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.tab20b(np.linspace(0, 1, 16))
    all_colors = np.vstack([colors, colors2])
    
    # Get coordinates and labels
    coords = results['coords']
    labels = results['semantic_labels']
    
    # Create figure with subplots - now with 3 columns
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 0: Original SVG
    ax0.set_title('Original SVG', fontsize=14)
    
    # Parse and render the SVG file
    ax0.set_title('Original SVG', fontsize=14)
    
    try:
        import xml.etree.ElementTree as ET
        from svgpathtools import parse_path
        
        # Parse SVG file
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        
        # Get SVG namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        if root.tag.startswith('{'):
            ns['svg'] = root.tag.split('}')[0][1:]
        
        # Get viewBox
        viewBox = root.get('viewBox', '0 0 100 100').split()
        min_x, min_y, width, height = map(float, viewBox)
        
        # Draw all paths
        for path_elem in root.findall('.//svg:path', ns) or root.findall('.//path'):
            try:
                d = path_elem.get('d', '')
                if d:
                    path = parse_path(d)
                    # Convert path to line segments for simple visualization
                    points = []
                    for segment in path:
                        # Sample points along the segment
                        for t in np.linspace(0, 1, 10):
                            point = segment.point(t)
                            points.append([point.real, point.imag])
                    
                    if points:
                        points = np.array(points)
                        ax0.plot(points[:, 0], points[:, 1], 'k-', linewidth=0.5)
            except:
                pass
        
        # Draw rectangles
        for rect in root.findall('.//svg:rect', ns) or root.findall('.//rect'):
            x = float(rect.get('x', 0))
            y = float(rect.get('y', 0))
            w = float(rect.get('width', 0))
            h = float(rect.get('height', 0))
            if w > 0 and h > 0:
                rect_patch = patches.Rectangle((x, y), w, h, 
                                             linewidth=0.5, edgecolor='black', facecolor='none')
                ax0.add_patch(rect_patch)
        
        # Draw circles
        for circle in root.findall('.//svg:circle', ns) or root.findall('.//circle'):
            cx = float(circle.get('cx', 0))
            cy = float(circle.get('cy', 0))
            r = float(circle.get('r', 0))
            if r > 0:
                circle_patch = patches.Circle((cx, cy), r,
                                            linewidth=0.5, edgecolor='black', facecolor='none')
                ax0.add_patch(circle_patch)
        
        # Draw lines
        for line in root.findall('.//svg:line', ns) or root.findall('.//line'):
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
            ax0.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
        
        # Draw polylines and polygons
        for poly in root.findall('.//svg:polyline', ns) or root.findall('.//polyline'):
            points_str = poly.get('points', '')
            if points_str:
                points = []
                for point in points_str.strip().split():
                    x, y = map(float, point.split(','))
                    points.append([x, y])
                if points:
                    points = np.array(points)
                    ax0.plot(points[:, 0], points[:, 1], 'k-', linewidth=0.5)
        
        for poly in root.findall('.//svg:polygon', ns) or root.findall('.//polygon'):
            points_str = poly.get('points', '')
            if points_str:
                points = []
                for point in points_str.strip().split():
                    x, y = map(float, point.split(','))
                    points.append([x, y])
                if points:
                    points.append(points[0])  # Close polygon
                    points = np.array(points)
                    ax0.plot(points[:, 0], points[:, 1], 'k-', linewidth=0.5)
        
        ax0.set_xlim(min_x, min_x + width)
        ax0.set_ylim(min_y + height, min_y)  # Invert Y axis for SVG
        ax0.set_aspect('equal')
        
    except Exception as e:
        print(f"Warning: Could not fully render SVG: {e}")
        # Fallback to point cloud
        ax0.scatter(coords[:, 0], coords[:, 1], c='black', s=1, alpha=0.5)
        ax0.axis('equal')
    
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    
    # Plot 1: Semantic segmentation
    ax1.set_title('Semantic Segmentation', fontsize=14)
    for label_id in np.unique(labels):
        mask = labels == label_id
        if np.sum(mask) > 0:
            color = all_colors[label_id % len(all_colors)]
            ax1.scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=10, alpha=0.6,
                       label=f'{class_names.get(label_id, f"Class {label_id}")} ({np.sum(mask)})')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Instance segmentation
    ax2.set_title(f'Instance Segmentation ({len(results["instances"])} instances)', fontsize=14)
    
    # Plot background points first
    ax2.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=10, alpha=0.3)
    
    # Plot each instance with a different color
    instance_colors = plt.cm.rainbow(np.linspace(0, 1, len(results['instances'])))
    
    for idx, inst in enumerate(results['instances']):
        # Handle both 'mask' and 'masks' keys
        mask_data = inst.get('mask') or inst.get('masks')
        if mask_data is not None:
            if isinstance(mask_data, np.ndarray):
                mask = mask_data > 0.5
            elif isinstance(mask_data, list) and len(mask_data) > 0:
                # Convert boolean list to numpy array
                mask = np.array(mask_data, dtype=bool)
            else:
                mask = np.array(mask_data) > 0.5
            
            if np.sum(mask) > 0:
                label = inst.get('labels', inst.get('label', 'Unknown'))
                if isinstance(label, list):
                    label = label[0] if label else 'Unknown'
                    
                score = inst.get('scores', inst.get('score', 0.0))
                if isinstance(score, (list, np.ndarray)):
                    score = score[0] if len(score) > 0 else 0.0
                
                label_name = class_names.get(label, f"Class {label}")
                ax2.scatter(coords[mask, 0], coords[mask, 1], 
                           c=[instance_colors[idx]], s=20, alpha=0.8,
                           label=f'{label_name} (score: {score:.2f})')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Run inference on a single SVG file')
    parser.add_argument('svg_file', type=str, help='Path to SVG file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/svg/svg_pointT.yaml', 
                        help='Path to config file')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--norm_type', type=str, default='mean', choices=['mean', 'min'],
                        help='Normalization type')
    parser.add_argument('--min_points', type=int, default=2048, help='Minimum number of points')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with reduced points')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    parser.add_argument('--vis_output', type=str, help='Path to save visualization image')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.svg_file):
        print(f"Error: SVG file not found: {args.svg_file}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    try:
        # Step 1: Parse SVG
        svg_data = parse_svg_file(args.svg_file)
        print(f"Parsed {len(svg_data['commands'])} elements from SVG")
        
        # Step 2: Preprocess data
        min_points = 64 if args.debug else args.min_points  # Use much fewer points in debug mode
        processed_data = preprocess_svg_data(svg_data, args.norm_type, min_points)
        print(f"Preprocessed data: {processed_data['points'].shape}")
        
        if args.debug:
            print("Debug mode: Using reduced point count (64) for testing")
        
        # Step 3: Load model
        model = load_model(args.checkpoint, config, args.device)
        
        # Step 4: Run inference
        results = perform_inference(model, processed_data, args.device)
        
        # Step 5: Map instances to SVG paths
        instance_to_elements = map_instances_to_svg_paths(results, svg_data)
        
        # Step 6: Group SVG paths by instance
        grouped_svg_path = None
        if args.output:
            # Create grouped SVG output path
            base_name = os.path.splitext(args.output)[0]
            grouped_svg_path = f"{base_name}_grouped.svg"
        
        instance_groups = group_svg_paths_by_instance(
            args.svg_file, 
            instance_to_elements,
            grouped_svg_path
        )
        
        # Step 7: Format and save results
        # Create a clean predictions dict without numpy arrays
        clean_predictions = {
            'semantic_predictions': results['semantic_predictions'],
            'instances': results['instances'],
            'num_instances': results['num_instances']
        }
        
        output_data = {
            'svg_file': args.svg_file,
            'num_elements': len(svg_data['commands']),
            'num_points': processed_data['num_points'],
            'predictions': clean_predictions,
            'instance_to_elements': instance_to_elements,
            'instance_groups': {
                str(k): {
                    'element_indices': v['element_indices'],
                    'num_elements': len(v['element_indices']),
                    'paths': [{'type': p['type'], **{k: v for k, v in p.items() if k not in ['type', 'element']}} 
                             for p in v['paths']]
                } for k, v in instance_groups.items()
            },
            'svg_dimensions': {
                'width': svg_data.get('width', 0),
                'height': svg_data.get('height', 0)
            }
        }
        
        # Print summary
        print("\n=== Inference Results ===")
        print(f"Number of instances detected: {results['num_instances']}")
        
        # Print detailed instance info
        if results['instances']:
            print("\nTop 5 instances:")
            for idx, inst in enumerate(results['instances'][:5]):
                # Get label - handle different formats
                label = inst.get('label', inst.get('labels', 'Unknown'))
                if isinstance(label, list):
                    label = label[0] if label else 'Unknown'
                
                # Get score - handle different formats
                score = inst.get('score', inst.get('scores', 0.0))
                if isinstance(score, list):
                    score = score[0] if score else 0.0
                
                # Get number of points
                num_points = inst.get('num_points', 0)
                if num_points == 0 and 'mask' in inst:
                    # Calculate from mask if not provided
                    mask = inst['mask']
                    if isinstance(mask, list):
                        num_points = sum(mask)
                    else:
                        num_points = np.sum(mask)
                
                print(f"  {idx+1}. Label: {label}, Score: {score:.3f}, Points: {num_points}")
        
        # Print instance grouping summary
        if instance_to_elements:
            print(f"\n=== Instance to SVG Path Mapping ===")
            print(f"Found {len(instance_to_elements)} instances mapped to SVG elements")
            
            for instance_id, element_indices in sorted(instance_to_elements.items())[:5]:
                if instance_id < len(results['instances']):
                    inst = results['instances'][instance_id]
                    label = inst.get('label', inst.get('labels', 'Unknown'))
                    if isinstance(label, list):
                        label = label[0] if label else 'Unknown'
                    print(f"\nInstance {instance_id} (Class: {label}):")
                    print(f"  - Contains {len(element_indices)} SVG elements")
                    print(f"  - Element indices: {element_indices[:10]}{'...' if len(element_indices) > 10 else ''}")
                    
                    # Show element types if available
                    if instance_id in instance_groups:
                        types = [p['type'] for p in instance_groups[instance_id]['paths']]
                        type_counts = {}
                        for t in types:
                            type_counts[t] = type_counts.get(t, 0) + 1
                        print(f"  - Element types: {dict(type_counts)}")
        
        if grouped_svg_path and os.path.exists(grouped_svg_path):
            print(f"\n✓ Grouped SVG saved to: {grouped_svg_path}")
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Visualize results if requested
        if args.visualize or args.vis_output:
            print("\nGenerating visualization...")
            visualize_results(results, args.svg_file, args.vis_output)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if args.device == 'cuda':
            print("\nTip: Try running with --device cpu to avoid CUDA errors")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
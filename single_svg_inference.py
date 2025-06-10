#!/usr/bin/env python3
"""
Single SVG file inference script for SymPoint model.
Reads an SVG file, preprocesses it, and performs predictions.
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
            # If it's already a list of instances, use it directly
            instances = instances_data
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
        'coords': coords.cpu().numpy(),  # Keep original coords for visualization
        'semantic_labels': semantic_labels.cpu().numpy()
    }


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
        if 'masks' in inst and inst['masks'] is not None:
            masks = inst['masks']
            if isinstance(masks, np.ndarray):
                mask = masks > 0.5
            else:
                mask = np.array(masks) > 0.5
            
            if np.sum(mask) > 0:
                label = inst.get('labels', inst.get('label', 'Unknown'))
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
        
        # Step 5: Format and save results
        output_data = {
            'svg_file': args.svg_file,
            'num_elements': len(svg_data['commands']),
            'num_points': processed_data['num_points'],
            'predictions': results,
            'svg_dimensions': {
                'width': svg_data.get('width', 0),
                'height': svg_data.get('height', 0)
            }
        }
        
        # Print summary
        print("\n=== Inference Results ===")
        print(f"Number of instances detected: {results['num_instances']}")
        if results['instances']:
            print("\nTop 5 instances:")
            for i, inst in enumerate(results['instances'][:5]):
                # Debug: print instance keys for first instance
                if i == 0:
                    print(f"Instance keys: {list(inst.keys())}")
                    if 'masks' in inst:
                        mask_val = inst['masks']
                        print(f"Mask type: {type(mask_val)}, shape/len: {len(mask_val) if hasattr(mask_val, '__len__') else 'N/A'}")
                
                # Handle different instance formats - check types
                labels = inst.get('labels', None)
                scores = inst.get('scores', None)
                masks = inst.get('masks', None)
                
                # Get label - could be int or list
                if isinstance(labels, int):
                    label = labels
                elif isinstance(labels, list) and labels:
                    label = labels[0]
                else:
                    label = 'Unknown'
                
                # Get score - could be float or list
                if isinstance(scores, (int, float)):
                    score = float(scores)
                elif isinstance(scores, list) and scores:
                    score = float(scores[0])
                else:
                    score = 0.0
                
                # Count points from mask
                if masks is not None:
                    if isinstance(masks, np.ndarray):
                        # Binary mask - count True/1 values
                        num_points = int(np.sum(masks > 0.5))
                    elif isinstance(masks, list) and len(masks) > 0:
                        if isinstance(masks[0], list):
                            num_points = sum(masks[0]) if masks[0] else 0
                        else:
                            num_points = sum(1 for m in masks if m > 0.5)
                    else:
                        num_points = 0
                else:
                    num_points = inst.get('num_points', 0)
                
                print(f"  {i+1}. Label: {label}, Score: {score:.3f}, Points: {num_points}")
        
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
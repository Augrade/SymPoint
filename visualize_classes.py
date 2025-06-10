#!/usr/bin/env python3
"""
Script to visualize samples from all original semantic classes in the SVG dataset.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from collections import defaultdict
import random

# Semantic categories mapping (from svg.py)
CATEGORIES = {
    1: {"name": "single door", "color": [224, 62, 155]},
    2: {"name": "double door", "color": [157, 34, 101]},
    3: {"name": "sliding door", "color": [232, 116, 91]},
    4: {"name": "folding door", "color": [101, 54, 72]},
    5: {"name": "revolving door", "color": [172, 107, 133]},
    6: {"name": "rolling door", "color": [142, 76, 101]},
    7: {"name": "window", "color": [96, 78, 245]},
    8: {"name": "bay window", "color": [26, 2, 219]},
    9: {"name": "blind window", "color": [63, 140, 221]},
    10: {"name": "opening symbol", "color": [233, 59, 217]},
    11: {"name": "sofa", "color": [122, 181, 145]},
    12: {"name": "bed", "color": [94, 150, 113]},
    13: {"name": "chair", "color": [66, 107, 81]},
    14: {"name": "table", "color": [123, 181, 114]},
    15: {"name": "TV cabinet", "color": [94, 150, 83]},
    16: {"name": "Wardrobe", "color": [66, 107, 59]},
    17: {"name": "cabinet", "color": [145, 182, 112]},
    18: {"name": "shelf", "color": [152, 147, 200]},
    19: {"name": "showering room", "color": [113, 151, 82]},
    20: {"name": "bathtub", "color": [112, 103, 178]},
    21: {"name": "toilet", "color": [81, 107, 58]},
    22: {"name": "washing basin", "color": [172, 183, 113]},
    23: {"name": "kitchen", "color": [141, 152, 83]},
    24: {"name": "refrigerator", "color": [80, 72, 147]},
    25: {"name": "airconditioner", "color": [100, 108, 59]},
    26: {"name": "washing machine", "color": [182, 170, 112]},
    27: {"name": "other furniture", "color": [238, 124, 162]},
    28: {"name": "stairs", "color": [247, 206, 75]},
    29: {"name": "elevator", "color": [237, 112, 45]},
    30: {"name": "escalator", "color": [233, 59, 46]},
    31: {"name": "row chairs", "color": [172, 107, 151]},
    32: {"name": "parking spot", "color": [102, 67, 62]},
    33: {"name": "wall", "color": [167, 92, 32]},
    34: {"name": "curtain wall", "color": [121, 104, 178]},
    35: {"name": "railing", "color": [64, 52, 105]},
}

def collect_class_samples(data_root, max_files=100, samples_per_class=3):
    """Collect samples of each class from the dataset."""
    
    if not os.path.exists(data_root):
        print(f"Error: Directory {data_root} does not exist!")
        return {}
    
    json_files = glob(os.path.join(data_root, "*.json"))
    if len(json_files) == 0:
        print(f"No JSON files found in {data_root}")
        return {}
    
    # Limit files for faster processing
    if len(json_files) > max_files:
        json_files = random.sample(json_files, max_files)
    
    print(f"Scanning {len(json_files)} files for class samples...")
    
    # Collect samples for each class
    class_samples = defaultdict(list)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        args = np.array(data["args"]).reshape(-1, 8)
        semantic_ids = data['semanticIds']
        instance_ids = data['instanceIds']
        
        # Group paths by semantic class
        class_paths = defaultdict(list)
        for i, (sem_id, inst_id) in enumerate(zip(semantic_ids, instance_ids)):
            if sem_id in CATEGORIES:
                class_paths[sem_id].append({
                    'coords': args[i],
                    'instance_id': inst_id,
                    'file': os.path.basename(json_file)
                })
        
        # Add samples if we don't have enough yet
        for sem_id, paths in class_paths.items():
            if len(class_samples[sem_id]) < samples_per_class:
                needed = samples_per_class - len(class_samples[sem_id])
                class_samples[sem_id].extend(paths[:needed])
    
    return dict(class_samples)

def plot_path_sample(ax, coords, color, alpha=0.7):
    """Plot a single path sample."""
    # coords is (8,) array: [x1,y1,x2,y2,x3,y3,x4,y4]
    coords = coords.reshape(4, 2) * 140  # Denormalize
    
    # Plot as connected line
    ax.plot(coords[:, 0], coords[:, 1], 
           color=np.array(color)/255.0, linewidth=2, alpha=alpha, marker='o', markersize=3)
    
    # Also plot as filled polygon for better visibility
    polygon = patches.Polygon(coords, closed=False, 
                            facecolor=np.array(color)/255.0, alpha=0.3, edgecolor='none')
    ax.add_patch(polygon)

def visualize_all_classes(class_samples, output_file="class_visualization.png"):
    """Create a comprehensive visualization of all classes."""
    
    # Calculate grid size
    n_classes = len(CATEGORIES)
    cols = 6  # 6 columns
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3))
    axes = axes.flatten() if rows > 1 else [axes]
    
    plot_idx = 0
    
    for sem_id in sorted(CATEGORIES.keys()):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        cat_info = CATEGORIES[sem_id]
        name = cat_info['name']
        color = cat_info['color']
        
        ax.set_title(f"{sem_id}: {name}", fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        
        if sem_id in class_samples and len(class_samples[sem_id]) > 0:
            # Plot samples
            samples = class_samples[sem_id][:3]  # Max 3 samples
            
            for i, sample in enumerate(samples):
                coords = sample['coords']
                plot_path_sample(ax, coords, color, alpha=0.8-i*0.2)
            
            # Set axis limits based on data
            all_coords = np.vstack([s['coords'].reshape(4, 2) * 140 for s in samples])
            margin = 10
            ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
            ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
            
            # Add sample info
            ax.text(0.02, 0.98, f"{len(samples)} samples", 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            # No samples found
            ax.text(0.5, 0.5, 'No samples\nfound', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_file}")

def create_class_summary(class_samples):
    """Create a summary of found samples."""
    print(f"\n{'='*80}")
    print("CLASS SAMPLE SUMMARY")
    print(f"{'='*80}")
    print(f"{'ID':<4} {'Class Name':<20} {'Samples Found':<15} {'Sample Files':<30}")
    print(f"{'-'*4} {'-'*20} {'-'*15} {'-'*30}")
    
    for sem_id in sorted(CATEGORIES.keys()):
        name = CATEGORIES[sem_id]['name']
        samples = class_samples.get(sem_id, [])
        count = len(samples)
        
        # Get unique files
        files = list(set([s['file'] for s in samples[:3]]))
        files_str = ', '.join(files) if files else 'None'
        if len(files_str) > 30:
            files_str = files_str[:27] + '...'
        
        print(f"{sem_id:<4} {name:<20} {count:<15} {files_str:<30}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize samples from all semantic classes")
    parser.add_argument("--data_root", type=str, default="dataset/train/jsons",
                        help="Path to dataset directory")
    parser.add_argument("--max_files", type=int, default=100,
                        help="Maximum number of files to scan (default: 100)")
    parser.add_argument("--samples_per_class", type=int, default=3,
                        help="Number of samples per class to collect (default: 3)")
    parser.add_argument("--output", type=str, default="class_visualization.png",
                        help="Output file name (default: class_visualization.png)")
    
    args = parser.parse_args()
    
    # Check alternate paths
    data_root = args.data_root
    if not os.path.exists(data_root):
        alt_paths = [
            "dataset/train/jsons",
            "dataset/train",
            os.path.join(args.data_root, "jsons")
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                data_root = alt_path
                print(f"Using alternate path: {data_root}")
                break
    
    # Collect samples
    class_samples = collect_class_samples(data_root, args.max_files, args.samples_per_class)
    
    if not class_samples:
        print("No samples found!")
        return
    
    # Create summary
    create_class_summary(class_samples)
    
    # Create visualization
    print(f"\nCreating visualization with {len(class_samples)} classes...")
    visualize_all_classes(class_samples, args.output)

if __name__ == "__main__":
    main()
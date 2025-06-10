#!/usr/bin/env python3
"""
Script to analyze class distribution for semantic and instance IDs in the SVG dataset.
"""

import json
import os
from glob import glob
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Semantic categories mapping (from svg.py)
CATEGORIES = {
    1: {"name": "single door", "isthing": 1},
    2: {"name": "double door", "isthing": 1},
    3: {"name": "sliding door", "isthing": 1},
    4: {"name": "folding door", "isthing": 1},
    5: {"name": "revolving door", "isthing": 1},
    6: {"name": "rolling door", "isthing": 1},
    7: {"name": "window", "isthing": 1},
    8: {"name": "bay window", "isthing": 1},
    9: {"name": "blind window", "isthing": 1},
    10: {"name": "opening symbol", "isthing": 1},
    11: {"name": "sofa", "isthing": 1},
    12: {"name": "bed", "isthing": 1},
    13: {"name": "chair", "isthing": 1},
    14: {"name": "table", "isthing": 1},
    15: {"name": "TV cabinet", "isthing": 1},
    16: {"name": "Wardrobe", "isthing": 1},
    17: {"name": "cabinet", "isthing": 1},
    18: {"name": "shelf", "isthing": 1},
    19: {"name": "showering room", "isthing": 1},
    20: {"name": "bathtub", "isthing": 1},
    21: {"name": "toilet", "isthing": 1},
    22: {"name": "washing basin", "isthing": 1},
    23: {"name": "kitchen", "isthing": 1},
    24: {"name": "refrigerator", "isthing": 1},
    25: {"name": "airconditioner", "isthing": 1},
    26: {"name": "washing machine", "isthing": 1},
    27: {"name": "other furniture", "isthing": 1},
    28: {"name": "stairs", "isthing": 1},
    29: {"name": "elevator", "isthing": 1},
    30: {"name": "escalator", "isthing": 1},
    31: {"name": "row chairs", "isthing": 0},
    32: {"name": "parking spot", "isthing": 0},
    33: {"name": "wall", "isthing": 0},
    34: {"name": "curtain wall", "isthing": 0},
    35: {"name": "railing", "isthing": 0},
    36: {"name": "background", "isthing": 0},
}

def analyze_dataset(data_root):
    """Analyze the class distribution in the dataset."""
    
    # Check if directory exists
    if not os.path.exists(data_root):
        print(f"Error: Directory {data_root} does not exist!")
        print("Please run 'python download_data.py' first to download the dataset.")
        return
    
    # Find all JSON files
    json_files = glob(os.path.join(data_root, "*.json"))
    
    if len(json_files) == 0:
        print(f"No JSON files found in {data_root}")
        return
    
    print(f"Found {len(json_files)} JSON files in {data_root}")
    
    # Initialize counters
    semantic_counts = defaultdict(int)
    instance_counts = defaultdict(lambda: defaultdict(int))  # semantic_id -> instance_id -> count
    total_paths = 0
    thing_instances = defaultdict(set)  # semantic_id -> set of unique instance IDs
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        semantic_ids = data['semanticIds']
        instance_ids = data['instanceIds']
        
        # Count occurrences
        for sem_id, inst_id in zip(semantic_ids, instance_ids):
            semantic_counts[sem_id] += 1
            instance_counts[sem_id][inst_id] += 1
            
            # Track unique instances for thing classes
            if inst_id >= 0:  # Thing class
                thing_instances[sem_id].add(inst_id)
        
        total_paths += len(semantic_ids)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"DATASET ANALYSIS: {data_root}")
    print(f"{'='*80}")
    print(f"Total files: {len(json_files)}")
    print(f"Total paths: {total_paths:,}")
    
    print(f"\n{'='*80}")
    print("SEMANTIC CLASS DISTRIBUTION")
    print(f"{'='*80}")
    print(f"{'ID':>4} {'Class Name':<20} {'Type':<6} {'Count':>10} {'Percentage':>10} {'Unique Instances':>17}")
    print(f"{'-'*4} {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*17}")
    
    # Sort by semantic ID
    for sem_id in sorted(semantic_counts.keys()):
        count = semantic_counts[sem_id]
        percentage = (count / total_paths) * 100
        
        # Get category info
        if sem_id in CATEGORIES:
            cat_info = CATEGORIES[sem_id]
            name = cat_info['name']
            is_thing = 'thing' if cat_info['isthing'] else 'stuff'
        elif sem_id == 35:  # Background (mapped from 36)
            name = 'background'
            is_thing = 'stuff'
        else:
            name = f'unknown_{sem_id}'
            is_thing = '?'
        
        # Count unique instances
        if is_thing == 'thing':
            unique_instances = len(thing_instances.get(sem_id, set()))
            instance_str = str(unique_instances)
        else:
            instance_str = 'N/A (stuff)'
        
        print(f"{sem_id:>4} {name:<20} {is_thing:<6} {count:>10,} {percentage:>9.2f}% {instance_str:>17}")
    
    print(f"\n{'='*80}")
    print("INSTANCE ID DISTRIBUTION PER SEMANTIC CLASS")
    print(f"{'='*80}")
    
    for sem_id in sorted(semantic_counts.keys()):
        if sem_id in CATEGORIES:
            cat_info = CATEGORIES[sem_id]
            name = cat_info['name']
            is_thing = cat_info['isthing']
        elif sem_id == 35:
            name = 'background'
            is_thing = 0
        else:
            name = f'unknown_{sem_id}'
            is_thing = None
        
        print(f"\n{sem_id}: {name}")
        
        # Sort instance IDs
        inst_dist = instance_counts[sem_id]
        if is_thing:
            # For thing classes, show distribution of instance counts
            inst_counts = [count for inst_id, count in inst_dist.items() if inst_id >= 0]
            if inst_counts:
                print(f"  Unique instances: {len(inst_counts)}")
                print(f"  Paths per instance: min={min(inst_counts)}, max={max(inst_counts)}, "
                      f"avg={np.mean(inst_counts):.1f}, median={np.median(inst_counts):.1f}")
        else:
            # For stuff classes, should all be -1
            for inst_id, count in sorted(inst_dist.items()):
                print(f"  Instance {inst_id}: {count:,} paths")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze class distribution in SVG dataset")
    parser.add_argument("--data_root", type=str, default="dataset/train",
                        help="Path to dataset directory (default: dataset/train)")
    
    args = parser.parse_args()
    
    # Also check the alternate path mentioned by user
    if not os.path.exists(args.data_root):
        alt_path = os.path.join(args.data_root, "jsons")
        if os.path.exists(alt_path):
            print(f"Using alternate path: {alt_path}")
            analyze_dataset(alt_path)
        else:
            analyze_dataset(args.data_root)
    else:
        analyze_dataset(args.data_root)
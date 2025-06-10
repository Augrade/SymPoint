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
    
    # Call remapped analysis
    analyze_remapped_distribution(semantic_counts, instance_counts, thing_instances, total_paths)

def get_remapped_category(sem_id):
    """Get remapped category for a semantic ID."""
    if 1 <= sem_id <= 6:
        return "doors"
    elif 7 <= sem_id <= 10:
        return "windows"
    elif 11 <= sem_id <= 27:
        return "furniture"
    elif sem_id == 28:
        return "stairs"
    elif 29 <= sem_id <= 30:
        return "equipment"
    elif sem_id in [33, 34]:  # wall, curtain wall
        return "walls"
    else:
        return "other/background"

def analyze_remapped_distribution(semantic_counts, instance_counts, thing_instances, total_paths):
    """Analyze distribution with remapped categories."""
    print(f"\n{'='*80}")
    print("REMAPPED CLASS DISTRIBUTION")
    print(f"{'='*80}")
    
    # Aggregate counts by remapped category
    remapped_counts = defaultdict(int)
    remapped_instances = defaultdict(set)
    remapped_sem_ids = defaultdict(list)
    
    for sem_id, count in semantic_counts.items():
        remapped_cat = get_remapped_category(sem_id)
        remapped_counts[remapped_cat] += count
        remapped_sem_ids[remapped_cat].append(sem_id)
        
        # Aggregate unique instances for thing classes
        if sem_id in thing_instances:
            remapped_instances[remapped_cat].update(thing_instances[sem_id])
    
    # Define order for display
    category_order = ["doors", "windows", "furniture", "stairs", "equipment", "walls", "other/background"]
    
    print(f"{'Category':<20} {'Count':>12} {'Percentage':>12} {'Semantic IDs':<30} {'Unique Instances':>17}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*30} {'-'*17}")
    
    for category in category_order:
        if category in remapped_counts:
            count = remapped_counts[category]
            percentage = (count / total_paths) * 100
            sem_ids = sorted(remapped_sem_ids[category])
            sem_ids_str = str(sem_ids)
            
            # Count unique instances
            if category in ["walls", "other/background"]:
                instance_str = "N/A (stuff)"
            else:
                unique_count = len(remapped_instances.get(category, set()))
                instance_str = str(unique_count) if unique_count > 0 else "0"
            
            print(f"{category:<20} {count:>12,} {percentage:>11.2f}% {sem_ids_str:<30} {instance_str:>17}")
    
    # Print detailed breakdown
    print(f"\n{'='*80}")
    print("DETAILED REMAPPED BREAKDOWN")
    print(f"{'='*80}")
    
    for category in category_order:
        if category in remapped_sem_ids:
            print(f"\n{category.upper()}:")
            sem_ids = sorted(remapped_sem_ids[category])
            
            for sem_id in sem_ids:
                count = semantic_counts.get(sem_id, 0)
                percentage = (count / total_paths) * 100 if total_paths > 0 else 0
                
                # Get original name
                if sem_id in CATEGORIES:
                    name = CATEGORIES[sem_id]['name']
                elif sem_id == 35:
                    name = 'background'
                else:
                    name = f'unknown_{sem_id}'
                
                # Get unique instances
                if sem_id in thing_instances:
                    unique_instances = len(thing_instances[sem_id])
                    instance_str = f"{unique_instances} instances"
                else:
                    instance_str = "stuff class"
                
                print(f"  {sem_id:>3}: {name:<20} {count:>10,} ({percentage:>5.2f}%) - {instance_str}")

def test_remapping():
    """Test the remapping functionality from the SVG dataset."""
    print(f"\n{'='*80}")
    print("TESTING REMAPPED CLASS FUNCTIONALITY")
    print(f"{'='*80}")
    
    # Import and test the remapping
    import sys
    sys.path.append('/Users/kai/Documents/SymPoint')
    from svgnet.data.svg import SVGDataset
    import logging
    
    # Create a mock logger
    logger = logging.getLogger()
    
    try:
        # Test with remapped classes
        dataset = SVGDataset(
            data_root="dataset/train/jsons",
            split="train",
            data_norm="mean",
            aug=None,
            repeat=1,
            logger=logger,
            exclude_railing=False,
            use_remapped_classes=True
        )
        
        print("✓ Successfully created SVGDataset with remapped classes")
        print(f"  - Semantic remapping: {dataset.semantic_remapping}")
        
        # Load one sample to test
        if len(dataset.data_list) > 0:
            _, _, label, _ = dataset.load(dataset.data_list[0], 0)
            unique_semantic_ids = np.unique(label[:, 0])
            print(f"  - Sample file semantic IDs after remapping: {sorted(unique_semantic_ids)}")
            print(f"  - Expected range: 0-6 (7 remapped classes)")
            
            if max(unique_semantic_ids) <= 6:
                print("✓ Remapping working correctly!")
            else:
                print("✗ Remapping may have issues - found IDs > 6")
        
    except Exception as e:
        print(f"✗ Error testing remapped classes: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze class distribution in SVG dataset")
    parser.add_argument("--data_root", type=str, default="dataset/train",
                        help="Path to dataset directory (default: dataset/train)")
    parser.add_argument("--test_remapping", action="store_true",
                        help="Test the remapping functionality")
    
    args = parser.parse_args()
    
    # Test remapping if requested
    if args.test_remapping:
        test_remapping()
        exit()
    
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
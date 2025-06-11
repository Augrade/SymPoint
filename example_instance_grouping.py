#!/usr/bin/env python3
"""
Example script demonstrating how to use instance segmentation to group SVG paths.
This script shows how to:
1. Run inference on an SVG file
2. Map instance masks back to original SVG elements
3. Group SVG paths by instance
4. Create a new SVG with grouped elements
"""

import json
import argparse
from pathlib import Path


def analyze_instance_groups(json_file):
    """Analyze and display instance grouping results from inference output."""
    
    # Load the inference results
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"SVG File: {data['svg_file']}")
    print(f"Total SVG elements: {data['num_elements']}")
    print(f"Total instances detected: {data['predictions']['num_instances']}")
    
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
    
    # Analyze instance groups
    if 'instance_groups' in data:
        print("\n=== Instance Groups ===")
        for instance_id, group_info in data['instance_groups'].items():
            # Get instance label
            inst_idx = int(instance_id)
            if inst_idx < len(data['predictions']['instances']):
                inst = data['predictions']['instances'][inst_idx]
                label = inst.get('label', 'Unknown')
                label_name = class_names.get(label, f'Class {label}')
            else:
                label_name = 'Unknown'
            
            print(f"\nInstance {instance_id} ({label_name}):")
            print(f"  - Number of SVG elements: {group_info['num_elements']}")
            print(f"  - Element indices: {group_info['element_indices']}")
            
            # Count element types
            type_counts = {}
            for path in group_info['paths']:
                path_type = path['type']
                type_counts[path_type] = type_counts.get(path_type, 0) + 1
            
            print(f"  - Element types: {dict(type_counts)}")
            
            # Show first few paths
            print(f"  - Sample paths:")
            for i, path in enumerate(group_info['paths'][:3]):
                if path['type'] == 'path':
                    print(f"    {i+1}. Path: d='{path['d'][:50]}...'")
                elif path['type'] == 'circle':
                    print(f"    {i+1}. Circle: cx={path['cx']}, cy={path['cy']}, r={path['r']}")
                elif path['type'] == 'rect':
                    print(f"    {i+1}. Rectangle: x={path['x']}, y={path['y']}, w={path['width']}, h={path['height']}")
    
    # Show instance-to-element mapping
    if 'instance_to_elements' in data:
        print("\n=== Instance to Element Mapping Summary ===")
        total_mapped = sum(len(elems) for elems in data['instance_to_elements'].values())
        print(f"Total elements mapped to instances: {total_mapped}")
        print(f"Coverage: {total_mapped / data['num_elements'] * 100:.1f}% of all elements")


def main():
    parser = argparse.ArgumentParser(description='Analyze instance grouping results')
    parser.add_argument('json_file', type=str, help='Path to inference output JSON file')
    args = parser.parse_args()
    
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return
    
    analyze_instance_groups(args.json_file)
    
    # Check if grouped SVG exists
    base_name = Path(args.json_file).stem
    grouped_svg = Path(args.json_file).parent / f"{base_name}_grouped.svg"
    if grouped_svg.exists():
        print(f"\n✓ Grouped SVG file available at: {grouped_svg}")
        print("  Open this file in a vector graphics editor to see grouped instances")


if __name__ == '__main__':
    main()
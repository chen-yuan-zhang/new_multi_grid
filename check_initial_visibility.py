"""
Verify Initial Visibility for Distance = 7

This script checks whether initial_distance=7 always results in initial_visibility=False,
given that the maximum FOV range is 5 cells.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import glob

def calculate_manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def check_if_target_in_fov(observer_pos, target_pos, observer_dir, view_size=5):
    """
    Check if target is in observer's FOV based on actual FOV geometry.
    
    FOV is a 5x5 square that extends in the direction the agent is facing.
    Agent is positioned at one edge of the square.
    
    Direction mapping: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
    """
    ox, oy = observer_pos
    tx, ty = target_pos
    
    # Calculate top-left corner of FOV based on direction
    if observer_dir == 0:  # RIGHT
        top_x = ox
        top_y = oy - view_size // 2
    elif observer_dir == 1:  # DOWN
        top_x = ox - view_size // 2
        top_y = oy
    elif observer_dir == 2:  # LEFT
        top_x = ox - view_size + 1
        top_y = oy - view_size // 2
    elif observer_dir == 3:  # UP
        top_x = ox - view_size // 2
        top_y = oy - view_size + 1
    else:
        return False
    
    # Check if target is within the 5x5 FOV square
    in_x_range = top_x <= tx < top_x + view_size
    in_y_range = top_y <= ty < top_y + view_size
    
    return in_x_range and in_y_range

def analyze_initial_visibility(csv_file):
    """Analyze initial visibility by distance from CSV dataset."""
    
    print("="*80)
    print("🔍 INITIAL VISIBILITY ANALYSIS FOR DISTANCE = 7")
    print("="*80)
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return None
    
    print(f"\n📖 Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} scenarios\n")
    
    # Check required columns
    required_cols = ['initial_distance', 'observer_pos', 'target_pos', 'observer_dir']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return None
    
    print("-"*80)
    print("📊 FOV Specifications:")
    print("-"*80)
    print("• FOV Size: 5×5 = 25 cells")
    print("• FOV Shape: Square extending in facing direction")
    print("• Agent Position: At one edge of the FOV square")
    print("• Maximum visibility range:")
    print("  - In facing direction: 4-5 cells ahead")
    print("  - Perpendicular: ±2 cells to each side")
    print("  - Behind: 0 cells (no rear visibility)")
    print("• Maximum possible distance in FOV: ~5 cells (even with perfect alignment)")
    
    print("\n" + "-"*80)
    print("📐 Analysis by Initial Distance:")
    print("-"*80)
    
    distances = sorted(df['initial_distance'].unique())
    results_by_distance = {}
    
    for distance in distances:
        subset = df[df['initial_distance'] == distance]
        
        print(f"\n🎯 Distance = {int(distance)}")
        print(f"   Total scenarios: {len(subset)}")
        
        # Analyze visibility for this distance
        visibility_results = []
        max_actual_distance = 0
        min_actual_distance = float('inf')
        
        for idx, row in subset.iterrows():
            try:
                observer_pos = eval(row['observer_pos']) if isinstance(row['observer_pos'], str) else row['observer_pos']
                target_pos = eval(row['target_pos']) if isinstance(row['target_pos'], str) else row['target_pos']
                observer_dir = int(row['observer_dir'])
                
                # Calculate actual Manhattan distance
                actual_distance = calculate_manhattan_distance(observer_pos, target_pos)
                max_actual_distance = max(max_actual_distance, actual_distance)
                min_actual_distance = min(min_actual_distance, actual_distance)
                
                # Check if target should be in FOV
                in_fov = check_if_target_in_fov(observer_pos, target_pos, observer_dir)
                
                visibility_results.append({
                    'in_fov': in_fov,
                    'actual_distance': actual_distance,
                    'observer_pos': observer_pos,
                    'target_pos': target_pos,
                    'observer_dir': observer_dir
                })
            except Exception as e:
                print(f"   ⚠️ Error processing row {idx}: {e}")
                continue
        
        if visibility_results:
            visible_count = sum(1 for r in visibility_results if r['in_fov'])
            total_count = len(visibility_results)
            visibility_rate = (visible_count / total_count * 100) if total_count > 0 else 0
            
            results_by_distance[distance] = {
                'total': total_count,
                'visible': visible_count,
                'rate': visibility_rate,
                'min_distance': min_actual_distance,
                'max_distance': max_actual_distance
            }
            
            print(f"   Distance range: {min_actual_distance:.1f} - {max_actual_distance:.1f}")
            print(f"   Initially visible: {visible_count}/{total_count} ({visibility_rate:.1f}%)")
            
            if visibility_rate == 0:
                print(f"   ✅ CONFIRMED: 0% initial visibility (as expected for distance {int(distance)})")
            elif visibility_rate < 10:
                print(f"   ⚠️  Low visibility: Only {visibility_rate:.1f}% visible")
            else:
                print(f"   ❌ UNEXPECTED: {visibility_rate:.1f}% visible (should be 0% for distance > 5)")
                # Show some examples
                visible_examples = [r for r in visibility_results if r['in_fov']][:3]
                for i, ex in enumerate(visible_examples):
                    print(f"      Example {i+1}: Obs={ex['observer_pos']} (dir={ex['observer_dir']}), "
                          f"Target={ex['target_pos']}, Distance={ex['actual_distance']}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY:")
    print("="*80)
    
    for distance in sorted(results_by_distance.keys()):
        result = results_by_distance[distance]
        status = "✅" if result['rate'] == 0 or distance <= 5 else "❌"
        print(f"{status} Distance {int(distance)}: {result['visible']}/{result['total']} visible "
              f"({result['rate']:.1f}%)")
    
    print("\n" + "-"*80)
    print("🎯 VERIFICATION FOR DISTANCE = 7:")
    print("-"*80)
    
    if 7 in results_by_distance:
        result = results_by_distance[7]
        print(f"\nTotal scenarios with distance=7: {result['total']}")
        print(f"Initially visible: {result['visible']}")
        print(f"Initial visibility rate: {result['rate']:.1f}%")
        
        if result['rate'] == 0:
            print("\n✅ ✅ ✅ VERIFIED: Initial visibility is 0% for distance = 7")
            print("   This is CORRECT because:")
            print("   • Maximum FOV range is ~5 cells")
            print("   • Distance 7 exceeds maximum FOV range")
            print("   • Target cannot be visible at initial state")
        else:
            print(f"\n❌ ISSUE FOUND: {result['rate']:.1f}% of distance-7 scenarios are initially visible")
            print("   This should NOT happen given FOV constraints.")
            print("   Possible causes:")
            print("   • Distance calculation error")
            print("   • FOV calculation error")
            print("   • Data generation issue")
    else:
        print("\n⚠️ No scenarios found with initial_distance = 7")
        print("   Available distances:", sorted(results_by_distance.keys()))
    
    print("\n" + "="*80)
    
    return results_by_distance

def find_recent_csv():
    """Find the most recent CSV file with results."""
    patterns = ['*results*.csv', 'test_*.csv', '*.csv']
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            # Filter out evaluation results (we want generation results)
            gen_files = [f for f in files if 'evaluation' not in f.lower()]
            if gen_files:
                return max(gen_files, key=lambda x: Path(x).stat().st_mtime)
            return max(files, key=lambda x: Path(x).stat().st_mtime)
    
    return None

def main():
    """Main function."""
    
    # Get CSV file from argument or find recent one
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_recent_csv()
        if csv_file:
            print(f"📄 Using most recent file: {csv_file}\n")
        else:
            print("❌ No CSV files found.")
            print("   Usage: python check_initial_visibility.py <csv_file>")
            print("   Or generate data using generator_test.py first.")
            return
    
    analyze_initial_visibility(csv_file)

if __name__ == "__main__":
    main()

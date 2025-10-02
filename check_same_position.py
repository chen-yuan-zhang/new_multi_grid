"""
Quick check: Are observer and target starting at the same position?

This could explain 100% initial visibility - if they're at the same cell,
they would always see each other regardless of FOV.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import glob

def check_same_position(csv_file):
    """Check if observer and target start at the same position."""
    
    print("="*80)
    print("🔍 CHECKING FOR SAME-POSITION BUG")
    print("="*80)
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    print(f"\n📖 Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} scenarios\n")
    
    # Check a sample of scenarios
    print("-"*80)
    print("🔍 Checking Sample Scenarios:")
    print("-"*80)
    
    same_pos_count = 0
    
    for i, row in df.head(20).iterrows():
        try:
            observer_pos = eval(row['observer_pos']) if isinstance(row['observer_pos'], str) else row['observer_pos']
            target_pos = eval(row['target_pos']) if isinstance(row['target_pos'], str) else row['target_pos']
            initial_distance = row['initial_distance']
            
            # Calculate actual distance
            actual_dist = abs(observer_pos[0] - target_pos[0]) + abs(observer_pos[1] - target_pos[1])
            
            same_pos = (observer_pos == target_pos)
            if same_pos:
                same_pos_count += 1
            
            print(f"\nScenario {i}:")
            print(f"  Initial distance (specified): {initial_distance}")
            print(f"  Observer: {observer_pos}")
            print(f"  Target: {target_pos}")
            print(f"  Actual distance: {actual_dist}")
            
            if same_pos:
                print(f"  ❌ BUG: SAME POSITION! This would cause 100% visibility")
            elif actual_dist != initial_distance:
                print(f"  ⚠️  Distance mismatch: specified={initial_distance}, actual={actual_dist}")
            else:
                print(f"  ✅ Correct")
                
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY:")
    print("="*80)
    
    if same_pos_count > 0:
        print(f"\n❌ CRITICAL BUG FOUND:")
        print(f"   {same_pos_count}/20 scenarios have observer and target at SAME POSITION")
        print(f"   This explains 100% initial visibility!")
        print(f"\n   Root cause:")
        print(f"   • Observer and target are both using the SAME starting position")
        print(f"   • They're at same cell → always visible to each other")
        print(f"   • This overrides FOV distance limitations")
    else:
        print(f"\n✅ No same-position issues found in sample")
        print(f"   All observer/target pairs have different positions")
        print(f"\n   If you're still seeing 100% visibility, the issue might be:")
        print(f"   1. Positions are very close (within FOV range)")
        print(f"   2. FOV checking logic has a bug")
        print(f"   3. Target is always in front of observer")
    
    print("\n" + "="*80)

def main():
    """Main function."""
    
    # Find CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        patterns = ['*results*.csv', 'test_*.csv']
        for pattern in patterns:
            files = glob.glob(pattern)
            gen_files = [f for f in files if 'evaluation' not in f.lower()]
            if gen_files:
                csv_file = max(gen_files, key=lambda x: Path(x).stat().st_mtime)
                print(f"📄 Using most recent file: {csv_file}\n")
                break
        else:
            print("❌ No CSV files found.")
            print("   Usage: python check_same_position.py <csv_file>")
            return
    
    check_same_position(csv_file)

if __name__ == "__main__":
    main()

"""
Quick script to check what's in the generated pickle file.
"""

import gzip
import pickle
import os

# Check if file exists
pickle_file = 'results.pkl.gz'

if not os.path.exists(pickle_file):
    print(f"❌ File '{pickle_file}' does not exist!")
    print(f"\nFiles in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.pkl.gz') or f.endswith('.pkl'):
            print(f"  - {f}")
else:
    print(f"✅ File '{pickle_file}' exists")
    file_size = os.path.getsize(pickle_file)
    print(f"📦 File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    if file_size == 0:
        print("❌ File is empty (0 bytes)!")
    else:
        print(f"\n📖 Reading pickle file...")
        try:
            with gzip.open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Successfully loaded pickle file")
            print(f"\n📊 Data Information:")
            print(f"  Type: {type(data)}")
            print(f"  Length: {len(data)} scenarios")
            
            if len(data) == 0:
                print("\n❌ The pickle file contains an empty list!")
            else:
                print(f"\n✅ File contains {len(data)} scenarios")
                
                # Show first scenario
                first = data[0]
                print(f"\n📋 First Scenario Keys:")
                for key in first.keys():
                    value = first[key]
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                    else:
                        print(f"  - {key}: {type(value).__name__}")
                
                # Show summary
                print(f"\n📊 Dataset Summary:")
                sizes = [r['size'] for r in data]
                print(f"  Grid sizes: {sorted(set(sizes))}")
                
                traj_lengths = [len(r['all_actions']) for r in data]
                print(f"  Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, avg={sum(traj_lengths)/len(traj_lengths):.1f}")
                
                cost_types = [r['hidden_cost_type'] for r in data]
                print(f"  Hidden cost types: {sorted(set(cost_types))}")
                
        except Exception as e:
            print(f"❌ Error reading pickle file: {e}")
            import traceback
            traceback.print_exc()

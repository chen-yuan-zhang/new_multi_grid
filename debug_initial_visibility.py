"""
Debug Initial Visibility Issue

This script investigates why initial visibility is 100% even for distance=7.
We'll inspect the actual observation data to see what's in the image.
"""

import numpy as np
import json
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action

def debug_initial_visibility():
    """
    Create a simple test case with distance=7 and check the initial observation.
    """
    
    print("="*80)
    print("🔍 DEBUGGING INITIAL VISIBILITY ISSUE")
    print("="*80)
    
    # Create a simple 15x15 grid with distance 7
    size = 15
    base_grid = np.zeros((size, size), dtype=np.int32)
    
    # Place some walls
    base_grid[7, :] = 1  # Horizontal wall in middle
    base_grid[7, 7] = 0  # Gap in middle
    
    # Set positions with distance = 7
    observer_pos = (3, 7)  # Left side
    target_pos = (10, 7)   # Right side, distance = 7
    observer_dir = 0  # Facing RIGHT
    target_dir = 2    # Facing LEFT
    
    # Calculate Manhattan distance
    distance = abs(observer_pos[0] - target_pos[0]) + abs(observer_pos[1] - target_pos[1])
    print(f"\n📍 Setup:")
    print(f"   Grid size: {size}×{size}")
    print(f"   Observer: {observer_pos} facing {'RIGHT' if observer_dir == 0 else 'dir=' + str(observer_dir)}")
    print(f"   Target: {target_pos} facing {'LEFT' if target_dir == 2 else 'dir=' + str(target_dir)}")
    print(f"   Manhattan distance: {distance}")
    
    # Create goals
    goals = [(1, 1), (1, 13), (13, 1), (13, 13)]
    goal = goals[0]
    
    # Create environment
    env = AGREnv(
        base_grid=base_grid,
        goals=goals,
        goal=goal,
        hidden_cost=None,
        enable_hidden_cost=False,
        agents_start_pos=[observer_pos, target_pos],
        agents_start_dir=[observer_dir, target_dir],
        render_mode=None
    )
    
    observation, info = env.reset()
    
    print(f"\n🔍 Analyzing Initial Observation:")
    print(f"   Observation type: {type(observation)}")
    print(f"   Observation keys: {observation.keys() if isinstance(observation, dict) else 'N/A'}")
    
    # Get observer's observation
    if isinstance(observation, dict) and 0 in observation:
        obs = observation[0]
    else:
        obs = observation
    
    print(f"\n📊 Observer Observation Contents:")
    print(f"   Keys: {obs.keys()}")
    print(f"   Image shape: {obs['image'].shape}")
    print(f"   Direction: {obs.get('direction', 'N/A')}")
    
    # Check what's in the image
    image = obs['image']
    unique_values = np.unique(image[..., 0])  # First channel contains object types
    print(f"\n🎨 Unique object types in image (first channel):")
    print(f"   Values: {unique_values}")
    
    # Check for agent (type 10)
    has_agent_10 = 10 in image[..., 0]
    has_agent_any = 10 in image
    print(f"\n👤 Agent Detection:")
    print(f"   10 in image[..., 0]: {has_agent_10}")
    print(f"   10 in image (any channel): {has_agent_any}")
    
    # Show where 10 appears
    if has_agent_any:
        locations = np.where(image == 10)
        print(f"   Locations where 10 appears:")
        for i in range(len(locations[0])):
            print(f"      Position ({locations[0][i]}, {locations[1][i]}), Channel {locations[2][i]}")
    
    # Check target_pos in observation
    has_target_pos = "target_pos" in obs
    print(f"\n🎯 Target Information:")
    print(f"   'target_pos' in obs: {has_target_pos}")
    if has_target_pos:
        print(f"   target_pos value: {obs['target_pos']}")
        print(f"   ❌ TARGET IS MARKED AS VISIBLE (SHOULD NOT BE at distance {distance}!)")
    else:
        print(f"   ✅ Target is NOT visible (correct for distance {distance})")
    
    # Visualize the FOV
    print(f"\n📐 Field of View Analysis:")
    print(f"   Observer view_size: {env.observer.view_size}")
    
    # Calculate FOV boundaries based on direction
    view_size = env.observer.view_size
    ox, oy = observer_pos
    
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
    
    print(f"   FOV top-left: ({top_x}, {top_y})")
    print(f"   FOV bottom-right: ({top_x + view_size - 1}, {top_y + view_size - 1})")
    print(f"   FOV range X: [{top_x}, {top_x + view_size - 1}]")
    print(f"   FOV range Y: [{top_y}, {top_y + view_size - 1}]")
    
    # Check if target is in FOV
    tx, ty = target_pos
    in_fov_x = top_x <= tx <= top_x + view_size - 1
    in_fov_y = top_y <= ty <= top_y + view_size - 1
    in_fov = in_fov_x and in_fov_y
    
    print(f"\n   Target at {target_pos}:")
    print(f"      In FOV X range: {in_fov_x}")
    print(f"      In FOV Y range: {in_fov_y}")
    print(f"      In FOV: {in_fov}")
    
    if in_fov and distance == 7:
        print(f"\n   ⚠️  WARNING: Target IS in FOV despite distance = 7!")
        print(f"   This means FOV extends further than expected.")
    elif not in_fov and has_target_pos:
        print(f"\n   ❌ BUG FOUND: Target NOT in FOV but marked as visible!")
    elif not in_fov and not has_target_pos:
        print(f"\n   ✅ CORRECT: Target not in FOV and not marked as visible")
    
    # Print the image grid (just object types)
    print(f"\n🖼️  Image Grid (object types only, 5×5):")
    for i in range(view_size):
        row_str = "   "
        for j in range(view_size):
            val = image[i, j, 0]
            if val == 0:
                row_str += " . "  # unseen
            elif val == 1:
                row_str += "   "  # empty
            elif val == 2:
                row_str += " # "  # wall
            elif val == 10:
                row_str += " T "  # target (agent type 10)
            else:
                row_str += f"{val:2d} "
        print(row_str)
    
    env.close()
    
    print("\n" + "="*80)
    print("🔍 Diagnosis:")
    print("="*80)
    
    if has_target_pos and distance == 7:
        if in_fov:
            print("\n❓ FINDING: Target IS visible at distance 7 because it's in FOV")
            print("   Possible explanations:")
            print("   1. FOV extends 7 cells in the facing direction")
            print("   2. FOV calculation is different than expected")
            print("   3. Observer facing direction places target in FOV")
        else:
            print("\n❌ BUG CONFIRMED: Target marked visible but NOT in FOV!")
            print("   The condition 'if 10 in obs['image']' is incorrect")
            print("   Need to investigate mod_obs() function in goal_prediction.py")
    else:
        print("\n✅ Working as expected for this test case")
    
    print("="*80)

if __name__ == "__main__":
    debug_initial_visibility()

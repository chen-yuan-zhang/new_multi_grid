"""
Test to check if actor movement is blocked by walls
"""
import numpy as np
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import Direction

def test_wall_collision():
    """Test if actor movement fails due to wall collisions"""
    
    # Create environment
    env = AGREnv(size=10)
    obs, info = env.reset()
    
    print("🧱 Wall Collision Test")
    print("=" * 40)
    
    # Get initial positions and directions
    observer_pos = env.observer.pos
    target_pos = env.target.pos
    observer_dir = env.observer.dir
    target_dir = env.target.dir
    
    print(f"Observer: pos={observer_pos}, dir={target_dir} ({Direction(observer_dir).name})")
    print(f"Target: pos={target_pos}, dir={target_dir} ({Direction(target_dir).name})")
    print()
    
    # Print the grid around target
    print("🎯 Grid around Target:")
    grid = env.grid.encode()[:, :, 0]  # Get object types
    target_x, target_y = target_pos
    
    print("Grid layout (0=empty, 1=wall):")
    for y in range(max(0, target_y-2), min(env.height, target_y+3)):
        row_str = ""
        for x in range(max(0, target_x-2), min(env.width, target_x+3)):
            if (x, y) == tuple(target_pos):
                row_str += "T "  # Target position
            elif grid[x, y] == 1:  # Wall
                row_str += "█ "
            elif grid[x, y] == 0:  # Empty
                row_str += ". "
            else:
                row_str += f"{grid[x, y]} "
        print(f"  {row_str}")
    print()
    
    # Check what's in front of target
    target_dir_vec = Direction(target_dir).to_vec()
    next_pos = (target_pos[0] + target_dir_vec[0], target_pos[1] + target_dir_vec[1])
    
    print(f"Target facing: {Direction(target_dir).name}")
    print(f"Direction vector: {target_dir_vec}")
    print(f"Next position if moves forward: {next_pos}")
    
    # Check if next position is valid
    if (0 <= next_pos[0] < env.width and 0 <= next_pos[1] < env.height):
        cell_type = grid[next_pos[0], next_pos[1]]
        print(f"Cell at next position: {cell_type} ({'wall' if cell_type == 1 else 'empty' if cell_type == 0 else 'other'})")
        
        if cell_type == 1:
            print("🚫 WALL DETECTED! Target cannot move forward.")
        else:
            print("✅ Path is clear, target should be able to move.")
    else:
        print("🚫 BOUNDARY! Target would move out of bounds.")
    
    print()
    
    # Test the actual movement
    print("🎬 Testing actual movement:")
    actions = {0: Action.stay, 1: Action.forward}  # Observer stays, target moves forward
    
    obs_new, reward, terminated, truncated, info = env.step(actions)
    
    new_target_pos = env.target.pos
    print(f"Target position after forward action: {new_target_pos}")
    
    if tuple(new_target_pos) == tuple(target_pos):
        print("❌ Target did not move (likely blocked by wall)")
    else:
        print("✅ Target moved successfully")
    
    # Test turning instead
    print()
    print("🔄 Testing turn action:")
    env.reset()  # Reset to initial state
    
    actions = {0: Action.stay, 1: Action.left}  # Observer stays, target turns left
    obs_new, reward, terminated, truncated, info = env.step(actions)
    
    new_target_dir = env.target.dir
    print(f"Target direction after turn left: {Direction(new_target_dir).name} (was {Direction(target_dir).name})")
    
    if new_target_dir != target_dir:
        print("✅ Turn action works correctly")
    else:
        print("❌ Turn action failed")

if __name__ == "__main__":
    test_wall_collision()
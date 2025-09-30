"""
Debug Environment Actions - Simple Test
"""

import numpy as np
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import Direction

def debug_simple_action():
    """Debug a single action to see what's happening."""
    print("🔍 Debug Single Action")
    print("=" * 30)
    
    env = AGREnv(
        size=6,
        agents_start_pos=[(1, 1), (3, 3)],
        agents_start_dir=[Direction.right, Direction.left]
    )
    
    obs, info = env.reset()
    
    print(f"Environment setup complete")
    
    print(f"Observer: {env.observer}")
    print(f"Target: {env.target}")
    
    # Try a simple action
    actions = {
        0: Action.forward,  # observer
        1: Action.forward   # target
    }
    
    print(f"\\nBefore step:")
    print(f"  Observer pos: {env.observer.pos}, dir: {env.observer.dir}")
    print(f"  Target pos: {env.target.pos}, dir: {env.target.dir}")
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    print(f"\\nAfter step:")
    print(f"  Observer pos: {env.observer.pos}, dir: {env.observer.dir}")
    print(f"  Target pos: {env.target.pos}, dir: {env.target.dir}")
    
    print(f"\\nActions executed: {actions}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    
    env.close()

if __name__ == "__main__":
    debug_simple_action()
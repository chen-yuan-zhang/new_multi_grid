"""
Simplified Dataset Generator for Goal Recognition Experiments

This module generates datasets containing actor trajectories with different behavior styles.
"""

import random
import numpy as np
import pandas as pd
import json
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.target import AstarTarget

# Configuration
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZES = [8, 10]  # Smaller sizes for testing
INITIAL_DISTANCES = [3, 5]
NUM_SCENARIOS = 2  # Reduced for testing
BEHAVIOR_STYLES = ["like_wall", "hate_wall", "like_edge", "hate_edge"]

def generate_hidden_cost_matrix(size, base_grid, style_type):
    """Generate a single hidden cost matrix based on behavior style."""
    cost_matrix = 10 * np.ones((size, size))
    
    # Find wall positions
    wall_rows, wall_cols = np.where(base_grid == 1)
    
    for i in range(size):
        for j in range(size):
            if base_grid[i, j] == 0:  # Free cell
                # Distance to closest wall
                if len(wall_rows) > 0:
                    wall_dist = min(abs(wall_rows[k] - i) + abs(wall_cols[k] - j) 
                                  for k in range(len(wall_rows)))
                else:
                    wall_dist = size
                
                # Distance to edge
                edge_dist = min(i, j, size - i - 1, size - j - 1)
                
                # Set cost based on style
                if style_type == "like_wall":
                    cost_matrix[i, j] = wall_dist
                elif style_type == "hate_wall":
                    cost_matrix[i, j] = 1 / (wall_dist + 1)
                elif style_type == "like_edge":
                    cost_matrix[i, j] = edge_dist
                elif style_type == "hate_edge":
                    cost_matrix[i, j] = 1 / (edge_dist + 1)
    
    return cost_matrix

def generate_trajectory(env):
    """Generate actor trajectory using A* with hidden costs."""
    obs, info = env.reset()
    target_agent = AstarTarget(env)
    
    actions = []
    step = 0
    max_steps = 100  # Safety limit
    
    while not env.unwrapped.is_done() and step < max_steps:
        # Get target action
        target_action = target_agent.compute_action(obs)
        actions.append(target_action)
        
        # Step environment
        agent_actions = [0, target_action]  # Observer action is dummy
        obs, reward, terminated, truncated, info = env.step(agent_actions)
        step += 1
    
    return actions

def main():
    """Generate the dataset."""
    print("🎯 Generating simplified dataset...")
    
    results = []
    scenario_count = 0
    
    for size in GRID_SIZES:
        print(f"📐 Grid size: {size}x{size}")
        
        for layout_idx, initial_distance in enumerate(INITIAL_DISTANCES):
            print(f"  🎯 Initial distance: {initial_distance}")
            
            # Generate base environment
            env = AGREnv(size=size)
            obs, info = env.reset()
            base_grid = info['base_grid']
            env.close()
            
            for scenario_id in range(NUM_SCENARIOS):
                print(f"    📋 Scenario {scenario_id}")
                
                # Generate scenario setup
                setup_env = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
                obs, info = setup_env.reset()
                
                goals = info['goals']
                goal = info['goal']
                start_pos = info['agents_start_pos']
                start_dir = info['agents_start_dir']
                setup_env.close()
                
                for style_idx, style_name in enumerate(BEHAVIOR_STYLES):
                    print(f"      🎭 Style: {style_name}")
                    
                    # Generate hidden cost matrix
                    hidden_cost = generate_hidden_cost_matrix(size, base_grid, style_name)
                    
                    # Create environment with hidden costs
                    cost_env = AGREnv(
                        size=size,
                        initial_distance=initial_distance,
                        base_grid=base_grid,
                        goals=goals,
                        goal=goal,
                        enable_hidden_cost=True,
                        hidden_cost=hidden_cost,
                        agents_start_pos=start_pos,
                        agents_start_dir=start_dir
                    )
                    
                    # Generate trajectory
                    try:
                        actions = generate_trajectory(cost_env)
                        cost_env.close()
                        
                        # Store result
                        result = {
                            'scenario_id': scenario_count,
                            'size': size,
                            'layout_id': layout_idx,
                            'initial_distance': initial_distance,
                            'behavior_style': style_name,
                            'base_grid': json.dumps(base_grid.tolist()),
                            'hidden_cost': json.dumps(hidden_cost.tolist()),
                            'goals': str(goals),
                            'goal': str(goal),
                            'observer_pos': str(start_pos[0]),
                            'target_pos': str(start_pos[1]),
                            'observer_dir': start_dir[0],
                            'target_dir': start_dir[1],
                            'all_actions': json.dumps([int(a) for a in actions]),
                            'total_steps': len(actions)
                        }
                        
                        results.append(result)
                        scenario_count += 1
                        print(f"        ✅ Generated {len(actions)} steps")
                        
                    except Exception as e:
                        print(f"        ❌ Error: {e}")
                        if 'cost_env' in locals():
                            cost_env.close()
                        continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        filename = "goal_recognition_dataset_simple.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Saved {len(results)} scenarios to {filename}")
        
        # Print summary
        print(f"\n📊 Dataset Summary:")
        print(f"  Total scenarios: {len(df)}")
        print(f"  Grid sizes: {sorted(df['size'].unique())}")
        print(f"  Behavior styles: {list(df['behavior_style'].unique())}")
        print(f"  Avg trajectory length: {df['total_steps'].mean():.1f}")
    else:
        print("\n❌ No scenarios generated")

if __name__ == "__main__":
    main()
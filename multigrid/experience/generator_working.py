"""
Working Dataset Generator for Goal Recognition Experiments

Based on the original generator structure but with improved organization.
"""

import random
import numpy as np
import pandas as pd
import json

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.target import AstarTarget

def set_all_seed(seed):
    """Set random seeds for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)

# Configuration
SEED = 123
set_all_seed(SEED)

def generate_hidden_cost_matrices(size, base_grid):
    """Generate 4 different hidden cost matrices."""
    # Wall positions
    rows, cols = np.where(base_grid == 1)
    
    cost_matrix_1 = 10 * np.ones((size, size))  # like_wall
    cost_matrix_2 = 10 * np.ones((size, size))  # hate_wall
    cost_matrix_3 = 10 * np.ones((size, size))  # like_edge
    cost_matrix_4 = 10 * np.ones((size, size))  # hate_edge

    for i in range(size):
        for j in range(size):
            if base_grid[i, j] == 0:  # Free cell
                # Distance to closest wall
                if len(rows) > 0:
                    min_dist = min(abs(rows[k] - i) + abs(cols[k] - j) for k in range(len(rows)))
                else:
                    min_dist = size
                
                # Distance to edge
                edge_dist = min(i, j, size - i - 1, size - j - 1)
                
                # Set costs
                cost_matrix_1[i, j] = min_dist                    # like wall
                cost_matrix_2[i, j] = 1 / (min_dist + 1)         # hate wall
                cost_matrix_3[i, j] = edge_dist                  # like edge
                cost_matrix_4[i, j] = 1 / (edge_dist + 1)       # hate edge

    return [cost_matrix_1, cost_matrix_2, cost_matrix_3, cost_matrix_4]

def generate_trajectory(env):
    """Generate actor trajectory with hidden costs."""
    obs, info = env.reset()
    target_agent = AstarTarget(env)
    
    all_actions = []
    all_imgs = []
    all_obs = []
    
    step = 0
    max_steps = 100
    
    while not env.unwrapped.is_done() and step < max_steps:
        # Store observation and image
        all_obs.append(obs[1])  # Target agent observation
        all_imgs.append(env.grid.render(tile_size=32, agents=env.unwrapped.agents[1:], highlight_mask=None))
        
        # Get target action
        target_action = target_agent.compute_action(obs)
        all_actions.append(target_action)
        
        # Create actions for all agents
        # Observer stays still (Action 6 = stay) to not interfere with target trajectory
        actions = {agent.index: 6 for agent in env.unwrapped.agents}  # 6 = Action.stay
        actions[1] = target_action  # Target is agent 1 (moves according to hidden costs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        step += 1
    
    return all_actions, all_imgs, all_obs

def main():
    """Generate the dataset."""
    
    # Parameters - systematic dataset generation
    sizes = [10, 12, 15]  # Different environment sizes
    initial_distances = [3, 5, 7]  # Different starting distances
    num_layouts = 5  # Number of different layouts per configuration
    num_scenarios = 4  # Number of different start position scenarios per layout
    style_names = ["like_wall", "hate_wall", "like_edge", "hate_edge"]
    
    results = []
    
    print("🎯 Generating Goal Recognition Dataset")
    print(f"Grid sizes: {sizes}")
    print(f"Initial distances: {initial_distances}")
    print(f"Layouts per configuration: {num_layouts}")
    print(f"Scenarios per layout: {num_scenarios}")
    print(f"Behavior styles: {len(style_names)}")
    print(f"Total scenarios: {len(sizes) * len(initial_distances) * num_layouts * num_scenarios * len(style_names)}")
    print()

    for size in sizes:
        print(f"📐 Processing grid size: {size}x{size}")
        
        for initial_distance in initial_distances:
            print(f"  🎯 Initial distance: {initial_distance}")
            
            for layout_id in range(num_layouts):
                print(f"    🏗️  Layout {layout_id + 1}/{num_layouts}")
                
                # Generate base grid for this layout
                env = AGREnv(size=size)
                obs, info = env.reset()
                base_grid = info['base_grid']
                env.close()
            
                # Generate hidden cost matrices
                hidden_costs = generate_hidden_cost_matrices(size, base_grid)
                
                for scenario_id in range(num_scenarios):
                    print(f"      📋 Scenario {scenario_id + 1}/{num_scenarios}: generating start positions...")
                    
                    # Generate start positions and goals
                    env_grid = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
                    obs, info = env_grid.reset()
                    
                    goals = info['goals']
                    goal = info['goal']
                    agents_start_pos = info['agents_start_pos']
                    agents_start_dir = info['agents_start_dir']
                    env_grid.close()
                    
                    for style_id, hidden_cost in enumerate(hidden_costs):
                        style_name = style_names[style_id]
                        print(f"        🎭 Style {style_id} ({style_name}): computing trajectory...")
                    
                        try:
                            # Create environment with hidden costs
                            env_agents = AGREnv(
                                size=size, 
                                initial_distance=initial_distance, 
                                base_grid=base_grid, 
                                goals=goals,
                                goal=goal, 
                                enable_hidden_cost=True, 
                                hidden_cost=hidden_cost, 
                                agents_start_dir=agents_start_dir, 
                                agents_start_pos=agents_start_pos
                            )
                            
                            # Generate trajectory
                            all_actions, all_imgs, all_obs = generate_trajectory(env_agents)
                            env_agents.close()
                            
                            # Save result
                            result = {
                                "base_grid": json.dumps(base_grid.tolist()),
                                "hidden_cost": json.dumps(hidden_cost.tolist()),
                                "observer_pos": agents_start_pos[0],
                                "target_pos": agents_start_pos[1],
                                "observer_dir": agents_start_dir[0],
                                "target_dir": agents_start_dir[1],
                                'size': size,
                                'layout_id': layout_id,
                                'initial_distance': initial_distance,
                                'scenario_id': scenario_id,
                                'hidden_cost_type': style_id,
                                'hidden_cost_style': style_name,
                                'goals': goals,
                                'goal': goal,
                                'all_actions': json.dumps([a.value for a in all_actions]),
                                'all_imgs': all_imgs,
                                'all_obs': all_obs,
                                'total_steps': len(all_actions)
                            }
                            
                            results.append(result)
                            print(f"          ✅ Generated {len(all_actions)} steps | Total: {len(results)}")
                            
                        except Exception as e:
                            print(f"          ❌ Error: {e}")
                            continue

    # Save results
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total scenarios: {len(df)}")
        print(f"  Grid sizes: {sorted(df['size'].unique())}")
        print(f"  Behavior styles: {list(df['hidden_cost_style'].unique())}")
        print(f"  Average trajectory length: {df['total_steps'].mean():.1f} steps")
        
        output_file = "formal_dataset_v0.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Dataset saved to: {output_file}")
        
        # Analyze behavior diversity
        print(f"\n📊 Behavior Analysis:")
        behavior_stats = df.groupby('hidden_cost_style').agg({
            'total_steps': ['mean', 'std', 'count']
        }).round(2)
        print(behavior_stats)
        
        # Save detailed behavior analysis
        behavior_analysis_file = "behavior_analysis_v0.csv"
        behavior_detailed = df.groupby(['hidden_cost_style', 'size', 'initial_distance']).agg({
            'total_steps': ['mean', 'std', 'count'],
            'scenario_id': 'count'
        }).reset_index()
        behavior_detailed.to_csv(behavior_analysis_file, index=False)
        print(f"📈 Behavior analysis saved to: {behavior_analysis_file}")
    else:
        print("\n❌ No scenarios generated successfully")

if __name__ == "__main__":
    main()
"""
Dataset Generator for Goal Recognition Experiments

This module generates datasets containing:
1. Environment layouts (base grids with walls)
2. Hidden cost matrices representing different actor behavior styles
3. Optimal action sequences computed by actors using A* with hidden costs
4. Complete scenario information for goal recognition evaluation

The generated data is used to evaluate observer algorithms' ability to infer
actor goals and behavior patterns from observed actions.
"""

import random
import argparse
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Any

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.target import AstarTarget

def set_all_seed(seed: int) -> None:
    """Set random seeds for reproducible dataset generation."""
    random.seed(seed)
    np.random.seed(seed)
    
# Configuration constants
SEED = 123
set_all_seed(SEED)

# Experiment parameters
GRID_SIZES = [10, 20, 30]  # Different environment sizes
INITIAL_DISTANCES = [3, 5, 7]  # Different starting distances between observer and target
NUM_SCENARIOS = 4  # Number of different start position scenarios per layout
HIDDEN_COST_STYLES = [
    "like_wall",    # Prefers staying close to walls
    "hate_wall",    # Avoids walls
    "like_edge",    # Prefers edges of the environment
    "hate_edge"     # Avoids edges of the environment
]

def generate_hidden_cost_matrices(size: int, base_grid: np.ndarray) -> List[np.ndarray]:
    """
    Generate different hidden cost matrices representing various actor behavior styles.
    
    Args:
        size: Grid size (size x size)
        base_grid: Base environment grid (0=free, 1=wall)
        
    Returns:
        List of 4 hidden cost matrices for different behavior styles
    """
    # Find wall positions
    rows, cols = np.where(base_grid == 1)
    
    # Initialize cost matrices
    cost_matrices = [
        10 * np.ones((size, size)),  # like_wall
        10 * np.ones((size, size)),  # hate_wall  
        10 * np.ones((size, size)),  # like_edge
        10 * np.ones((size, size))   # hate_edge
    ]
    
    for i in range(size):
        for j in range(size):
            if base_grid[i, j] == 0:  # Only process free cells
                # Distance to closest wall
                if len(rows) > 0:
                    wall_dist = np.min(np.abs(rows - i) + np.abs(cols - j))
                else:
                    wall_dist = size  # No walls, use maximum distance
                
                # Distance to edge
                edge_dist = min(i, j, size - i - 1, size - j - 1)
                
                # Set costs based on behavior style
                cost_matrices[0][i, j] = wall_dist                      # like_wall: prefer close to walls
                cost_matrices[1][i, j] = 1 / (wall_dist + 1)           # hate_wall: avoid walls
                cost_matrices[2][i, j] = edge_dist                      # like_edge: prefer edges
                cost_matrices[3][i, j] = 1 / (edge_dist + 1)           # hate_edge: avoid edges
    
    return cost_matrices

def generate_actor_trajectory(env: AGREnv) -> Tuple[List[int], List[np.ndarray], List[Dict]]:
    """
    Generate optimal actor trajectory using A* with hidden costs.
    
    Args:
        env: Configured environment with hidden costs
        
    Returns:
        Tuple of (action_sequence, image_sequence, observation_sequence)
    """
    obs, info = env.reset()
    target_agent = AstarTarget(env)
    
    all_actions = []
    all_imgs = []
    all_obs = []
    
    while not env.unwrapped.is_done():
        # Store current state
        all_obs.append(obs[1])  # Target agent's observation
        all_imgs.append(env.grid.render(tile_size=32, agents=env.unwrapped.agents[1:], highlight_mask=None))
        
        # Compute actions
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        actions[1] = target_agent.compute_action(obs)  # Target uses A* with hidden costs
        all_actions.append(actions[1])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
    
    return all_actions, all_imgs, all_obs

# Main generation loop
results = []

print("🎯 Generating Goal Recognition Dataset...")
print(f"Grid sizes: {GRID_SIZES}")
print(f"Initial distances: {INITIAL_DISTANCES}")
print(f"Scenarios per layout: {NUM_SCENARIOS}")
print(f"Behavior styles: {len(HIDDEN_COST_STYLES)}")
print(f"Total scenarios to generate: {len(GRID_SIZES) * len(INITIAL_DISTANCES) * NUM_SCENARIOS * len(HIDDEN_COST_STYLES)}")
print()

for size in GRID_SIZES:
    print(f"📐 Processing grid size: {size}x{size}")
    
    for layout_id, initial_distance in enumerate(INITIAL_DISTANCES):
        print(f"  🎯 Layout {layout_id}: initial_distance = {initial_distance}")
        
        # Generate base environment layout
        env = AGREnv(size=size)
        obs, info = env.reset()
        base_grid = info['base_grid']
        env.close()
        
        # Generate hidden cost matrices for different behavior styles
        hidden_cost_matrices = generate_hidden_cost_matrices(size, base_grid)
        
        for scenario_id in range(NUM_SCENARIOS):
            print(f"    📍 Scenario {scenario_id}: generating start positions...")
            
            # Generate start positions and goal configuration
            env_setup = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
            obs, info = env_setup.reset()
            
            # Extract scenario configuration
            goals = info['goals']
            goal = info['goal']
            agents_start_pos = info['agents_start_pos']
            agents_start_dir = info['agents_start_dir']
            env_setup.close()
            
            # Verify layout consistency
            assert (info['base_grid'] == base_grid).all(), "Base grid mismatch"
            
            for style_id, hidden_cost in enumerate(hidden_cost_matrices):
                style_name = HIDDEN_COST_STYLES[style_id]
                print(f"      🎭 Style {style_id} ({style_name}): computing optimal trajectory...")
                
                # Create environment with hidden costs for actor behavior
                env_with_costs = AGREnv(
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
                
                # Verify environment setup
                obs, info = env_with_costs.reset()
                assert info['agents_start_pos'] == agents_start_pos, "Start positions mismatch"
                assert info['agents_start_dir'] == agents_start_dir, "Start directions mismatch"
                assert info['goals'] == goals, "Goals mismatch"
                assert info['goal'] == goal, "Target goal mismatch"
                
                # Generate optimal actor trajectory using A* with hidden costs
                all_actions, all_imgs, all_obs = generate_actor_trajectory(env_with_costs)
                env_with_costs.close()
                
                # Store complete scenario data
                scenario_data = {
                    # Environment configuration
                    "base_grid": json.dumps(base_grid.tolist()),
                    "hidden_cost": json.dumps(hidden_cost.tolist()),
                    "size": size,
                    "layout_id": layout_id,
                    "initial_distance": initial_distance,
                    "scenario_id": scenario_id,
                    "hidden_cost_type": style_id,
                    "hidden_cost_style": style_name,
                    
                    # Agent configuration
                    "observer_pos": agents_start_pos[0],
                    "target_pos": agents_start_pos[1],
                    "observer_dir": agents_start_dir[0],
                    "target_dir": agents_start_dir[1],
                    
                    # Goal configuration
                    "goals": goals,
                    "goal": goal,
                    
                    # Generated trajectory data
                    "all_actions": json.dumps([a.value for a in all_actions]),
                    "all_imgs": all_imgs,
                    "all_obs": all_obs,
                    "total_steps": len(all_actions)
                }
                
                results.append(scenario_data)
                print(f"        ✅ Generated {len(all_actions)} steps | Total scenarios: {len(results)}")

# Create DataFrame and save results
print(f"\n💾 Saving {len(results)} scenarios to CSV...")

try:
    df = pd.DataFrame(results)
    
    # Add summary statistics
    print("\n📊 Dataset Summary:")
    print(f"  Total scenarios: {len(df)}")
    print(f"  Grid sizes: {sorted(df['size'].unique())}")
    print(f"  Behavior styles: {df['hidden_cost_style'].unique()}")
    print(f"  Average trajectory length: {df['total_steps'].mean():.1f} steps")
    print(f"  Min/Max trajectory length: {df['total_steps'].min()}/{df['total_steps'].max()} steps")
    
    # Save to CSV
    output_file = "goal_recognition_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset saved successfully to: {output_file}")
    print(f"   File size: {len(df)} rows × {len(df.columns)} columns")
    
except Exception as e:
    print(f"❌ Error saving dataset: {e}")
    print("Attempting to save partial results...")
    
    # Try to save what we have
    if results:
        backup_df = pd.DataFrame(results)
        backup_df.to_csv("dataset_backup.csv", index=False)
        print(f"✅ Partial dataset saved to: dataset_backup.csv ({len(results)} scenarios)")
    else:
        print("❌ No data to save")

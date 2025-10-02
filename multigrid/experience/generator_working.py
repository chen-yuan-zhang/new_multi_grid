"""
Working Dataset Generator for Goal Recognition Experiments

Based on the original generator structure but with improved organization.
Saves as pickle format. Does NOT store pre-rendered images to minimize file size.
Instead, stores base_gri    # Save results as compressed pickle with timestamp
    if results:
        timestamp = int(time.time())
        output_file = f"results_{timestamp}.pkl.gz"
        
        print(f"\n💾 Saving {len(results)} scenarios to pickle format...")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Dataset saved to: {output_file}")
        
        # Delete the last checkpoint file now that final file is saved
        if last_checkpoint_file and os.path.exists(last_checkpoint_file):
            try:
                os.remove(last_checkpoint_file)
                print(f"🗑️  Deleted final checkpoint: {last_checkpoint_file}")
            except Exception as e:
                print(f"⚠️  Could not delete final checkpoint: {e}")
        
        # Print summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"  Total scenarios: {len(results)}")ory states (target_pos, target_dir) so users 
can render images themselves if needed.
Only keeps trajectories with >5 steps for meaningful goal recognition.
"""

import random
import numpy as np
import pandas as pd
import pickle
import gzip
import time
import os

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.target import AstarTarget

def set_all_seed(seed):
    """Set random seeds for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)

# Configuration
SEED = 42
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
    """Generate actor trajectory with hidden costs. Compute full path once with A*, store minimal state info."""
    obs, info = env.reset()
    target_agent = AstarTarget(env)
    
    # Compute the full path once using A* with hidden costs
    try:
        # Trigger A* to compute the full path by calling compute_action once
        first_action = target_agent.compute_action(obs)
        
        # Now target_agent.path contains the full path: [(action, (pos, dir)), ...]
        if target_agent.path is None or len(target_agent.path) <= 1:
            return [], []
        
        # Extract just the actions from the path
        # path format: [(action, (pos, dir)), ...]
        # Note: index 0 is the initial state, so actions start from index 1
        all_actions = [step[0] for step in target_agent.path[1:]]
        
        if len(all_actions) == 0:
            return [], []
            
    except Exception as e:
        print(f"          ⚠️  Error computing path: {e}")
        return [], []
    
    # Reset environment and execute the precomputed path, collecting minimal observations
    obs, info = env.reset()
    all_obs = []
    
    for step, target_action in enumerate(all_actions):
        # Store only position and direction (minimal data to reduce file size)
        # Users can render images themselves using: env.grid.render(tile_size=32, agents=[target_agent], highlight_mask=None)
        # along with base_grid, target_pos, target_dir from the saved data
        all_obs.append({
            'target_pos': obs[1]['target_pos'],
            'target_dir': obs[1]['target_dir']
        })
        
        # Execute action
        actions = {agent.index: 3 for agent in env.unwrapped.agents}
        actions[1] = target_action
        obs, _, _, _, _ = env.step(actions)
        
        # Check if we've reached the goal
        if env.unwrapped.is_done():
            break
    
    return all_actions, all_obs

def main():
    """Generate the dataset."""
    
    # Parameters - systematic dataset generation
    sizes = [10, 12, 15]  # Different environment sizes
    initial_distances = [3, 5, 7]  # Different starting distances
    num_layouts = 30  # Number of different layouts per configuration
    num_scenarios = 5  # Number of different start position scenarios per layout
    style_names = ["like_wall", "hate_wall", "like_edge", "hate_edge"]
    
    results = []
    checkpoint_interval = 9999999  # Save checkpoint every 50 scenarios
    last_checkpoint_file = None  # Track the last checkpoint file to delete it later
    
    print("🎯 Generating Goal Recognition Dataset")
    print(f"Grid sizes: {sizes}")
    print(f"Initial distances: {initial_distances}")
    print(f"Layouts per configuration: {num_layouts}")
    print(f"Scenarios per layout: {num_scenarios}")
    print(f"Behavior styles: {len(style_names)}")
    print(f"Total scenarios: {len(sizes) * len(initial_distances) * num_layouts * num_scenarios * len(style_names)}")
    print(f"Checkpoint interval: every {checkpoint_interval} scenarios")
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
                            
                            # Generate trajectory (no images, just minimal state info for rendering later)
                            all_actions, all_obs = generate_trajectory(env_agents)
                            env_agents.close()
                            
                            # Only keep trajectories with more than 5 steps for meaningful goal recognition
                            if len(all_actions) <= 5:
                                print(f"          ⚠️  Skipped: trajectory too short ({len(all_actions)} steps)")
                                continue
                            
                            # Save result with base_grid included for later rendering
                            # Users can render images using: env.grid.render() with base_grid, target_pos, target_dir
                            result = {
                                'size': size,
                                'layout_id': layout_id,
                                'initial_distance': initial_distance,
                                'scenario_id': scenario_id,
                                'hidden_cost_type': style_id,
                                'base_grid': base_grid,  # Include base_grid for rendering
                                'start_positions': agents_start_pos[1],  # Target position
                                'start_directions': agents_start_dir[1],  # Target direction
                                'goals': goals,
                                'goal': goal,
                                'all_actions': all_actions,  # Keep as Action objects
                                'all_obs': all_obs  # Only target_pos and target_dir - enough to reconstruct trajectory
                            }
                            
                            results.append(result)
                            print(f"          ✅ Generated {len(all_actions)} steps | Total: {len(results)}")
                            
                            # Periodic checkpoint to avoid losing all progress
                            if len(results) % checkpoint_interval == 0:
                                # Delete previous checkpoint before creating new one
                                if last_checkpoint_file and os.path.exists(last_checkpoint_file):
                                    try:
                                        os.remove(last_checkpoint_file)
                                        print(f"🗑️  Deleted old checkpoint: {last_checkpoint_file}")
                                    except Exception as e:
                                        print(f"⚠️  Could not delete old checkpoint: {e}")
                                
                                # Save new checkpoint
                                timestamp = int(time.time())
                                checkpoint_file = f"checkpoint_{timestamp}.pkl.gz"
                                print(f"\n💾 Checkpoint: Saving {len(results)} scenarios to {checkpoint_file}...")
                                with gzip.open(checkpoint_file, 'wb') as f:
                                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                                print(f"✅ Checkpoint saved\n")
                                last_checkpoint_file = checkpoint_file
                            
                        except KeyboardInterrupt:
                            print(f"\n⚠️  Interrupted by user. Saving {len(results)} results collected so far...")
                            raise  # Re-raise to exit and save
                        except MemoryError:
                            print(f"          ❌ Memory Error: Out of memory, skipping this scenario")
                            continue
                        except Exception as e:
                            import traceback
                            print(f"          ❌ Error: {e}")
                            print(f"          Stack trace: {traceback.format_exc()}")
                            continue
                        finally:
                            # Ensure environment is closed even if error occurs
                            try:
                                env_agents.close()
                            except:
                                pass

    # Save results as compressed pickle (matching original_generator.py format)
    if results:
        output_file = "results.pkl.gz"
        
        print(f"\n� Saving {len(results)} scenarios to pickle format...")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Dataset saved to: {output_file}")
        
        # Print summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"  Total scenarios: {len(results)}")
        
        sizes = [r['size'] for r in results]
        print(f"  Grid sizes: {sorted(set(sizes))}")
        
        cost_types = [r['hidden_cost_type'] for r in results]
        print(f"  Hidden cost types: {sorted(set(cost_types))}")
        
        trajectory_lengths = [len(r['all_actions']) for r in results]
        print(f"  Average trajectory length: {np.mean(trajectory_lengths):.1f} steps")
        print(f"  Min/Max trajectory length: {min(trajectory_lengths)}/{max(trajectory_lengths)} steps")
        
        # Count by behavior style
        print(f"\n📊 Scenarios by hidden cost type:")
        for cost_type in sorted(set(cost_types)):
            count = cost_types.count(cost_type)
            style_name = style_names[cost_type]
            print(f"  Type {cost_type} ({style_name}): {count} scenarios")
        
    else:
        print("\n❌ No scenarios generated successfully")

if __name__ == "__main__":
    main()
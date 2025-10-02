#!/usr/bin/env python3
"""
Test the generator with a small subset to verify the structure works
"""

# Import the modified generator
import sys
sys.path.append('multigrid/experience')

# Temporarily modify the generator for testing
import generator_working

# Backup original values
original_sizes = generator_working.main.__code__.co_consts
original_main = generator_working.main

def test_generator():
    """Test generator with small parameters"""
    
    # Override parameters for testing
    def small_main():
        import random
        import numpy as np
        import pandas as pd
        import json
        from multigrid.envs.goal_prediction import AGREnv
        from multigrid.gr_pursuer.agents.target import AstarTarget
        from generator_working import generate_hidden_cost_matrices, generate_trajectory
        
        # Small test parameters
        sizes = [10]  # Just one size
        initial_distances = [3]  # Just one distance
        num_layouts = 2  # Just 2 layouts
        num_scenarios = 1  # Just 1 scenario per layout
        style_names = ["like_wall", "hate_wall"]  # Just 2 styles
        
        results = []
        
        print("🎯 Testing Generator Configuration")
        print(f"Total scenarios: {len(sizes) * len(initial_distances) * num_layouts * num_scenarios * len(style_names)}")
        
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
                        print(f"      📋 Scenario {scenario_id + 1}/{num_scenarios}")
                        
                        # Generate start positions and goals
                        env_grid = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
                        obs, info = env_grid.reset()
                        
                        goals = info['goals']
                        goal = info['goal']
                        agents_start_pos = info['agents_start_pos']
                        agents_start_dir = info['agents_start_dir']
                        env_grid.close()
                        
                        for style_id, style_name in enumerate(style_names):
                            hidden_cost = hidden_costs[style_id]
                            print(f"        🎭 Style {style_id} ({style_name})")
                            
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
                                    'size': size,
                                    'layout_id': layout_id,
                                    'initial_distance': initial_distance,
                                    'scenario_id': scenario_id,
                                    'hidden_cost_type': style_id,
                                    'hidden_cost_style': style_name,
                                    'total_steps': len(all_actions)
                                }
                                
                                results.append(result)
                                print(f"          ✅ Generated {len(all_actions)} steps | Total: {len(results)}")
                                
                            except Exception as e:
                                print(f"          ❌ Error: {e}")
                                continue
        
        # Show results
        if results:
            df = pd.DataFrame(results)
            print(f"\n📊 Test Results:")
            print(f"  Total scenarios: {len(df)}")
            print(f"  Grid sizes: {sorted(df['size'].unique())}")
            print(f"  Layouts per config: {sorted(df['layout_id'].unique())}")
            print(f"  Behavior styles: {list(df['hidden_cost_style'].unique())}")
            print(f"  Average trajectory length: {df['total_steps'].mean():.1f} steps")
            print(df[['size', 'layout_id', 'hidden_cost_style', 'total_steps']])
            return True
        else:
            print("\n❌ No scenarios generated successfully")
            return False
    
    return small_main()

if __name__ == "__main__":
    success = test_generator()
    print(f"\n🏁 Test {'PASSED' if success else 'FAILED'}")
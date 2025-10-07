#!/usr/bin/env python3
"""
Validate that all plans in the dataset can achieve their goals.
"""

import sys
import os
import numpy as np
import json
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action


def validate_scenario(row, scenario_idx):
    """
    Validate a single scenario - check if the stored actions reach the goal.
    
    Returns:
        dict with validation results
    """
    try:
        # Parse data
        base_grid = np.array(json.loads(row['base_grid']))
        hidden_cost = np.array(json.loads(row['hidden_cost']))
        goals = eval(row['goals'])
        goal = eval(row['goal'])
        observer_pos = eval(row['observer_pos'])
        target_pos = eval(row['target_pos'])
        observer_dir = row['observer_dir']
        target_dir = row['target_dir']
        stored_actions = [Action(v) for v in json.loads(row['all_actions'])]
        
        # Create environment
        env = AGREnv(
            base_grid=base_grid,
            goals=goals,
            goal=goal,
            hidden_cost=hidden_cost,
            enable_hidden_cost=True,
            agents_start_pos=[observer_pos, target_pos],
            agents_start_dir=[observer_dir, target_dir],
            render_mode=None
        )
        
        obs, info = env.reset()
        
        # Track trajectory
        positions = [tuple(target_pos)]
        
        # Execute all stored actions
        for action in stored_actions:
            obs, rew, term, trunc, info = env.step({0: Action.stay, 1: action})
            new_pos = tuple(env.agents[1].pos)
            positions.append(new_pos)
            
            # Check if reached goal
            if new_pos == goal:
                break
        
        final_pos = tuple(env.agents[1].pos)
        reached_goal = (final_pos == goal)
        unique_positions = len(set(positions))
        
        return {
            'scenario_idx': scenario_idx,
            'valid': True,
            'reached_goal': reached_goal,
            'final_pos': final_pos,
            'goal_pos': goal,
            'start_pos': tuple(target_pos),
            'num_actions': len(stored_actions),
            'unique_positions': unique_positions,
            'distance_to_goal': abs(final_pos[0] - goal[0]) + abs(final_pos[1] - goal[1]),
            'behavior': row['hidden_cost_style'],
            'size': row['size']
        }
        
    except Exception as e:
        return {
            'scenario_idx': scenario_idx,
            'valid': False,
            'error': str(e),
            'reached_goal': False
        }


def main():
    print("📖 Loading dataset...")
    df = pd.read_csv('results_test_new.csv')
    print(f"   Found {len(df)} scenarios\n")
    
    print("🔍 Validating all scenarios...")
    print("   This may take a few minutes...\n")
    
    results = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Processing scenario {idx + 1}/{len(df)}...")
        
        result = validate_scenario(row, idx)
        results.append(result)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("📊 VALIDATION RESULTS")
    print("=" * 80)
    
    valid_scenarios = [r for r in results if r['valid']]
    invalid_scenarios = [r for r in results if not r['valid']]
    
    print(f"✅ Valid scenarios: {len(valid_scenarios)}/{len(results)} ({100*len(valid_scenarios)/len(results):.1f}%)")
    
    if invalid_scenarios:
        print(f"❌ Invalid scenarios (parsing errors): {len(invalid_scenarios)}")
        print(f"   First few errors:")
        for r in invalid_scenarios[:3]:
            print(f"      Scenario {r['scenario_idx']}: {r.get('error', 'Unknown error')}")
    print()
    
    # Goal achievement statistics
    reached_goal = [r for r in valid_scenarios if r['reached_goal']]
    not_reached = [r for r in valid_scenarios if not r['reached_goal']]
    
    print("🎯 GOAL ACHIEVEMENT:")
    print(f"   Reached goal: {len(reached_goal)}/{len(valid_scenarios)} ({100*len(reached_goal)/len(valid_scenarios):.1f}%)")
    print(f"   Did NOT reach goal: {len(not_reached)}/{len(valid_scenarios)} ({100*len(not_reached)/len(valid_scenarios):.1f}%)")
    print()
    
    if reached_goal:
        actions_counts = [r['num_actions'] for r in reached_goal]
        print(f"   Successful trajectories:")
        print(f"      Average actions: {np.mean(actions_counts):.1f}")
        print(f"      Min/Max actions: {min(actions_counts)}/{max(actions_counts)}")
        print(f"      Average unique positions: {np.mean([r['unique_positions'] for r in reached_goal]):.1f}")
    print()
    
    if not_reached:
        print(f"   Failed trajectories:")
        print(f"      Average unique positions: {np.mean([r['unique_positions'] for r in not_reached]):.1f}")
        print(f"      Average distance to goal: {np.mean([r['distance_to_goal'] for r in not_reached]):.1f}")
        
        # Show distribution of final distances
        distances = [r['distance_to_goal'] for r in not_reached]
        print(f"\n   Distribution of distances to goal:")
        distance_counts = Counter(distances)
        for dist in sorted(distance_counts.keys())[:10]:
            count = distance_counts[dist]
            pct = 100 * count / len(not_reached)
            print(f"      Distance {dist}: {count} scenarios ({pct:.1f}%)")
        
        # Show some examples
        print(f"\n   Sample failed scenarios:")
        for r in not_reached[:5]:
            print(f"      Scenario {r['scenario_idx']}: {r['behavior']}, size={r['size']}")
            print(f"         Start: {r['start_pos']}, Goal: {r['goal_pos']}, Final: {r['final_pos']}")
            print(f"         Actions: {r['num_actions']}, Unique positions: {r['unique_positions']}, Distance: {r['distance_to_goal']}")
    print()
    
    # Statistics by behavior type
    print("📊 BY BEHAVIOR TYPE:")
    behaviors = set(r['behavior'] for r in valid_scenarios)
    for behavior in sorted(behaviors):
        behavior_scenarios = [r for r in valid_scenarios if r['behavior'] == behavior]
        behavior_success = [r for r in behavior_scenarios if r['reached_goal']]
        success_rate = 100 * len(behavior_success) / len(behavior_scenarios) if behavior_scenarios else 0
        print(f"   {behavior:15s}: {len(behavior_success):4d}/{len(behavior_scenarios):4d} reached goal ({success_rate:5.1f}%)")
    print()
    
    # Statistics by size
    print("📊 BY GRID SIZE:")
    sizes = sorted(set(r['size'] for r in valid_scenarios))
    for size in sizes:
        size_scenarios = [r for r in valid_scenarios if r['size'] == size]
        size_success = [r for r in size_scenarios if r['reached_goal']]
        success_rate = 100 * len(size_success) / len(size_scenarios) if size_scenarios else 0
        print(f"   Size {size:2d}x{size:2d}: {len(size_success):4d}/{len(size_scenarios):4d} reached goal ({success_rate:5.1f}%)")
    print()
    
    # Summary
    print("=" * 80)
    print("💡 SUMMARY:")
    print("=" * 80)
    
    if len(reached_goal) == len(valid_scenarios):
        print("✅ ALL trajectories successfully reach their goals!")
        print("   The dataset is valid and ready for evaluation.")
    else:
        pct_failed = 100 * len(not_reached) / len(valid_scenarios)
        print(f"⚠️  {len(not_reached)} trajectories ({pct_failed:.1f}%) do NOT reach their goals")
        print("   This could indicate:")
        print("   - Truncated trajectories (stopped before reaching goal)")
        print("   - Invalid action sequences in the dataset")
        print("   - Environment simulation issues")
    
    # Save detailed results
    results_df = pd.DataFrame(valid_scenarios)
    results_df.to_csv('validation_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: validation_results.csv")


if __name__ == "__main__":
    main()

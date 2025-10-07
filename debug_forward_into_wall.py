#!/usr/bin/env python3
"""
Debug script to check if trajectories with forward-into-wall actions actually reach the goal.
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import json
from collections import Counter
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.gr_pursuer.astar import get_successor


def check_trajectory_validity(scenario_config: Dict[str, Any], scenario_idx: int) -> Dict[str, Any]:
    """
    Check if a trajectory successfully reaches the goal and analyze forward-into-wall actions.
    """
    # Extract configuration
    base_grid = scenario_config['base_grid']
    goals = scenario_config['goals']
    goal = scenario_config['goal']
    hidden_cost = scenario_config['hidden_cost']
    observer_pos = scenario_config['observer_pos']
    target_pos = scenario_config['target_pos']
    observer_dir = scenario_config['observer_dir']
    target_dir = scenario_config['target_dir']
    target_actions = scenario_config['target_actions']
    
    # Setup environment
    agents_start_pos = [observer_pos, target_pos]
    agents_start_dir = [observer_dir, target_dir]
    
    env = AGREnv(
        base_grid=base_grid,
        goals=goals, 
        goal=goal,
        hidden_cost=hidden_cost,
        enable_hidden_cost=True,
        agents_start_pos=agents_start_pos,
        agents_start_dir=agents_start_dir,
        render_mode=None
    )
    
    observation, info = env.reset()
    
    # Track statistics
    forward_into_wall_count = 0
    forward_into_wall_steps = []
    positions_over_time = [tuple(env.agents[1].pos)]
    
    # Execute trajectory
    for step, target_action in enumerate(target_actions):
        # Get current state
        current_pos = tuple(env.agents[1].pos)
        current_dir = int(env.agents[1].dir)
        pos_state_param = (env.agents[1].pos, current_dir)
        
        # Get possible successors (what actions are actually available)
        successors = get_successor(env, pos_state_param)
        available_actions = [int(action) for action, _ in successors]
        num_available = len(successors)
        
        # Check what action the agent is trying to do
        target_action_int = int(target_action)
        target_action_name = Action(target_action_int).name
        
        # Is forward available?
        forward_available = int(Action.forward) in available_actions
        
        # Is the agent trying to go forward when it's blocked?
        if target_action_int == int(Action.forward) and not forward_available:
            forward_into_wall_count += 1
            forward_into_wall_steps.append({
                'step': step,
                'pos': current_pos,
                'dir': current_dir,
                'available': [Action(a).name for a in available_actions],
                'tried': target_action_name
            })
        
        # Execute the action
        observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
        
        # Track new position
        new_pos = tuple(env.agents[1].pos)
        positions_over_time.append(new_pos)
    
    # Check if reached goal
    final_pos = tuple(env.agents[1].pos)
    goal_pos = tuple(goal)
    reached_goal = (final_pos == goal_pos)
    
    # Check if agent made progress (moved at all)
    unique_positions = len(set(positions_over_time))
    made_progress = unique_positions > 1
    
    return {
        'scenario_idx': scenario_idx,
        'reached_goal': reached_goal,
        'final_pos': final_pos,
        'goal_pos': goal_pos,
        'start_pos': tuple(target_pos),
        'trajectory_length': len(target_actions),
        'forward_into_wall_count': forward_into_wall_count,
        'forward_into_wall_steps': forward_into_wall_steps,
        'unique_positions': unique_positions,
        'made_progress': made_progress,
        'positions': positions_over_time
    }


def main():
    dataset_path = "results_test_new.csv"
    
    print(f"📖 Loading dataset: {dataset_path}")
    scenarios_df = pd.read_csv(dataset_path)
    print(f"   Found {len(scenarios_df)} scenarios")
    
    # Parse scenarios
    scenarios = []
    for idx, row in scenarios_df.iterrows():
        try:
            config = {
                'base_grid': np.array(json.loads(row['base_grid'])),
                'goals': eval(row['goals']),
                'goal': eval(row['goal']),
                'hidden_cost': np.array(json.loads(row['hidden_cost'])),
                'observer_pos': eval(row['observer_pos']),
                'target_pos': eval(row['target_pos']),
                'observer_dir': row['observer_dir'],
                'target_dir': row['target_dir'],
                'target_actions': [Action(v) for v in json.loads(row['all_actions'])]
            }
            scenarios.append((idx, config))
        except Exception as e:
            continue
    
    print(f"✅ Parsed {len(scenarios)} scenarios successfully")
    print()
    
    # Analyze all scenarios
    print("🔍 Analyzing all scenarios for forward-into-wall actions...")
    
    results = []
    scenarios_with_forward_into_wall = []
    scenarios_without_goal = []
    
    for idx, scenario in scenarios:
        result = check_trajectory_validity(scenario, idx)
        results.append(result)
        
        if result['forward_into_wall_count'] > 0:
            scenarios_with_forward_into_wall.append(result)
        
        if not result['reached_goal']:
            scenarios_without_goal.append(result)
    
    print("\n" + "="*80)
    print("📊 TRAJECTORY VALIDITY ANALYSIS")
    print("="*80)
    print(f"Total scenarios analyzed: {len(results)}")
    print()
    
    # Goal reaching statistics
    reached_goal_count = sum(1 for r in results if r['reached_goal'])
    print(f"✅ Scenarios that reached goal: {reached_goal_count} ({100*reached_goal_count/len(results):.2f}%)")
    print(f"❌ Scenarios that did NOT reach goal: {len(results)-reached_goal_count} ({100*(len(results)-reached_goal_count)/len(results):.2f}%)")
    print()
    
    # Forward into wall statistics
    with_forward_into_wall = len(scenarios_with_forward_into_wall)
    print(f"🧱 Scenarios with 'forward into wall' actions: {with_forward_into_wall} ({100*with_forward_into_wall/len(results):.2f}%)")
    
    if with_forward_into_wall > 0:
        avg_forward_into_wall = np.mean([r['forward_into_wall_count'] for r in scenarios_with_forward_into_wall])
        max_forward_into_wall = max([r['forward_into_wall_count'] for r in scenarios_with_forward_into_wall])
        print(f"   Average forward-into-wall actions per affected scenario: {avg_forward_into_wall:.2f}")
        print(f"   Max forward-into-wall actions in a single scenario: {max_forward_into_wall}")
    print()
    
    # Cross-tabulation
    forward_and_reached = sum(1 for r in scenarios_with_forward_into_wall if r['reached_goal'])
    forward_and_not_reached = sum(1 for r in scenarios_with_forward_into_wall if not r['reached_goal'])
    
    print("📊 CROSS-TABULATION:")
    print(f"   Forward-into-wall AND reached goal: {forward_and_reached}")
    print(f"   Forward-into-wall AND did NOT reach goal: {forward_and_not_reached}")
    print()
    
    if forward_and_not_reached > 0:
        print("⚠️  WARNING: Some trajectories have forward-into-wall actions AND don't reach the goal!")
        print("   This suggests the trajectories in the dataset may be invalid/incomplete.")
    
    # Show examples
    if scenarios_without_goal:
        print("\n" + "="*80)
        print("🔍 SAMPLE SCENARIOS THAT DID NOT REACH GOAL:")
        print("="*80)
        for i, result in enumerate(scenarios_without_goal[:5]):  # Show first 5
            print(f"\nScenario {result['scenario_idx']}:")
            print(f"   Start: {result['start_pos']}")
            print(f"   Goal: {result['goal_pos']}")
            print(f"   Final: {result['final_pos']}")
            print(f"   Trajectory length: {result['trajectory_length']}")
            print(f"   Unique positions visited: {result['unique_positions']}")
            print(f"   Forward-into-wall actions: {result['forward_into_wall_count']}")
            
            if result['forward_into_wall_count'] > 0:
                print(f"   Forward-into-wall details:")
                for detail in result['forward_into_wall_steps'][:3]:  # Show first 3
                    print(f"      Step {detail['step']}: at {detail['pos']} facing dir {detail['dir']}")
                    print(f"         Available: {detail['available']}, Tried: {detail['tried']}")
    
    # Check if trajectories are actually making progress
    no_progress = sum(1 for r in results if not r['made_progress'])
    if no_progress > 0:
        print(f"\n⚠️  {no_progress} scenarios made NO progress (stayed in same position)")
    
    # Summary
    print("\n" + "="*80)
    print("💡 KEY FINDINGS:")
    print("="*80)
    if reached_goal_count == len(results):
        print("✅ All trajectories successfully reach their goals")
    else:
        print(f"❌ {len(results)-reached_goal_count} trajectories do NOT reach their goals")
        print("   This indicates potential issues with the dataset trajectories")
    
    if with_forward_into_wall > 0:
        pct_forward_wall = 100 * with_forward_into_wall / len(results)
        print(f"\n🧱 {pct_forward_wall:.1f}% of scenarios contain 'forward into wall' actions")
        print("   This explains the low random baseline accuracy in 3-action states!")
        
        if forward_and_not_reached > 0:
            print("\n⚠️  CRITICAL: Some trajectories with forward-into-wall DON'T reach the goal")
            print("   This suggests the trajectories may be from failed planning attempts")
            print("   or the dataset includes incomplete/invalid trajectories")


if __name__ == "__main__":
    main()

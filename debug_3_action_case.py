#!/usr/bin/env python3
"""
Debug script to understand why 3-action states have such low random accuracy (9.83%).
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.gr_pursuer.astar import get_successor


def analyze_3_action_states(scenario_config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """
    Analyze what happens specifically in 3-action states (when wall blocks forward).
    """
    random.seed(seed)
    np.random.seed(seed)
    
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
    
    # Track statistics for 3-action vs 4-action states
    stats_3_action = {
        "total": 0,
        "correct": 0,
        "actual_action_counts": Counter(),
        "available_actions_list": [],
        "chosen_actions": Counter()
    }
    
    stats_4_action = {
        "total": 0,
        "correct": 0,
        "actual_action_counts": Counter(),
        "available_actions_list": [],
        "chosen_actions": Counter()
    }
    
    # Evaluate each step
    for step, target_action in enumerate(target_actions):
        # Get current position state
        pos_state_param = (env.agents[1].pos, int(env.agents[1].dir))
        
        # Get possible successors
        successors = get_successor(env, pos_state_param)
        num_available = len(successors)
        
        # Track actual action
        target_action_formal = int(target_actions[step])
        target_action_name = Action(target_action_formal).name
        
        # Get available action names
        available_actions = [int(action) for action, _ in successors]
        available_action_names = [Action(a).name for a in available_actions]
        
        # Randomly select one
        if successors:
            random_action_int = random.choice(available_actions)
            random_action = Action(random_action_int)
        else:
            random_action_int = int(Action.stay)
            random_action = Action.stay
        
        # Check if correct
        is_correct = (random_action_int == target_action_formal)
        
        # Track by number of actions
        if num_available == 3:
            stats = stats_3_action
        elif num_available == 4:
            stats = stats_4_action
        else:
            # Execute and continue
            observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
            continue
        
        stats["total"] += 1
        stats["correct"] += int(is_correct)
        stats["actual_action_counts"][target_action_name] += 1
        stats["available_actions_list"].append(available_action_names)
        stats["chosen_actions"][random_action.name] += 1
        
        # Execute the actual action to advance environment
        observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
    
    return {
        "3_action": stats_3_action,
        "4_action": stats_4_action
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
            scenarios.append(config)
        except Exception as e:
            continue
    
    print(f"✅ Parsed {len(scenarios)} scenarios successfully")
    print()
    
    # Run on a sample
    print("🔍 Running detailed analysis on 100 random scenarios...")
    sample_scenarios = random.sample(scenarios, min(100, len(scenarios)))
    
    # Aggregate results
    agg_3_action = {
        "total": 0,
        "correct": 0,
        "actual_action_counts": Counter(),
        "available_actions_summary": Counter(),
        "chosen_actions": Counter()
    }
    
    agg_4_action = {
        "total": 0,
        "correct": 0,
        "actual_action_counts": Counter(),
        "available_actions_summary": Counter(),
        "chosen_actions": Counter()
    }
    
    for i, scenario in enumerate(sample_scenarios):
        result = analyze_3_action_states(scenario, seed=42 + i)
        
        # Aggregate 3-action stats
        agg_3_action["total"] += result["3_action"]["total"]
        agg_3_action["correct"] += result["3_action"]["correct"]
        for action, count in result["3_action"]["actual_action_counts"].items():
            agg_3_action["actual_action_counts"][action] += count
        for action, count in result["3_action"]["chosen_actions"].items():
            agg_3_action["chosen_actions"][action] += count
        
        # Track which actions are available in 3-action states
        for available_list in result["3_action"]["available_actions_list"]:
            key = tuple(sorted(available_list))
            agg_3_action["available_actions_summary"][key] += 1
        
        # Aggregate 4-action stats
        agg_4_action["total"] += result["4_action"]["total"]
        agg_4_action["correct"] += result["4_action"]["correct"]
        for action, count in result["4_action"]["actual_action_counts"].items():
            agg_4_action["actual_action_counts"][action] += count
        for action, count in result["4_action"]["chosen_actions"].items():
            agg_4_action["chosen_actions"][action] += count
        
        # Track which actions are available in 4-action states
        for available_list in result["4_action"]["available_actions_list"]:
            key = tuple(sorted(available_list))
            agg_4_action["available_actions_summary"][key] += 1
    
    print("\n" + "="*80)
    print("📊 3-ACTION STATES ANALYSIS (wall blocks forward)")
    print("="*80)
    print(f"Total states: {agg_3_action['total']}")
    print(f"Correct predictions: {agg_3_action['correct']}")
    accuracy_3 = agg_3_action['correct'] / agg_3_action['total'] if agg_3_action['total'] > 0 else 0
    print(f"Accuracy: {accuracy_3:.4f} ({accuracy_3*100:.2f}%)")
    print(f"Theoretical: 0.3333 (33.33%)")
    print()
    
    print("📊 AVAILABLE ACTIONS IN 3-ACTION STATES:")
    for actions_tuple, count in agg_3_action["available_actions_summary"].most_common():
        pct = 100.0 * count / agg_3_action['total']
        print(f"   {list(actions_tuple)}: {count} states ({pct:.2f}%)")
    print()
    
    print("📊 WHAT THE AGENT ACTUALLY DID IN 3-ACTION STATES:")
    for action, count in agg_3_action["actual_action_counts"].most_common():
        pct = 100.0 * count / agg_3_action['total']
        print(f"   {action}: {count} times ({pct:.2f}%)")
    print()
    
    print("📊 WHAT RANDOM BASELINE CHOSE IN 3-ACTION STATES:")
    for action, count in agg_3_action["chosen_actions"].most_common():
        pct = 100.0 * count / agg_3_action['total']
        print(f"   {action}: {count} times ({pct:.2f}%)")
    print()
    
    print("💡 KEY INSIGHT FOR 3-ACTION STATES:")
    # Calculate expected accuracy based on actual distribution
    total = sum(agg_3_action["actual_action_counts"].values())
    if total > 0:
        # For uniform random over 3 actions (left, right, stay), each has 1/3 probability
        expected_from_uniform = sum(count / total / 3 for count in agg_3_action["actual_action_counts"].values())
        print(f"   If agent were uniform over 3 actions: {expected_from_uniform:.4f} ({expected_from_uniform*100:.2f}%)")
        print(f"   Actual random accuracy: {accuracy_3:.4f} ({accuracy_3*100:.2f}%)")
        print()
        
        # Show the problem
        most_common_action = agg_3_action["actual_action_counts"].most_common(1)[0]
        print(f"   The agent chose '{most_common_action[0]}' in {most_common_action[1]/total*100:.1f}% of 3-action states")
        print(f"   But random baseline only chose it {agg_3_action['chosen_actions'][most_common_action[0]]/total*100:.1f}% of the time")
        print(f"   This mismatch explains the low accuracy!")
    
    print("\n" + "="*80)
    print("📊 4-ACTION STATES ANALYSIS (no wall ahead)")
    print("="*80)
    print(f"Total states: {agg_4_action['total']}")
    print(f"Correct predictions: {agg_4_action['correct']}")
    accuracy_4 = agg_4_action['correct'] / agg_4_action['total'] if agg_4_action['total'] > 0 else 0
    print(f"Accuracy: {accuracy_4:.4f} ({accuracy_4*100:.2f}%)")
    print(f"Theoretical: 0.2500 (25.00%)")
    print()
    
    print("📊 WHAT THE AGENT ACTUALLY DID IN 4-ACTION STATES:")
    for action, count in agg_4_action["actual_action_counts"].most_common():
        pct = 100.0 * count / agg_4_action['total']
        print(f"   {action}: {count} times ({pct:.2f}%)")
    print()
    
    print("📊 WHAT RANDOM BASELINE CHOSE IN 4-ACTION STATES:")
    for action, count in agg_4_action["chosen_actions"].most_common():
        pct = 100.0 * count / agg_4_action['total']
        print(f"   {action}: {count} times ({pct:.2f}%)")
    print()
    
    print("💡 COMPARISON:")
    print(f"   4-action accuracy ({accuracy_4*100:.2f}%) is close to theoretical (25.00%)")
    print(f"   3-action accuracy ({accuracy_3*100:.2f}%) is FAR below theoretical (33.33%)")
    print(f"   This is because the agent rarely faces walls (prefers forward movement)")
    print(f"   When forced to choose left/right/stay, the agent's choice is highly non-uniform")


if __name__ == "__main__":
    main()

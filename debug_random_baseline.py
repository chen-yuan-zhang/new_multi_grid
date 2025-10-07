#!/usr/bin/env python3
"""
Debug script to understand why random baseline accuracy is 21.17% instead of ~27%.
This will track the distribution of actions chosen vs actions available.
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.gr_pursuer.astar import get_successor


def debug_random_on_scenario(scenario_config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """
    Debug random baseline on a single scenario, tracking action distributions.
    
    Returns:
        Dictionary with debugging statistics
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
    
    # Track statistics
    num_actions_available = []
    actual_action_counts = Counter()
    chosen_action_counts = Counter()
    correct_matches = 0
    total_steps = 0
    
    # Track by number of available actions
    accuracy_by_num_actions = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Evaluate each step
    for step, target_action in enumerate(target_actions):
        # Get current position state
        pos_state_param = (env.agents[1].pos, int(env.agents[1].dir))
        
        # Get possible successors
        successors = get_successor(env, pos_state_param)
        num_available = len(successors)
        num_actions_available.append(num_available)
        
        # Track actual action
        target_action_formal = int(target_actions[step])
        actual_action_counts[target_action_formal] += 1
        
        # Randomly select one of the available actions
        if successors:
            available_actions = [int(action) for action, _ in successors]
            random_action_int = random.choice(available_actions)
            random_action = Action(random_action_int)
        else:
            random_action_int = int(Action.stay)
            random_action = Action.stay
        
        chosen_action_counts[random_action_int] += 1
        
        # Check if correct
        is_correct = (random_action_int == target_action_formal)
        if is_correct:
            correct_matches += 1
        
        # Track by number of available actions
        accuracy_by_num_actions[num_available]["correct"] += int(is_correct)
        accuracy_by_num_actions[num_available]["total"] += 1
        total_steps += 1
        
        # Execute the actual action to advance environment
        observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
    
    # Calculate statistics
    avg_correctness = correct_matches / total_steps if total_steps > 0 else 0.0
    avg_num_actions = np.mean(num_actions_available) if num_actions_available else 0.0
    theoretical_accuracy = 1.0 / avg_num_actions if avg_num_actions > 0 else 0.0
    
    return {
        "accuracy": avg_correctness,
        "total_steps": total_steps,
        "correct_matches": correct_matches,
        "avg_num_actions": avg_num_actions,
        "theoretical_accuracy": theoretical_accuracy,
        "num_actions_dist": Counter(num_actions_available),
        "actual_action_counts": actual_action_counts,
        "chosen_action_counts": chosen_action_counts,
        "accuracy_by_num_actions": dict(accuracy_by_num_actions)
    }


def main():
    import json
    
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
            print(f"⚠️  Warning: Could not parse scenario {idx}: {e}")
            continue
    
    print(f"✅ Parsed {len(scenarios)} scenarios successfully")
    print()
    
    # Run on a sample
    print("🔍 Running debug analysis on 100 random scenarios...")
    sample_scenarios = random.sample(scenarios, min(100, len(scenarios)))
    
    all_results = []
    for i, scenario in enumerate(sample_scenarios):
        result = debug_random_on_scenario(scenario, seed=42 + i)
        all_results.append(result)
    
    # Aggregate results
    total_accuracy = np.mean([r["accuracy"] for r in all_results])
    total_theoretical = np.mean([r["theoretical_accuracy"] for r in all_results])
    total_avg_actions = np.mean([r["avg_num_actions"] for r in all_results])
    
    # Aggregate action distributions
    all_num_actions = Counter()
    all_actual_actions = Counter()
    all_chosen_actions = Counter()
    all_accuracy_by_num = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in all_results:
        for num, count in result["num_actions_dist"].items():
            all_num_actions[num] += count
        for action, count in result["actual_action_counts"].items():
            all_actual_actions[action] += count
        for action, count in result["chosen_action_counts"].items():
            all_chosen_actions[action] += count
        for num, stats in result["accuracy_by_num_actions"].items():
            all_accuracy_by_num[num]["correct"] += stats["correct"]
            all_accuracy_by_num[num]["total"] += stats["total"]
    
    print("\n" + "="*80)
    print("📊 AGGREGATE RESULTS")
    print("="*80)
    print(f"Actual random accuracy: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
    print(f"Theoretical accuracy (1/avg_actions): {total_theoretical:.4f} ({total_theoretical*100:.2f}%)")
    print(f"Average actions available per state: {total_avg_actions:.2f}")
    print(f"Gap: {(total_theoretical - total_accuracy)*100:.2f} percentage points")
    print()
    
    print("📊 DISTRIBUTION OF AVAILABLE ACTIONS:")
    total_states = sum(all_num_actions.values())
    for num in sorted(all_num_actions.keys()):
        count = all_num_actions[num]
        pct = 100.0 * count / total_states
        print(f"   {num} actions: {count:6d} states ({pct:5.2f}%)")
    print()
    
    print("📊 ACCURACY BY NUMBER OF AVAILABLE ACTIONS:")
    for num in sorted(all_accuracy_by_num.keys()):
        stats = all_accuracy_by_num[num]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        theoretical = 1.0 / num if num > 0 else 0.0
        gap = (theoretical - accuracy) * 100
        print(f"   {num} actions: {accuracy:.4f} ({accuracy*100:.2f}%) vs theoretical {theoretical:.4f} ({theoretical*100:.2f}%)")
        print(f"              Gap: {gap:+.2f} pp from {stats['total']} states")
    print()
    
    print("📊 ACTUAL ACTION DISTRIBUTION (what agent did):")
    action_names = {0: "left", 1: "right", 2: "forward", 3: "stay"}
    total_actual = sum(all_actual_actions.values())
    for action in sorted(all_actual_actions.keys()):
        count = all_actual_actions[action]
        pct = 100.0 * count / total_actual
        print(f"   {action_names.get(action, action)}: {count:6d} ({pct:5.2f}%)")
    print()
    
    print("📊 CHOSEN ACTION DISTRIBUTION (what random baseline chose):")
    total_chosen = sum(all_chosen_actions.values())
    for action in sorted(all_chosen_actions.keys()):
        count = all_chosen_actions[action]
        pct = 100.0 * count / total_chosen
        print(f"   {action_names.get(action, action)}: {count:6d} ({pct:5.2f}%)")
    print()
    
    # Check if actual actions are biased
    print("📊 COMPARISON:")
    print(f"   If random baseline matches actual, expected accuracy = 1/avg_actions = {total_theoretical:.4f}")
    print(f"   Actual accuracy = {total_accuracy:.4f}")
    print()
    print("💡 KEY INSIGHT:")
    if total_accuracy < total_theoretical:
        print(f"   The {(total_theoretical - total_accuracy)*100:.2f} pp gap suggests:")
        print(f"   - The actual agent's actions are NOT uniformly distributed")
        print(f"   - The agent prefers certain actions more than others")
        print(f"   - A random baseline that ignores this bias will underperform")


if __name__ == "__main__":
    main()

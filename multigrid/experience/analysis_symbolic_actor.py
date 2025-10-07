"""
Evaluate Symbolic Actor Model's Prediction Accuracy

This script evaluates the symbolic actor model's ability to predict target actions
on test scenarios from a CSV file using the distance-based transition probability model.
Also includes a random baseline for comparison.
"""

import argparse
import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from time import time

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.gr_pursuer.astar import get_successor


# Behavior type mapping
BEHAVIOR_TYPE_MAP = {
    0: "like_wall",
    1: "hate_wall", 
    2: "like_edge",
    3: "hate_edge"
}

BETA = 1  # Temperature parameter for symbolic model


def compute_symbolic_transition_probs(
    observer: BeliefUpdateObserver,
    pos_state: Tuple,
    goal: Tuple,
    behavior_idx: int,
    successors: List[Tuple],
    beta: float = BETA
) -> Dict[Tuple, float]:
    """
    Compute transition probabilities using the symbolic model.
    
    Args:
        observer: BeliefUpdateObserver instance with distance matrix
        pos_state: Current position state ((row, col), direction)
        goal: Goal position (row, col)
        behavior_idx: Behavior type index (0-3)
        successors: List of (action, successor_state) tuples
        beta: Temperature parameter
        
    Returns:
        Dictionary mapping successor states to probabilities
    """
    # Compute log probabilities to avoid underflow
    log_tran_probs = {}
    
    for action, succ in successors:
        next_pos, next_dir = succ
        succ_state = ((next_pos[0], next_pos[1]), next_dir)
        
        # Look up distance in the precomputed distance matrix
        if (succ_state, goal) in observer.dist_matrix:
            # Log probability: log(exp(-beta * (1 + dist))) = -beta * (1 + dist)
            log_tran_probs[succ_state] = -beta * (1 + observer.dist_matrix[(succ_state, goal)])
        else:
            # Unreachable state
            log_tran_probs[succ_state] = -np.inf
    
    # Normalize in log space using logsumexp
    log_values = list(log_tran_probs.values())
    if not log_values or all(v == -np.inf for v in log_values):
        # All states unreachable - uniform distribution
        uniform_prob = 1.0 / len(successors) if successors else 0.0
        return {
            ((next_pos[0], next_pos[1]), next_dir): uniform_prob
            for action, (next_pos, next_dir) in successors
        }
    
    # LogSumExp for numerical stability
    max_log_val = max(log_values)
    log_total = max_log_val + np.log(sum(np.exp(v - max_log_val) for v in log_values if v > -np.inf))
    
    # Convert to regular space with normalization
    tran_probs = {}
    for succ_state in log_tran_probs:
        normalized_log_prob = log_tran_probs[succ_state] - log_total
        # Convert back to regular space
        tran_probs[succ_state] = np.exp(normalized_log_prob) if normalized_log_prob > -700 else 0.0
    
    return tran_probs


def eval_symbolic_actor_on_scenario(
    scenario_config: Dict[str, Any], 
    hidden_cost_type: int, 
    verbose: bool = False
) -> Tuple[float, List[bool]]:
    """
    Evaluate the symbolic actor model on a single scenario.
    
    Args:
        scenario_config: Dictionary containing all scenario parameters
        hidden_cost_type: Integer representing behavior type (0-3)
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (average_correctness, correctness_list)
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
    
    if verbose:
        print(f"    🎯 Target goal: {goal}")
        print(f"    📍 Observer: {observer_pos} (dir {observer_dir})")
        print(f"    🎲 Target: {target_pos} (dir {target_dir})")
        print(f"    📏 Trajectory length: {len(target_actions)} steps")
        print(f"    🎭 Behavior type: {BEHAVIOR_TYPE_MAP.get(hidden_cost_type, 'unknown')}")
        
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
    
    # Create observer to get the distance matrix
    # We don't use neural predictor, just need the distance matrix
    observer = BeliefUpdateObserver(
        env=env,
        use_neural_predictor=False,
        use_log_space=True
    )
    
    correctness = []
    
    # Evaluate each step in the trajectory
    for step, target_action in enumerate(target_actions):
        # Get current position state
        pos_state_param = (env.agents[1].pos, int(env.agents[1].dir))
        
        # Get possible successors
        successors = get_successor(env, pos_state_param)
        actual_succ = None
        
        for action, succ in successors:
            if target_action == action:
                next_pos, next_dir = succ
                actual_succ = ((next_pos[0], next_pos[1]), next_dir)
                break
        
        # Verify target action is valid
        if actual_succ is None:
            if verbose:
                print(f"    ⚠️  Warning: Target action {target_action} not in successors at step {step}")
                print(f"        Available actions: {[a for a, _ in successors]}")
            correctness.append(False)
            # Still execute the action to maintain environment state
            observation, reward, terminated, truncated, info = env.step({0: Action.stay, 1: target_action})
            continue
        
        # Use symbolic model to get action probabilities
        try:
            tran_probs = compute_symbolic_transition_probs(
                observer=observer,
                pos_state=pos_state_param,
                goal=goal,
                behavior_idx=hidden_cost_type,
                successors=successors,
                beta=BETA
            )
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Error computing symbolic probabilities at step {step}: {e}")
            correctness.append(False)
            observation, reward, terminated, truncated, info = env.step({0: Action.stay, 1: target_action})
            continue
        
        # Find the action with maximum probability
        max_prob = -float('inf')
        max_succ = None
        for succ_state, prob in tran_probs.items():
            if prob > max_prob:
                max_prob = prob
                max_succ = succ_state
        
        # Map successor back to action
        predict_action = None
        for action, succ in successors:
            next_pos, next_dir = succ
            succ_state = ((next_pos[0], next_pos[1]), next_dir)
            if succ_state == max_succ:
                predict_action = action
                break
        
        # Check if prediction is correct
        target_action_formal = int(target_actions[step])
        is_correct = (int(predict_action) == target_action_formal)
        correctness.append(is_correct)
        
        if verbose:
            if is_correct:
                print(f"    ✅ Step {step}: Predicted {predict_action}, Actual {target_action} (prob: {max_prob:.4f})")
            else:
                print(f"    ❌ Step {step}: Predicted {predict_action}, Actual {target_action} (prob: {max_prob:.4f})")
        
        # Execute the actual action to advance environment
        observation, reward, terminated, truncated, info = env.step({0: Action.stay, 1: target_action})
    
    # Calculate average correctness
    avg_correctness = np.mean(correctness) if correctness else 0.0
    
    return avg_correctness, correctness


def eval_random_baseline_on_scenario(
    scenario_config: Dict[str, Any], 
    hidden_cost_type: int, 
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[float, List[bool]]:
    """
    Evaluate a random baseline on a single scenario.
    Randomly selects one of the available actions at each step.
    
    Args:
        scenario_config: Dictionary containing all scenario parameters
        hidden_cost_type: Integer representing behavior type (0-3)
        seed: Random seed for reproducibility (optional)
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (average_correctness, correctness_list)
    """
    if seed is not None:
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
    
    if verbose:
        print(f"    🎯 Target goal: {goal}")
        print(f"    🎲 Random baseline evaluation")
        
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
    
    correctness = []
    
    # Evaluate each step in the trajectory
    for step, target_action in enumerate(target_actions):
        # Get current position state
        pos_state_param = (env.agents[1].pos, int(env.agents[1].dir))
        
        # Get possible successors
        successors = get_successor(env, pos_state_param)
        
        # Randomly select one of the available actions
        if successors:
            random_action = random.choice([action for action, _ in successors])
        else:
            random_action = Action.stay
        
        # Check if prediction is correct
        target_action_formal = int(target_actions[step])
        is_correct = (int(random_action) == target_action_formal)
        correctness.append(is_correct)
        
        if verbose and is_correct:
            print(f"    🎲 Step {step}: Random guess {random_action} matched actual {target_action}")
        
        # Execute the actual action to advance environment
        observation, reward, terminated, truncated, info = env.step({0: Action.stay, 1: target_action})
    
    # Calculate average correctness
    avg_correctness = np.mean(correctness) if correctness else 0.0
    
    return avg_correctness, correctness


def main(dataset_path: Optional[str] = None, output_path: Optional[str] = None, verbose: bool = False, 
         use_random_baseline: bool = False, random_seed: int = 42, analyze_action_space: bool = False) -> None:
    """
    Run symbolic actor model evaluation on a dataset.
    
    Args:
        dataset_path: Path to CSV dataset file
        output_path: Path to save results CSV (optional)
        verbose: Whether to print detailed progress information
        use_random_baseline: Whether to use only random baseline
        random_seed: Random seed for random baseline
        analyze_action_space: Whether to analyze action space distribution
    """
    if dataset_path is None:
        print("❌ No dataset specified. Use --dataset to provide a CSV file.")
        return
        
    try:
        print(f"📖 Loading dataset: {dataset_path}")
        scenarios_df = pd.read_csv(dataset_path)
        print(f"📊 Found {len(scenarios_df)} scenarios")
        
        # Display dataset summary
        if 'size' in scenarios_df.columns:
            print(f"   Grid sizes: {sorted(scenarios_df['size'].unique())}")
        if 'hidden_cost_style' in scenarios_df.columns:
            print(f"   Behavior styles: {list(scenarios_df['hidden_cost_style'].unique())}")
        if 'hidden_cost_type' in scenarios_df.columns:
            print(f"   Behavior types: {sorted(scenarios_df['hidden_cost_type'].unique())}")
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Track results by behavior type for both models
    symbolic_correctness_by_behavior = {}
    symbolic_all_correctness = []
    random_correctness_by_behavior = {}
    random_all_correctness = []
    results_records = []
    
    # Action space analysis
    action_counts = []  # Track number of available actions per state
    
    model_name = "RANDOM BASELINE" if use_random_baseline else "SYMBOLIC"
    print(f"\n🚀 Starting evaluation with {model_name} actor model...")
    if not use_random_baseline:
        print(f"   Using distance-based transition probabilities (beta={BETA})")
    else:
        print(f"   Random seed: {random_seed}")
    if analyze_action_space:
        print(f"   Analyzing action space distribution...")
    start_time = time()
    
    for i, (idx, scenario_row) in enumerate(scenarios_df.iterrows()):
        scenario_num = i + 1
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 Scenario {scenario_num}/{len(scenarios_df)}")
            if 'scenario_id' in scenario_row:
                print(f"   ID: {scenario_row['scenario_id']}")
            if 'hidden_cost_style' in scenario_row:
                print(f"   Style: {scenario_row['hidden_cost_style']}")
            if 'size' in scenario_row:
                print(f"   Size: {scenario_row['size']}x{scenario_row['size']}")
        else:
            # Progress indicator
            if scenario_num % 10 == 0 or scenario_num == 1:
                print(f"🔍 Processing scenario {scenario_num}/{len(scenarios_df)}...")
            
        # Parse scenario configuration
        try:
            scenario_config = {
                'base_grid': np.array(json.loads(scenario_row['base_grid'])),
                'goals': eval(scenario_row['goals']),
                'goal': eval(scenario_row['goal']),
                'hidden_cost': np.array(json.loads(scenario_row['hidden_cost'])),
                'observer_pos': eval(scenario_row['observer_pos']),
                'target_pos': eval(scenario_row['target_pos']),
                'observer_dir': scenario_row['observer_dir'],
                'target_dir': scenario_row['target_dir'],
                'target_actions': [Action(v) for v in json.loads(scenario_row['all_actions'])]
            }
        except Exception as e:
            print(f"❌ Error parsing scenario {scenario_num}: {e}")
            continue
        
        if 'hidden_cost_type' in scenario_row:
            hidden_cost_type = int(scenario_row['hidden_cost_type'])
        else:
            print(f"❌ Dataset must include 'hidden_cost_type' column.")
            return
        
        # Evaluate the scenario with both models
        try:
            # Symbolic model
            symbolic_avg, symbolic_list = eval_symbolic_actor_on_scenario(
                scenario_config, 
                hidden_cost_type, 
                verbose
            )
            
            # Random baseline
            random_avg, random_list = eval_random_baseline_on_scenario(
                scenario_config,
                hidden_cost_type,
                seed=random_seed + scenario_num,  # Different seed per scenario
                verbose=verbose
            )
            
            # Action space analysis
            if analyze_action_space:
                # Count available actions for each step
                env_temp = AGREnv(
                    base_grid=scenario_config['base_grid'],
                    goals=scenario_config['goals'], 
                    goal=scenario_config['goal'],
                    hidden_cost=scenario_config['hidden_cost'],
                    enable_hidden_cost=True,
                    agents_start_pos=[scenario_config['observer_pos'], scenario_config['target_pos']],
                    agents_start_dir=[scenario_config['observer_dir'], scenario_config['target_dir']],
                    render_mode=None
                )
                env_temp.reset()
                
                for target_action in scenario_config['target_actions']:
                    pos_state = (env_temp.agents[1].pos, int(env_temp.agents[1].dir))
                    successors = get_successor(env_temp, pos_state)
                    action_counts.append(len(successors))
                    env_temp.step({0: Action.stay, 1: target_action})
            
        except Exception as e:
            print(f"❌ Error evaluating scenario {scenario_num}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
        
        # Track results by behavior type
        behavior_name = BEHAVIOR_TYPE_MAP.get(hidden_cost_type, f"type_{hidden_cost_type}")
        
        if behavior_name not in symbolic_correctness_by_behavior:
            symbolic_correctness_by_behavior[behavior_name] = []
            random_correctness_by_behavior[behavior_name] = []
        
        symbolic_correctness_by_behavior[behavior_name].append(symbolic_avg)
        symbolic_all_correctness.append(symbolic_avg)
        
        random_correctness_by_behavior[behavior_name].append(random_avg)
        random_all_correctness.append(random_avg)
        
        # Store results record
        results_records.append({
            'scenario_id': scenario_row.get('scenario_id', idx),
            'size': scenario_row.get('size', None),
            'behavior_type': hidden_cost_type,
            'behavior_name': behavior_name,
            'trajectory_length': len(scenario_config['target_actions']),
            'symbolic_accuracy': symbolic_avg,
            'random_accuracy': random_avg,
            'symbolic_correct': sum(symbolic_list),
            'random_correct': sum(random_list),
            'total_predictions': len(symbolic_list)
        })
        
        # Print intermediate results every 50 scenarios
        if scenario_num % 50 == 0:
            print(f"\n📊 Intermediate results after {scenario_num} scenarios:")
            print(f"   {'Behavior':<15} {'Symbolic':<12} {'Random':<12} {'Improvement':<12}")
            print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
            for behavior in sorted(symbolic_correctness_by_behavior.keys()):
                sym_avg = np.mean(symbolic_correctness_by_behavior[behavior])
                rand_avg = np.mean(random_correctness_by_behavior[behavior])
                improvement = ((sym_avg - rand_avg) / rand_avg * 100) if rand_avg > 0 else 0
                print(f"   {behavior:<15} {sym_avg:.4f}      {rand_avg:.4f}      +{improvement:.1f}%")
            sym_overall = np.mean(symbolic_all_correctness)
            rand_overall = np.mean(random_all_correctness)
            overall_improvement = ((sym_overall - rand_overall) / rand_overall * 100) if rand_overall > 0 else 0
            print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
            print(f"   {'Overall':<15} {sym_overall:.4f}      {rand_overall:.4f}      +{overall_improvement:.1f}%")
    
    # Calculate final statistics
    total_time = time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete!")
    print(f"{'='*80}")
    print(f"\n📊 FINAL RESULTS COMPARISON:")
    print(f"   Total scenarios: {len(symbolic_all_correctness)}")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Average time per scenario: {total_time/len(symbolic_all_correctness):.2f} seconds")
    
    print(f"\n🎯 ACCURACY BY BEHAVIOR TYPE:")
    print(f"   {'Behavior':<15} {'Symbolic':<25} {'Random':<25} {'Improvement':<12}")
    print(f"   {'-'*15} {'-'*25} {'-'*25} {'-'*12}")
    
    for behavior in sorted(symbolic_correctness_by_behavior.keys()):
        sym_accuracies = symbolic_correctness_by_behavior[behavior]
        rand_accuracies = random_correctness_by_behavior[behavior]
        
        sym_avg = np.mean(sym_accuracies)
        sym_std = np.std(sym_accuracies)
        rand_avg = np.mean(rand_accuracies)
        rand_std = np.std(rand_accuracies)
        
        improvement = ((sym_avg - rand_avg) / rand_avg * 100) if rand_avg > 0 else 0
        
        print(f"   {behavior:<15} {sym_avg:.4f} ± {sym_std:.4f} (n={len(sym_accuracies):<3}) "
              f"{rand_avg:.4f} ± {rand_std:.4f} (n={len(rand_accuracies):<3}) "
              f"+{improvement:>5.1f}%")
    
    sym_overall_avg = np.mean(symbolic_all_correctness)
    sym_overall_std = np.std(symbolic_all_correctness)
    rand_overall_avg = np.mean(random_all_correctness)
    rand_overall_std = np.std(random_all_correctness)
    overall_improvement = ((sym_overall_avg - rand_overall_avg) / rand_overall_avg * 100) if rand_overall_avg > 0 else 0
    
    print(f"   {'-'*15} {'-'*25} {'-'*25} {'-'*12}")
    print(f"   {'Overall':<15} {sym_overall_avg:.4f} ± {sym_overall_std:.4f}           "
          f"{rand_overall_avg:.4f} ± {rand_overall_std:.4f}           "
          f"+{overall_improvement:>5.1f}%")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Symbolic Model: {sym_overall_avg:.4f} ({sym_overall_avg*100:.2f}% accuracy)")
    print(f"   Random Baseline: {rand_overall_avg:.4f} ({rand_overall_avg*100:.2f}% accuracy)")
    print(f"   Improvement: +{overall_improvement:.1f}% ({(sym_overall_avg - rand_overall_avg)*100:.2f} percentage points)")
    print(f"\n   Note: The symbolic model uses only distance-to-goal information")
    print(f"         and does NOT consider behavior types (like/hate wall/edge).")
    print(f"         The {overall_improvement:.1f}% improvement over random shows it captures")
    print(f"         goal-directed behavior better than chance.")
    
    # Action space analysis
    if analyze_action_space and action_counts:
        print(f"\n🎲 ACTION SPACE ANALYSIS:")
        action_count_dist = {}
        for count in action_counts:
            action_count_dist[count] = action_count_dist.get(count, 0) + 1
        
        total_states = len(action_counts)
        print(f"   Total states analyzed: {total_states}")
        print(f"   Distribution of available actions per state:")
        for num_actions in sorted(action_count_dist.keys()):
            percentage = (action_count_dist[num_actions] / total_states) * 100
            print(f"      {num_actions} actions: {action_count_dist[num_actions]:5d} states ({percentage:5.2f}%)")
        
        avg_actions = np.mean(action_counts)
        print(f"\n   Average actions per state: {avg_actions:.2f}")
        print(f"   Theoretical random accuracy with {avg_actions:.2f} actions: {1/avg_actions:.4f} ({100/avg_actions:.2f}%)")
        print(f"   Actual random accuracy: {rand_overall_avg:.4f} ({rand_overall_avg*100:.2f}%)")
        
        print(f"\n   Note: Available actions = {{left, right, forward, stay}} minus blocked forward")
        print(f"         - 4 actions: No wall ahead (all actions available)")
        print(f"         - 3 actions: Wall ahead (forward blocked)")
    
    # Save results to CSV if output path is specified
    if output_path:
        results_df = pd.DataFrame(results_records)
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Symbolic Actor Model's Prediction Accuracy with Random Baseline Comparison"
    )
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to the CSV dataset file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save detailed results CSV (optional)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Temperature parameter for symbolic model (default: 1.0)')
    parser.add_argument('--random-only', action='store_true',
                       help='Evaluate only random baseline (for testing)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for random baseline (default: 42)')
    parser.add_argument('--analyze-action-space', action='store_true',
                       help='Analyze the distribution of available actions per state')
    
    args = parser.parse_args()
    
    # Update global BETA if provided
    if args.beta != 1.0:
        BETA = args.beta
        print(f"Using custom beta value: {BETA}")
    
    main(dataset_path=args.dataset, output_path=args.output, verbose=args.verbose,
         use_random_baseline=args.random_only, random_seed=args.random_seed,
         analyze_action_space=args.analyze_action_space)

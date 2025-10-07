"""
Evaluate Symbolic Actor Model's Prediction Accuracy

This script evaluates the symbolic actor model's ability to predict target actions
on test scenarios from a CSV file using the distance-based transition probability model.
"""

import argparse
import numpy as np
import pandas as pd
import json
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
            observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
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
            observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
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
        observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
    
    # Calculate average correctness
    avg_correctness = np.mean(correctness) if correctness else 0.0
    
    return avg_correctness, correctness


def main(dataset_path: Optional[str] = None, output_path: Optional[str] = None, verbose: bool = False) -> None:
    """
    Run symbolic actor model evaluation on a dataset.
    
    Args:
        dataset_path: Path to CSV dataset file
        output_path: Path to save results CSV (optional)
        verbose: Whether to print detailed progress information
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
    
    # Track results by behavior type
    correctness_by_behavior = {}
    all_correctness = []
    results_records = []
    
    print(f"\n🚀 Starting evaluation with SYMBOLIC actor model...")
    print(f"   Using distance-based transition probabilities (beta={BETA})")
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
        
        # Evaluate the scenario
        try:
            avg_correctness, correctness_list = eval_symbolic_actor_on_scenario(
                scenario_config, 
                hidden_cost_type, 
                verbose
            )
        except Exception as e:
            print(f"❌ Error evaluating scenario {scenario_num}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
        
        # Track results by behavior type
        behavior_name = BEHAVIOR_TYPE_MAP.get(hidden_cost_type, f"type_{hidden_cost_type}")
        if behavior_name not in correctness_by_behavior:
            correctness_by_behavior[behavior_name] = []
        
        correctness_by_behavior[behavior_name].append(avg_correctness)
        all_correctness.append(avg_correctness)
        
        # Store results record
        results_records.append({
            'scenario_id': scenario_row.get('scenario_id', idx),
            'size': scenario_row.get('size', None),
            'behavior_type': hidden_cost_type,
            'behavior_name': behavior_name,
            'trajectory_length': len(scenario_config['target_actions']),
            'avg_accuracy': avg_correctness,
            'correct_predictions': sum(correctness_list),
            'total_predictions': len(correctness_list)
        })
        
        # Print intermediate results every 50 scenarios
        if scenario_num % 50 == 0:
            print(f"\n📊 Intermediate results after {scenario_num} scenarios:")
            for behavior, accuracies in sorted(correctness_by_behavior.items()):
                behavior_avg = np.mean(accuracies)
                print(f"   {behavior}: {behavior_avg:.4f} ({len(accuracies)} scenarios)")
            overall_avg = np.mean(all_correctness)
            print(f"   Overall: {overall_avg:.4f}")
    
    # Calculate final statistics
    total_time = time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete!")
    print(f"{'='*80}")
    print(f"\n📊 FINAL RESULTS (SYMBOLIC ACTOR MODEL):")
    print(f"   Total scenarios: {len(all_correctness)}")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Average time per scenario: {total_time/len(all_correctness):.2f} seconds")
    
    print(f"\n🎯 ACCURACY BY BEHAVIOR TYPE:")
    for behavior in sorted(correctness_by_behavior.keys()):
        accuracies = correctness_by_behavior[behavior]
        behavior_avg = np.mean(accuracies)
        behavior_std = np.std(accuracies)
        behavior_min = np.min(accuracies)
        behavior_max = np.max(accuracies)
        print(f"   {behavior:15s}: {behavior_avg:.4f} ± {behavior_std:.4f} "
              f"(min: {behavior_min:.4f}, max: {behavior_max:.4f}, n={len(accuracies)})")
    
    overall_avg = np.mean(all_correctness)
    overall_std = np.std(all_correctness)
    print(f"\n   {'Overall':15s}: {overall_avg:.4f} ± {overall_std:.4f}")
    
    # Save results to CSV if output path is specified
    if output_path:
        results_df = pd.DataFrame(results_records)
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Symbolic Actor Model's Prediction Accuracy on Goal Recognition Scenarios"
    )
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to the CSV dataset file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save detailed results CSV (optional)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Temperature parameter for symbolic model (default: 1.0)')
    
    args = parser.parse_args()
    
    # Update global BETA if provided
    if args.beta != 1.0:
        BETA = args.beta
        print(f"Using custom beta value: {BETA}")
    
    main(dataset_path=args.dataset, output_path=args.output, verbose=args.verbose)

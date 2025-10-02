import argparse
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any
from time import time

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.target import AstarTarget
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.core.actions import Action
from multigrid.gr_pursuer.agents.neuro_predictor import neuro_predict
from multigrid.gr_pursuer.astar import get_successor


def eval_neuro_predictor_using_scenario(scenario_config: Dict[str, Any], hidden_cost_type:int, verbose: bool = False) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Run a single goal recognition scenario.
    
    Args:
        scenario_config: Dictionary containing all scenario parameters
        verbose: Whether to print detailed progress information
        
    Returns:
        Tuple of (success_flag, convergence_step, detailed_results)
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
    
    for step, target_action in enumerate(target_actions):
        # using neuro_predictor to predict the target_action 
        # we need to get (env, goal, behavior_type, successors, pos_state)
        env_param = env
        goal_param = env.goal
        behavior_type_param = hidden_cost_type
        pos_state_param = (env.agents[1].pos, int(env.agents[1].dir))
        successors = get_successor(env, pos_state_param)
        formatted_successors = []
        for action, succ in successors:
            next_pos, next_dir = succ
            formatted_successors.append((action, ((next_pos[0], next_pos[1]), next_dir)))
        successors_param = formatted_successors
        tran_probs = neuro_predict(
            env = env_param,
            goal = goal_param,
            behavior_type = behavior_type_param,
            successors = successors_param,
            pos_state = pos_state_param
        )
        
        max_prob = -100
        max_succ = None
        for cur_succ, prob in tran_probs.items():
            if prob > max_prob:
                max_prob = prob
                max_succ = cur_succ
                        
        predict_action = None 
        for action, succ in successors:
            if succ == max_succ:
                predict_action = action
                break
        
        if int(predict_action) == int(target_action):
            correctness.append(True)
        else:
            correctness.append(False)
        
        
    # correctness is a list of booleans, convert it into average correctness
    avg_correctness = np.mean(correctness)
    return avg_correctness
        
        
def main(dataset_path: Optional[str] = None, verbose: bool = False) -> None:
    """
    Run goal recognition evaluation on a dataset.
    
    Args:
        dataset_path: Path to CSV dataset file
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
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    correctness_dict_by_behavior = dict()
    
    print(f"\n🚀 Starting evaluation...")
    start_time = time()
    
    for i, (idx, scenario_row) in enumerate(scenarios_df.iterrows()):
        scenario_num = i + 1
        if verbose:
            print(f"\n🔍 Scenario {scenario_num}/{len(scenarios_df)}")
            if 'hidden_cost_style' in scenario_row:
                print(f"   Style: {scenario_row['hidden_cost_style']}")
            if 'size' in scenario_row:
                print(f"   Size: {scenario_row['size']}x{scenario_row['size']}")
        else:
            print(f"🔍 Scenario {scenario_num}/{len(scenarios_df)}", end=" ")
            
        # Parse scenario configuration
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
        if 'hidden_cost_style' not in scenario_row:
            hidden_cost_type = scenario_row['hidden_cost_type']
        else:
            hidden_cost_type = scenario_row['hidden_cost_style']
        
        correctness = eval_neuro_predictor_using_scenario(scenario_config, hidden_cost_type, verbose)
        if hidden_cost_type not in correctness_dict_by_behavior:
            correctness_dict_by_behavior[hidden_cost_type] = []
            
        correctness_dict_by_behavior[hidden_cost_type].append(correctness)
        
        # intermediate average correctness
        for behavior, correctness_list in correctness_dict_by_behavior.items():
            interm_avg = np.mean(correctness_list)
            print(f" | Style '{behavior}': {interm_avg:.4f}", end="")
     
        print("============================")
    avg_correctness = np.mean(correctness_list)
    total_time = time() - start_time
    print(f"\n✅ Evaluation complete!")
    print(f"   Average Correctness: {avg_correctness:.4f}")
    print(f"   Total Time: {total_time:.2f} seconds")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neuro Predictor on Goal Recognition Scenarios")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV dataset file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    main(dataset_path=args.dataset, verbose=args.verbose)
"""
Script to check that every potential goal is reachable from the start target location
in the given dataset. This helps identify scenarios where goals might be unreachable
due to walls or other obstacles.
"""

import argparse
import pandas as pd
import numpy as np
import json
from collections import deque
from typing import List, Tuple, Set

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    print(f"📖 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"📊 Found {len(df)} scenarios")
    return df

def bfs_reachable(grid: np.ndarray, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> bool:
    """
    Use BFS to check if goal_pos is reachable from start_pos in the grid.
    
    Args:
        grid: 2D numpy array where 0 = free cell, non-zero = obstacle
        start_pos: Starting position (row, col)
        goal_pos: Goal position (row, col)
    
    Returns:
        True if goal is reachable, False otherwise
    """
    # Check if start or goal is on an obstacle
    if grid[start_pos] != 0:
        return False
    if grid[goal_pos] != 0:
        return False
    
    # BFS
    queue = deque([start_pos])
    visited = set([start_pos])
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current = queue.popleft()
        
        # Check if we reached the goal
        if current == goal_pos:
            return True
        
        # Explore neighbors
        for dr, dc in directions:
            next_pos = (current[0] + dr, current[1] + dc)
            
            # Check bounds
            if (0 <= next_pos[0] < grid.shape[0] and 
                0 <= next_pos[1] < grid.shape[1] and
                next_pos not in visited and
                grid[next_pos] == 0):
                
                visited.add(next_pos)
                queue.append(next_pos)
    
    return False

def check_scenario_reachability(row: pd.Series, scenario_idx: int, verbose: bool = False) -> dict:
    """
    Check if all goals are reachable from the target's starting position.
    
    Args:
        row: DataFrame row containing scenario information
        scenario_idx: Index of the scenario for reporting
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with reachability results
    """
    # Parse grid and positions
    base_grid = np.array(json.loads(row['base_grid']))
    goals = eval(row['goals'])
    target_pos = eval(row['target_pos'])
    
    # Convert positions to tuples if they're numpy integers
    target_pos = tuple(int(x) for x in target_pos)
    goals = [tuple(int(x) for x in goal) for goal in goals]
    
    if verbose:
        print(f"\n🔍 Scenario {scenario_idx}:")
        print(f"   Grid size: {base_grid.shape}")
        print(f"   Target start: {target_pos}")
        print(f"   Goals: {goals}")
    
    results = {
        'scenario_idx': scenario_idx,
        'target_pos': target_pos,
        'all_reachable': True,
        'reachable_goals': [],
        'unreachable_goals': []
    }
    
    # Check each goal
    for goal in goals:
        is_reachable = bfs_reachable(base_grid, target_pos, goal)
        
        if is_reachable:
            results['reachable_goals'].append(goal)
            if verbose:
                print(f"   ✅ Goal {goal}: Reachable")
        else:
            results['unreachable_goals'].append(goal)
            results['all_reachable'] = False
            if verbose:
                print(f"   ❌ Goal {goal}: UNREACHABLE")
    
    return results

def main(dataset_path: str, verbose: bool = False, check_all: bool = False):
    """
    Main function to check goal reachability across the dataset.
    
    Args:
        dataset_path: Path to the CSV dataset
        verbose: Print detailed information for each scenario
        check_all: Continue checking all scenarios even after finding issues
    """
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Track results
    all_results = []
    problem_scenarios = []
    total_scenarios = len(df)
    checked_scenarios = 0
    
    print(f"\n🚀 Starting reachability check...")
    print("="*80)
    
    # Check each scenario
    for idx, row in df.iterrows():
        checked_scenarios += 1
        
        if not verbose and checked_scenarios % 50 == 0:
            print(f"Progress: {checked_scenarios}/{total_scenarios} scenarios checked...")
        
        results = check_scenario_reachability(row, idx, verbose)
        all_results.append(results)
        
        if not results['all_reachable']:
            problem_scenarios.append(results)
            print(f"\n❌ Problem in scenario {idx}:")
            print(f"   Target pos: {results['target_pos']}")
            print(f"   Unreachable goals: {results['unreachable_goals']}")
            print(f"   Reachable goals: {results['reachable_goals']}")
            
            if not check_all:
                print("\n⚠️  Stopping at first problem. Use --check-all to continue checking.")
                break
    
    # Summary
    print("\n" + "="*80)
    print("📊 Summary:")
    print(f"   Total scenarios checked: {checked_scenarios}")
    print(f"   Scenarios with all goals reachable: {checked_scenarios - len(problem_scenarios)}")
    print(f"   Scenarios with unreachable goals: {len(problem_scenarios)}")
    
    if len(problem_scenarios) > 0:
        print(f"\n❌ Found {len(problem_scenarios)} problematic scenarios!")
        print("\nProblematic scenario indices:")
        for result in problem_scenarios:
            print(f"   Scenario {result['scenario_idx']}: {len(result['unreachable_goals'])} unreachable goal(s)")
    else:
        print(f"\n✅ All goals are reachable in all {checked_scenarios} scenarios!")
    
    # Save results if there are problems
    if len(problem_scenarios) > 0:
        output_file = "unreachable_goals_report.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(problem_scenarios, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: {output_file}")
    
    return len(problem_scenarios) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if all goals are reachable from target start positions in dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset and stop at first problem
  python check_goal_reachability.py --dataset test_small.csv
  
  # Check all scenarios even if problems found
  python check_goal_reachability.py --dataset test_small.csv --check-all
  
  # Verbose mode with detailed output
  python check_goal_reachability.py --dataset test_small.csv --verbose
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each scenario"
    )
    
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all scenarios even after finding problems"
    )
    
    args = parser.parse_args()
    
    success = main(args.dataset, args.verbose, args.check_all)
    
    # Exit with appropriate code
    exit(0 if success else 1)

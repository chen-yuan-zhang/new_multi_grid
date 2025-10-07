#!/usr/bin/env python3
"""
Run complete ablation study with all combinations of observer modes.

This script runs experiments with all 9 combinations of:
- Observer action modes: greedy, stay, random
- Belief update modes: bayesian, optimal, uniform

Results are saved to separate CSV files for each combination.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(dataset_path, action_mode, belief_mode, verbose=False):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"Running: action_mode={action_mode}, belief_mode={belief_mode}")
    print(f"{'='*80}")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    main_script = script_dir / "main.py"
    
    cmd = [
        sys.executable,
        str(main_script),
        "--dataset", dataset_path,
        "--action-mode", action_mode,
        "--belief-mode", belief_mode
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed after {elapsed:.1f}s: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        raise

def main():
    """Run all combinations of observer modes."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete ablation study with all observer mode combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs 9 experiments with all combinations of:
- Observer action modes: greedy, stay, random  
- Belief update modes: bayesian, optimal, uniform

Results are saved to:
  results_greedy_bayesian.csv
  results_greedy_optimal.csv
  results_greedy_uniform.csv
  results_stay_bayesian.csv
  results_stay_optimal.csv
  results_stay_uniform.csv
  results_random_bayesian.csv
  results_random_optimal.csv
  results_random_uniform.csv

Examples:
  # Run all combinations
  python run_ablation_study.py --dataset results.csv
  
  # Run with verbose output
  python run_ablation_study.py --dataset results.csv --verbose
  
  # Run only specific modes
  python run_ablation_study.py --dataset results.csv --action-modes greedy stay
  python run_ablation_study.py --dataset results.csv --belief-modes bayesian optimal
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset file"
    )
    
    parser.add_argument(
        "--action-modes",
        nargs='+',
        choices=['greedy', 'stay', 'random'],
        default=['greedy', 'stay', 'random'],
        help="Observer action modes to test (default: all)"
    )
    
    parser.add_argument(
        "--belief-modes",
        nargs='+',
        choices=['bayesian', 'optimal', 'uniform'],
        default=['bayesian', 'optimal', 'uniform'],
        help="Belief update modes to test (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"🚀 Starting Ablation Study")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🎮 Action modes: {args.action_modes}")
    print(f"🧠 Belief modes: {args.belief_modes}")
    
    total_experiments = len(args.action_modes) * len(args.belief_modes)
    print(f"📈 Total experiments: {total_experiments}")
    
    # Track results
    completed = []
    failed = []
    
    overall_start = time.time()
    
    try:
        for i, action_mode in enumerate(args.action_modes):
            for j, belief_mode in enumerate(args.belief_modes):
                exp_num = i * len(args.belief_modes) + j + 1
                print(f"\n\n{'#'*80}")
                print(f"# Experiment {exp_num}/{total_experiments}")
                print(f"{'#'*80}")
                
                success = run_experiment(
                    str(dataset_path),
                    action_mode,
                    belief_mode,
                    args.verbose
                )
                
                config = f"{action_mode}_{belief_mode}"
                if success:
                    completed.append(config)
                else:
                    failed.append(config)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Ablation study interrupted by user")
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n\n{'='*80}")
    print(f"📊 Ablation Study Summary")
    print(f"{'='*80}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"Completed: {len(completed)}/{total_experiments}")
    print(f"Failed: {len(failed)}/{total_experiments}")
    
    if completed:
        print(f"\n✅ Successful experiments:")
        for config in completed:
            print(f"   - {config}")
            print(f"     Output: results_{config}.csv")
    
    if failed:
        print(f"\n❌ Failed experiments:")
        for config in failed:
            print(f"   - {config}")
    
    print(f"\n{'='*80}")
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    else:
        print("✅ All experiments completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

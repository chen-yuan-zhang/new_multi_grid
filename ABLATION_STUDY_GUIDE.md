# Observer Mode Ablation Study Guide

## Overview

This guide explains how to run systematic experiments with different observer configurations to understand which components contribute most to goal recognition performance.

## Observer Modes

### Action Modes (How the observer moves)

1. **`greedy`** (Default)
   - Actively moves toward the most likely actor position
   - Uses greedy action selection to minimize distance to target
   - Maximizes information gathering through active movement

2. **`stay`**
   - Remains stationary at initial position
   - Only observes when actor comes into view
   - Tests whether movement is necessary for tracking

3. **`random`**
   - Selects actions uniformly at random from available successors
   - Random exploration strategy
   - Control condition for comparing against intelligent movement

### Belief Update Modes (How the observer tracks)

1. **`bayesian`** (Default)
   - Full Bayesian inference with transition probabilities
   - Models uncertainty over actor position, goal, and behavior
   - Computes exact posterior distributions
   - Highest computational cost but most accurate

2. **`optimal`**
   - Point estimate tracking (maximum likelihood)
   - Assumes actor is at single most likely position
   - No uncertainty modeling
   - Lower computational cost, deterministic

3. **`uniform`**
   - Maintains uniform distribution over all positions
   - Does not update beliefs based on observations
   - No learning baseline
   - Minimal computational cost

## Running Experiments

### Single Experiment

Run a single configuration:

```bash
cd /mnt/c/active_gr/neurosymbolic_agr/new_multi_grid/multigrid/experience

# Default: greedy + bayesian
python main.py --dataset ../../results_test_new.csv

# Stationary observer with Bayesian tracking
python main.py --dataset ../../results_test_new.csv --action-mode stay --belief-mode bayesian

# Random movement with uniform baseline
python main.py --dataset ../../results_test_new.csv --action-mode random --belief-mode uniform
```

### Complete Ablation Study

Run all 9 combinations automatically:

```bash
cd /mnt/c/active_gr/neurosymbolic_agr/new_multi_grid/multigrid/experience

# Run all combinations
python run_ablation_study.py --dataset ../../results_test_new.csv

# Run with verbose output
python run_ablation_study.py --dataset ../../results_test_new.csv --verbose

# Run only specific modes
python run_ablation_study.py --dataset ../../results_test_new.csv --action-modes greedy stay
python run_ablation_study.py --dataset ../../results_test_new.csv --belief-modes bayesian optimal
```

### Output Files

Each experiment creates a separate results file:

```
results_greedy_bayesian.csv   # Active tracking + Full inference
results_greedy_optimal.csv    # Active tracking + Point estimate
results_greedy_uniform.csv    # Active tracking + No learning
results_stay_bayesian.csv     # Stationary + Full inference
results_stay_optimal.csv      # Stationary + Point estimate  
results_stay_uniform.csv      # Stationary + No learning
results_random_bayesian.csv   # Random movement + Full inference
results_random_optimal.csv    # Random movement + Point estimate
results_random_uniform.csv    # Random movement + No learning
```

## Experimental Matrix

| Action Mode | Belief Mode | Description | Use Case |
|-------------|-------------|-------------|----------|
| greedy | bayesian | **Full system** | Standard active goal recognition |
| greedy | optimal | Active + point est. | Fast active tracking |
| greedy | uniform | Active + baseline | Test movement only |
| stay | bayesian | Stationary + full | Passive monitoring |
| stay | optimal | Stationary + point | Fast passive tracking |
| stay | uniform | **Lower bound** | Worst case baseline |
| random | bayesian | Random + full | Random exploration |
| random | optimal | Random + point | Random + fast |
| random | uniform | Random + baseline | Control condition |

## Research Questions

### Q1: Does active movement help?

Compare action modes with same belief mode:

```bash
# Compare greedy vs stay with Bayesian beliefs
python main.py --dataset data.csv --action-mode greedy --belief-mode bayesian
python main.py --dataset data.csv --action-mode stay --belief-mode bayesian

# Compare greedy vs random with Bayesian beliefs  
python main.py --dataset data.csv --action-mode greedy --belief-mode bayesian
python main.py --dataset data.csv --action-mode random --belief-mode bayesian
```

**Expected**: Greedy should outperform stay and random if active information gathering helps.

### Q2: Does full Bayesian inference help?

Compare belief modes with same action mode:

```bash
# Compare belief modes with greedy movement
python main.py --dataset data.csv --action-mode greedy --belief-mode bayesian
python main.py --dataset data.csv --action-mode greedy --belief-mode optimal
python main.py --dataset data.csv --action-mode greedy --belief-mode uniform
```

**Expected**: Bayesian > optimal > uniform if uncertainty modeling helps.

### Q3: What is the performance hierarchy?

Run all 9 combinations and compare:

```bash
python run_ablation_study.py --dataset data.csv
```

**Expected hierarchy** (best to worst):
1. greedy + bayesian (full system)
2. greedy + optimal (active tracking, fast)
3. stay + bayesian (passive but accurate)
4. greedy + uniform (movement without learning)
5. stay + optimal (passive, point estimate)
6. random + bayesian (random movement)
7. stay + uniform (lower bound)

## Analysis

### Compare Results

After running experiments, analyze with Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all results
configs = [
    ('greedy', 'bayesian'),
    ('greedy', 'optimal'),
    ('greedy', 'uniform'),
    ('stay', 'bayesian'),
    ('stay', 'optimal'),
    ('stay', 'uniform'),
    ('random', 'bayesian'),
    ('random', 'optimal'),
    ('random', 'uniform')
]

results = {}
for action, belief in configs:
    df = pd.read_csv(f'results_{action}_{belief}.csv')
    results[f'{action}_{belief}'] = {
        'success_rate': df['eval_success'].mean(),
        'avg_convergence': df[df['eval_success']]['eval_convergence_step'].mean(),
        'avg_time': df['eval_execution_time'].mean(),
        'visibility_ratio': df['visibility_ratio'].mean()
    }

# Create comparison DataFrame
comparison = pd.DataFrame(results).T
print(comparison)

# Plot success rates
comparison['success_rate'].plot(kind='bar', figsize=(12, 6))
plt.ylabel('Success Rate')
plt.title('Success Rate by Observer Configuration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ablation_success_rates.png')
```

### Key Metrics

For each configuration, examine:

1. **Success Rate**: Percentage of scenarios where correct goal identified
2. **Convergence Step**: How quickly the observer identifies the goal
3. **Execution Time**: Computational cost per scenario
4. **Visibility Ratio**: How much of the trajectory the observer sees
5. **Confidence**: Final belief strength at convergence

### Expected Findings

1. **Movement matters**: greedy > stay > random (for same belief mode)
2. **Inference matters**: bayesian > optimal > uniform (for same action mode)
3. **Interaction effects**: Some combinations may work better than expected
4. **Visibility impact**: Higher visibility should correlate with success
5. **Tradeoffs**: Bayesian is most accurate but slowest

## Advanced Usage

### Custom Subsets

Test on specific scenarios:

```python
# Test only on large grids
python main.py --dataset large_grids_only.csv --action-mode greedy --belief-mode bayesian

# Test on specific behavior types
python main.py --dataset like_wall_scenarios.csv --action-mode stay --belief-mode optimal
```

### Parallel Execution

Run experiments in parallel:

```bash
# Terminal 1
python main.py --dataset data.csv --action-mode greedy --belief-mode bayesian &

# Terminal 2  
python main.py --dataset data.csv --action-mode stay --belief-mode bayesian &

# Terminal 3
python main.py --dataset data.csv --action-mode random --belief-mode bayesian &
```

### Resume Interrupted Runs

The main script saves results incrementally, so you can resume:

```bash
# If interrupted, results are saved in partial output files
# Just re-run the same command to continue or use different dataset
python main.py --dataset data.csv --action-mode greedy --belief-mode bayesian
```

## Troubleshooting

### Out of Memory

If running out of memory with large datasets:

1. Split dataset into smaller chunks
2. Run experiments sequentially instead of parallel
3. Use `optimal` or `uniform` belief modes (less memory)

### Slow Execution

If experiments are too slow:

1. Use smaller dataset (random sample)
2. Try `optimal` instead of `bayesian` belief mode
3. Skip `bayesian` mode for initial exploration

### Validation Errors

If seeing "Direction" or other validation errors:

1. Ensure environment is properly reset before observer creation
2. Check that dataset has valid grid configurations
3. Verify observer.py has the latest implementation

## Citation

If you use this ablation study framework in your research, please cite:

```bibtex
@misc{observer_ablation_2025,
  title={Observer Mode Ablation Study for Active Goal Recognition},
  author={Your Name},
  year={2025},
  note={Neurosymbolic Active Goal Recognition Framework}
}
```

## See Also

- `OBSERVER_MODES_IMPLEMENTATION.md`: Technical implementation details
- `test_observer_modes.py`: Unit tests for observer modes
- `main.py`: Single experiment runner
- `run_ablation_study.py`: Automated ablation study runner

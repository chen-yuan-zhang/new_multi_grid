# Observation Augmentation Guide

This guide explains how to augment observations with belief distributions in the multi-agent goal recognition environment.

## Overview

The belief tracking system allows you to:
1. Track goal beliefs (which goal the target agent is pursuing)
2. Track multi-behavior actor beliefs (where the target is likely to be for each behavior type)
3. Augment observations with this belief information for downstream processing

## Quick Start

```python
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver

# 1. Create environment
env = AGREnv(size=8, num_goals=3, max_steps=50)
obs, info = env.reset()

# 2. Create belief tracker
belief_observer = BeliefUpdateObserver(env)

# 3. Run simulation with belief updates
for step in range(10):
    # Observer computes action using beliefs (MCTS planning)
    observer_action = belief_observer.compute_action(obs[0])
    
    # Target takes some action (unknown to observer)
    target_action = 2  # Example action
    
    # Step environment
    actions = [observer_action, target_action]
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # 4. AUGMENT OBSERVATIONS WITH BELIEFS
    augmented_obs = belief_observer.augment_observation(next_obs)
    
    # 5. Access belief information
    goal_belief = augmented_obs[0]['goal_belief']
    actor_belief = augmented_obs[0]['actor_belief'] 
    behavior_patterns = augmented_obs[0]['behavior_patterns']
    
    obs = next_obs
```

## Augmented Observation Structure

After calling `belief_observer.augment_observation(obs)`, the observer's observation (index 0) will contain additional keys:

### `goal_belief`
- **Type**: `dict[goal_position, probability]`
- **Example**: `{(2, 3): 0.6, (4, 1): 0.4}`
- **Purpose**: Belief distribution over which goal the target is pursuing

### `actor_belief`
- **Type**: `dict[goal_position, list[belief_grid]]`
- **Example**: `{(2, 3): [grid_bt0, grid_bt1, grid_bt2, grid_bt3], (4, 1): [...]}`
- **Purpose**: Multi-behavior belief grids showing where the target is likely to be
- **Details**: 
  - Each goal has a list of 4 behavior grids (one per behavior type)
  - Each grid is 3D: `(width, height, 4_directions)`
  - Grid values sum to the goal belief probability

### `behavior_patterns`
- **Type**: `dict[behavior_name, dict[goal_position, belief_grid]]`
- **Example**: `{"behavior_0": {(2, 3): grid, (4, 1): grid}, "behavior_1": {...}}`
- **Purpose**: Alternative view organized by behavior type rather than goal

## Access Patterns

### Direct Access from BeliefUpdateObserver
```python
# Access beliefs directly (for internal computation)
goal_belief = belief_observer.goal_belief
actor_belief = belief_observer.actor_belief
```

### Access from Augmented Observations (Recommended)
```python
# Access through augmented observations (for integration)
augmented_obs = belief_observer.augment_observation(obs)
goal_belief = augmented_obs[0]['goal_belief']
actor_belief = augmented_obs[0]['actor_belief']
behavior_patterns = augmented_obs[0]['behavior_patterns']
```

### Behavior-Specific Access
```python
# Access specific behavior beliefs for a goal
goal = (2, 3)
behavior_idx = 0  # First behavior type

# Method 1: Through actor_belief
belief_grid = augmented_obs[0]['actor_belief'][goal][behavior_idx]

# Method 2: Through behavior_patterns  
belief_grid = augmented_obs[0]['behavior_patterns'][f'behavior_{behavior_idx}'][goal]

print(f"Grid shape: {belief_grid.shape}")  # (width, height, 4)
print(f"Total belief: {np.sum(belief_grid)}")
```

## Multi-Behavior Structure

The system tracks 4 behavior types (indexed 0-3) for each goal:
- **Behavior 0**: Default/baseline behavior
- **Behavior 1**: Alternative behavior pattern 1  
- **Behavior 2**: Alternative behavior pattern 2
- **Behavior 3**: Alternative behavior pattern 3

Each behavior has its own belief grid representing where the target agent is likely to be if pursuing that goal with that behavior pattern.

## Integration Examples

### Example 1: Simple Decision Making
```python
augmented_obs = belief_observer.augment_observation(obs)
goal_belief = augmented_obs[0]['goal_belief']

# Make decision based on most likely goal
most_likely_goal = max(goal_belief.items(), key=lambda x: x[1])
print(f"Target is most likely pursuing goal {most_likely_goal[0]} with probability {most_likely_goal[1]:.3f}")
```

### Example 2: Behavior Analysis
```python
augmented_obs = belief_observer.augment_observation(obs)
actor_belief = augmented_obs[0]['actor_belief']

for goal, behavior_grids in actor_belief.items():
    print(f"Goal {goal}:")
    for i, grid in enumerate(behavior_grids):
        total_belief = np.sum(grid)
        print(f"  Behavior {i}: {total_belief:.4f}")
```

### Example 3: Spatial Analysis
```python
augmented_obs = belief_observer.augment_observation(obs)
behavior_patterns = augmented_obs[0]['behavior_patterns']

# Analyze where target is most likely to be for behavior 0
behavior_0_beliefs = behavior_patterns['behavior_0']
for goal, grid in behavior_0_beliefs.items():
    max_pos = np.unravel_index(np.argmax(grid), grid.shape)
    max_belief = np.max(grid)
    print(f"Goal {goal}, Behavior 0: Most likely at {max_pos} with belief {max_belief:.4f}")
```

## Key Points

1. **Always call `augment_observation()`**: This is the main interface for accessing beliefs
2. **Multi-behavior structure**: Each goal has 4 behavior types tracked separately
3. **3D belief grids**: Shape is `(width, height, 4_directions)` for spatial + directional beliefs
4. **Probability conservation**: Beliefs sum to the corresponding goal probability
5. **Real-time updates**: Beliefs are updated based on target observations using Bayesian inference

## Demo Scripts

- `multigrid/neurosymbolic_gr.py`: Complete demonstration of belief tracking and augmentation
- `demo_belief_tracking.py`: Focused demo showing belief access patterns
- `test_belief_tracking.py`: Test suite validating all functionality

Run any of these to see the system in action:
```bash
python3 multigrid/neurosymbolic_gr.py
python3 demo_belief_tracking.py  
python3 test_belief_tracking.py
```
# Two-Agent Belief Tracking Integration Guide

This guide shows how to use the augmented `AGREnv` environment with `BeliefUpdateObserver` for two-agent scenarios with belief distributions.

## Key Components

### 1. AGREnv Environment
- Handles two agents: `observer` (agent 0) and `target` (agent 1)
- Provides `augment_obs_with_beliefs()` method to add belief distributions to observations

### 2. BeliefUpdateObserver
- Tracks beliefs about target behavior: `self.goal_belief` and `self.actor_belief`
- Provides `augment_observation()` method to add beliefs to observations
- Has multi-behavior belief structure: `{goal: [grid_bt0, grid_bt1, grid_bt2, grid_bt3]}`

## Usage Pattern

```python
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver

# 1. Create environment and reset
env = AGREnv(size=10, num_goals=3, max_steps=100)
obs, info = env.reset()

# 2. Create belief observer
belief_observer = BeliefUpdateObserver(env)

# 3. Main simulation loop
for step in range(max_steps):
    # Observer computes action using belief tracking
    observer_action = belief_observer.compute_action(obs[0])
    
    # Target action (replace with your target policy)
    target_action = your_target_policy(obs[1])
    
    # Step environment with both actions
    obs, rewards, terminations, truncations, infos = env.step([observer_action, target_action])
    
    # Augment observations with belief distributions
    augmented_obs = belief_observer.augment_observation(obs)
    
    # Now augmented_obs[0] contains:
    # - "goal_belief": {goal: probability}
    # - "actor_belief": {goal: [grid_bt0, grid_bt1, grid_bt2, grid_bt3]}  
    # - "behavior_patterns": {"behavior_0": {goal: 3D_grid}, ...}
    
    # Use augmented observations for next iteration
    obs = augmented_obs
```

## Augmented Observation Structure

After calling `belief_observer.augment_observation(obs)`, the observer's observation (`obs[0]`) contains:

### Original Fields
- `"image"`: Agent's visual observation
- `"observer_pos"`: Observer position
- `"observer_dir"`: Observer direction  
- `"target_pos"`: Target position (if visible)
- `"target_dir"`: Target direction (if visible)

### New Belief Fields
- `"goal_belief"`: `{goal1: 0.3, goal2: 0.5, goal3: 0.2}` - Goal probability distribution
- `"actor_belief"`: `{goal1: [grid_bt0, grid_bt1, grid_bt2, grid_bt3], goal2: [...], goal3: [...]}` - Raw multi-behavior actor belief
- `"behavior_patterns"`: Multi-level dict with behavior-specific beliefs:
  ```python
  {
    "behavior_0": {goal1: 3D_array, goal2: 3D_array, goal3: 3D_array},
    "behavior_1": {goal1: 3D_array, goal2: 3D_array, goal3: 3D_array},
    "behavior_2": {goal1: 3D_array, goal2: 3D_array, goal3: 3D_array},
    "behavior_3": {goal1: 3D_array, goal2: 3D_array, goal3: 3D_array}
  }
  ```

## Direct Belief Access

You can also directly access beliefs from the `BeliefUpdateObserver`:

```python
# Goal beliefs
goal_probs = belief_observer.goal_belief

# Multi-behavior actor beliefs  
actor_beliefs = belief_observer.actor_belief  # {goal: [grid_bt0, grid_bt1, grid_bt2, grid_bt3]}

# Access specific behavior belief
behavior_0_beliefs = {goal: belief_list[0] for goal, belief_list in actor_beliefs.items()}

# From augmented observations - same raw structure
raw_actor_belief = augmented_obs[0]["actor_belief"]  # {goal: [grid_bt0, grid_bt1, grid_bt2, grid_bt3]}
goal1_behavior2_grid = raw_actor_belief[(1, 1)][2]  # Behavior type 2 belief for goal (1,1)
```

## Benefits

1. **Rich Belief Data**: Access to complete belief state including behavior-specific distributions
2. **Two-Agent Support**: Proper integration with multi-agent environment
3. **Flexible Access**: Both direct access and observation augmentation patterns
4. **Behavior Analysis**: Separate tracking of different behavior types
5. **Simple Integration**: Minimal changes to existing code structure

## Example Files

- `example_belief_obs.py`: Complete working examples
- `multigrid/neurosymbolic_gr.py`: Simple interaction demo
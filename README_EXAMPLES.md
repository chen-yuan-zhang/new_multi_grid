# Running the Belief Tracking Examples

This directory contains examples to test and demonstrate the belief tracking functionality.

## Quick Start

### 1. Simple Demo (Recommended)
```bash
python demo_belief_tracking.py
```

This runs a simple demonstration showing:
- Environment setup with 2 agents
- Belief tracking initialization  
- Multi-step simulation with belief updates
- Different ways to access belief data

### 2. Comprehensive Tests
```bash
python test_belief_tracking.py
```

This runs a full test suite including:
- Basic functionality tests
- Observation augmentation tests
- Multi-step simulation tests
- Belief access pattern tests

### 3. Original Example
```bash
python example_belief_obs.py
```

This runs the original detailed example with more verbose output.

## What You Should See

### Successful Output
- ✅ checkmarks indicating successful operations
- Goal belief distributions updating over time
- Raw multi-behavior actor beliefs for each goal
- Different belief access patterns working

### Expected Belief Structure

**Goal Beliefs**: `{goal1: 0.3, goal2: 0.5, goal3: 0.2}`
- Probability distribution over goals

**Actor Beliefs**: `{goal1: [grid_bt0, grid_bt1, grid_bt2, grid_bt3], ...}`
- Raw multi-behavior structure
- 4 behavior types per goal
- Each behavior is a 3D grid (x, y, direction)

**Behavior Patterns**: Same data organized by behavior type first

## Troubleshooting

### Import Errors
Make sure you're running from the project root directory:
```bash
cd /path/to/new_multi_grid
python demo_belief_tracking.py
```

### Missing Dependencies
The examples require:
- numpy
- matplotlib (for visualization in BeliefUpdateObserver)
- multigrid environment

### Slow Execution
The neural predictor (`neuro_predict`) can be slow. The examples use small environments (size=6-8) to minimize runtime.

## Customization

You can modify the examples by changing:
- Environment size: `AGREnv(size=10, ...)`
- Number of goals: `AGREnv(num_goals=4, ...)`
- Number of steps: `for step in range(10):`
- Target behavior: Modify the `target_action` patterns

## Next Steps

After running these examples successfully, you can:
1. Integrate belief tracking into your own agents
2. Use the augmented observations for learning algorithms
3. Analyze behavior patterns from the multi-behavior beliefs
4. Extend the system with additional behavior types
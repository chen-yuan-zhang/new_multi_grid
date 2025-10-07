#!/usr/bin/env python3
"""
Test if step function executes actions correctly.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action

# Create a simple 5x5 grid with no walls
base_grid = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])

hidden_cost = np.ones_like(base_grid, dtype=float)
goals = [(2, 2)]
goal = (2, 2)
observer_pos = (1, 1)
target_pos = (3, 3)
observer_dir = 0  # right
target_dir = 0  # right

print("Simple Test: Open Grid")
print(f"Target starts at: {target_pos}")
print(f"Target direction: {target_dir} (right)")
print(f"Goal: {goal}")
print()

# Setup environment
env = AGREnv(
    base_grid=base_grid,
    goals=goals, 
    goal=goal,
    hidden_cost=hidden_cost,
    enable_hidden_cost=False,
    agents_start_pos=[observer_pos, target_pos],
    agents_start_dir=[observer_dir, target_dir],
    render_mode=None
)

observation, info = env.reset()

print(f"After reset:")
print(f"  Agent 0 (observer): pos={tuple(env.agents[0].pos)}, dir={env.agents[0].dir}")
print(f"  Agent 1 (target): pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

# Test 1: Move target forward (should move from (3,3) to (3,2) since facing right=0)
print("Test 1: Target forward, Observer stay")
print("  Expected: Target should move one cell in the direction it's facing")
obs, rew, term, trunc, info = env.step([Action.stay, Action.forward])
print(f"  Agent 1 (target): pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

# Test 2: Turn target left
print("Test 2: Target left, Observer stay")
obs, rew, term, trunc, info = env.step([Action.stay, Action.left])
print(f"  Agent 1 (target): pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

# Test 3: Move target forward again
print("Test 3: Target forward, Observer stay")
obs, rew, term, trunc, info = env.step([Action.stay, Action.forward])
print(f"  Agent 1 (target): pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

# Test 4: Move observer forward
print("Test 4: Observer forward, Target stay")
obs, rew, term, trunc, info = env.step([Action.forward, Action.stay])
print(f"  Agent 0 (observer): pos={tuple(env.agents[0].pos)}, dir={env.agents[0].dir}")
print(f"  Agent 1 (target): pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

print("="*60)
print("Now test with the original dataset scenario...")
print("="*60)

# Original scenario from dataset
base_grid = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1], 
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1], 
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

hidden_cost = np.array([
    [10.0]*10, [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0],
    [10.0, 10.0, 1.0, 2.0, 1.0, 10.0, 1.0, 2.0, 1.0, 10.0],
    [10.0, 1.0, 2.0, 1.0, 10.0, 1.0, 2.0, 2.0, 1.0, 10.0],
    [10.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 10.0, 10.0],
    [10.0, 10.0, 1.0, 2.0, 1.0, 1.0, 10.0, 1.0, 1.0, 10.0],
    [10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0],
    [10.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 10.0, 10.0, 10.0],
    [10.0, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0],
    [10.0]*10
])

goals = [(6, 8), (2, 7), (3, 2)]
goal = (3, 2)
observer_pos = (8, 3)
target_pos = (8, 6)
observer_dir = 1  # down
target_dir = 2  # left

print(f"\nTarget starts at: {target_pos}, direction: {target_dir} (left)")
print(f"Goal: {goal}")
print(f"Grid value at target position: {base_grid[target_pos]}")
print(f"Grid value left of target: {base_grid[8][5]}")
print(f"Grid value right of target: {base_grid[8][7]}")
print()

env2 = AGREnv(
    base_grid=base_grid,
    goals=goals, 
    goal=goal,
    hidden_cost=hidden_cost,
    enable_hidden_cost=True,
    agents_start_pos=[observer_pos, target_pos],
    agents_start_dir=[observer_dir, target_dir],
    render_mode=None
)

observation, info = env2.reset()
print(f"After reset: Agent 1 at {tuple(env2.agents[1].pos)}, dir={env2.agents[1].dir}")

# First action is forward
print("\nExecuting first action: forward")
print(f"  Target is facing left (dir=2)")
print(f"  Expected: should try to move from (8,6) to (8,5)")
obs, rew, term, trunc, info = env2.step([Action.stay, Action.forward])
print(f"  Result: Agent 1 at {tuple(env2.agents[1].pos)}, dir={env2.agents[1].dir}")
print(f"  Did it move? {tuple(env2.agents[1].pos) != target_pos}")

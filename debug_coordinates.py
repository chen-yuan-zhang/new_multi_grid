#!/usr/bin/env python3
"""
Debug coordinate system and movement.
"""

import sys
import os
import numpy as np
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import DIR_TO_VEC

# Load first scenario
df = pd.read_csv('results_test_new.csv')
row = df.iloc[0]

base_grid = np.array(json.loads(row['base_grid']))
hidden_cost = np.array(json.loads(row['hidden_cost']))
goals = eval(row['goals'])
goal = eval(row['goal'])
observer_pos = eval(row['observer_pos'])
target_pos = eval(row['target_pos'])
observer_dir = row['observer_dir']
target_dir = row['target_dir']

print("Grid shape:", base_grid.shape)
print("Target start:", target_pos, "dir:", target_dir)
print("Direction vector for dir=2 (left):", DIR_TO_VEC[target_dir])
print()

# Show what's around the target
print("Grid around target (8, 6):")
for i in range(max(0, 6), min(10, 9)):
    print(f"  Row {i}: {base_grid[i, 4:9]}")
print()

# Create environment
env = AGREnv(
    base_grid=base_grid,
    goals=goals,
    goal=goal,
    hidden_cost=hidden_cost,
    enable_hidden_cost=True,
    agents_start_pos=[observer_pos, target_pos],
    agents_start_dir=[observer_dir, target_dir],
    render_mode=None
)

obs, info = env.reset()

print("After reset:")
print(f"  env.agents[1].pos = {env.agents[1].pos} (type: {type(env.agents[1].pos)})")
print(f"  env.agents[1].dir = {env.agents[1].dir}")
print(f"  env.agents[1].state.pos = {env.agents[1].state.pos}")
print(f"  env.agents[1].state.dir = {env.agents[1].state.dir}")
print()

# Check grid at agent position
agent_pos = env.agents[1].pos
print(f"Grid value at agent position {tuple(agent_pos)}: {env.grid.get(*agent_pos)}")
print()

# Try to move forward manually
print("Manual forward calculation:")
fwd_pos = agent_pos + DIR_TO_VEC[target_dir]
print(f"  Current: {agent_pos}, dir={target_dir}")
print(f"  Forward vector: {DIR_TO_VEC[target_dir]}")
print(f"  Should move to: {fwd_pos}")
print(f"  Grid at target: {env.grid.get(*fwd_pos) if 0 <= fwd_pos[0] < 10 and 0 <= fwd_pos[1] < 10 else 'OUT OF BOUNDS'}")
print()

# Actually step
print("Executing forward action:")
print(f"  Before: pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
obs, rew, term, trunc, info = env.step([Action.stay, Action.forward])
print(f"  After:  pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

# Check if agent can move
print("Checking if position is blocked:")
print(f"  Current grid position: {tuple(env.agents[1].pos)}")
print(f"  Grid value: {env.grid.get(*env.agents[1].pos)}")
print(f"  Can agent move? front_cell = env.grid.get(*fwd_pos)")

# Check grid encoding
print("\nGrid encoding:")
print("  0 = free cell")
print("  1 = wall")
print()
print("Base grid around target:")
print(base_grid[6:9, 4:9])

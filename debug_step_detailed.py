#!/usr/bin/env python3
"""
Add detailed debugging to step execution.
"""

import sys
import os
import numpy as np
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action

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

print("Target:", target_pos, "dir:", target_dir)
print("Goal:", goal)
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
print(f"  Target pos: {env.agents[1].pos}")
print(f"  Target dir: {env.agents[1].dir}")
print(f"  Target front_pos: {env.agents[1].front_pos}")
print()

# Check what action we're giving
action_to_execute = [Action.stay, Action.forward]
print(f"Executing: {action_to_execute}")
print(f"  Observer (agent 0) action: {action_to_execute[0].name}")
print(f"  Target (agent 1) action: {action_to_execute[1].name}")
print()

# Check grid state
fwd_pos = env.agents[1].front_pos
print(f"Forward position calculation:")
print(f"  front_pos = {fwd_pos}")
print(f"  is_valid_pos = {env.grid.is_valid_pos(*fwd_pos)}")
print(f"  grid.get(*fwd_pos) = {env.grid.get(*fwd_pos)}")
print()

# Step
print("Calling env.step()...")
obs, rew, term, trunc, info = env.step(action_to_execute)

print(f"\nAfter step:")
print(f"  Target pos: {env.agents[1].pos}")
print(f"  Target dir: {env.agents[1].dir}")
print(f"  Did it move? {env.agents[1].pos != target_pos}")
print()

# Try with different action format
print("="*60)
print("Testing with dict format:")
env.reset()
print(f"Before: {env.agents[1].pos}")

# Use dict format
action_dict = {0: Action.stay, 1: Action.forward}
obs, rew, term, trunc, info = env.step(action_dict)
print(f"After: {env.agents[1].pos}")
print(f"Moved? {env.agents[1].pos != target_pos}")

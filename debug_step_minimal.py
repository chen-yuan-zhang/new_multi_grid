#!/usr/bin/env python3
"""
Minimal reproduction of the problem with heavy debugging.
"""

import sys
import os
import numpy as np
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action

# Load first scenario from dataset
df = pd.read_csv("results_test_new.csv")
row = df.iloc[0]

print("Loading scenario 0...")
print(f"Observer: {row['observer_pos']}, dir={row['observer_dir']}")
print(f"Target: {row['target_pos']}, dir={row['target_dir']}")
print(f"Goal: {row['goal']}")
print(f"Actions: {row['all_actions']}")
print()

base_grid = np.array(json.loads(row['base_grid']))
hidden_cost = np.array(json.loads(row['hidden_cost']))
goals = eval(row['goals'])
goal = eval(row['goal'])
observer_pos = eval(row['observer_pos'])
target_pos = eval(row['target_pos'])
observer_dir = row['observer_dir']
target_dir = row['target_dir']
all_actions = [Action(v) for v in json.loads(row['all_actions'])]

print(f"Parsed:")
print(f"  Observer: {observer_pos}, dir={observer_dir}")
print(f"  Target: {target_pos}, dir={target_dir}")
print(f"  Goal: {goal}")
print(f"  Num actions: {len(all_actions)}")
print(f"  Actions: {[a.name for a in all_actions[:5]]}...")
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

print(f"After reset:")
print(f"  Agent 0: {tuple(env.agents[0].pos)}, dir={env.agents[0].dir}")  
print(f"  Agent 1: {tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print(f"  env.target: {env.target}")
print(f"  env.target == env.agents[1]: {env.target == env.agents[1]}")
print()

# Execute first 3 actions with detailed output
for i in range(min(3, len(all_actions))):
    action = all_actions[i]
    print(f"Step {i}: Executing action={action.name} ({action.value})")
    print(f"  Before: Agent1 pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
    print(f"  Calling env.step([Action.stay, {action.name}])")
    
    obs, rew, term, trunc, info = env.step([Action.stay, action])
    
    print(f"  After:  Agent1 pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
    print(f"  Reward: {rew}")
    print(f"  Term: {term}")
    print(f"  Trunc: {trunc}")
    print()

print(f"Final check:")
print(f"  Agent 1 position: {tuple(env.agents[1].pos)}")
print(f"  Goal position: {goal}")
print(f"  Distance: {abs(env.agents[1].pos[0] - goal[0]) + abs(env.agents[1].pos[1] - goal[1])}")

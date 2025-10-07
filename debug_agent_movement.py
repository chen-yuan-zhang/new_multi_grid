#!/usr/bin/env python3
"""
Debug which agent is which and verify action execution.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action

# First scenario from the dataset
base_grid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

hidden_cost = np.array([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0], [10.0, 10.0, 1.0, 2.0, 1.0, 10.0, 1.0, 2.0, 1.0, 10.0], [10.0, 1.0, 2.0, 1.0, 10.0, 1.0, 2.0, 2.0, 1.0, 10.0], [10.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 10.0, 10.0], [10.0, 10.0, 1.0, 2.0, 1.0, 1.0, 10.0, 1.0, 1.0, 10.0], [10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0], [10.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 10.0, 10.0, 10.0], [10.0, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])

goals = [(6, 8), (2, 7), (3, 2)]
goal = (3, 2)
observer_pos = (8, 3)
target_pos = (8, 6)
observer_dir = 1
target_dir = 2
target_actions = [Action(v) for v in [2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2]]

print("Initial Setup:")
print(f"  Observer: pos={observer_pos}, dir={observer_dir}")
print(f"  Target: pos={target_pos}, dir={target_dir}")
print(f"  Goal: {goal}")
print(f"  Actions: {[a.name for a in target_actions]}")
print()

# Setup environment
agents_start_pos = [observer_pos, target_pos]
agents_start_dir = [observer_dir, target_dir]

env = AGREnv(
    base_grid=base_grid,
    goals=goals, 
    goal=goal,
    hidden_cost=hidden_cost,
    enable_hidden_cost=True,
    agents_start_pos=agents_start_pos,
    agents_start_dir=agents_start_dir,
    render_mode=None
)

observation, info = env.reset()

print(f"After reset:")
print(f"  Agent 0: pos={tuple(env.agents[0].pos)}, dir={env.agents[0].dir}")
print(f"  Agent 1: pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
print()

print("Executing trajectory:")
for step, target_action in enumerate(target_actions):
    print(f"\nStep {step}: Action={target_action.name} ({target_action})")
    print(f"  Before: Agent1 pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
    
    # Execute action
    observation, reward, terminated, truncated, info = env.step([Action.stay, target_action])
    
    print(f"  After:  Agent1 pos={tuple(env.agents[1].pos)}, dir={env.agents[1].dir}")
    print(f"  Moved: {observation, reward, terminated, truncated}")

print(f"\n\nFinal position: {tuple(env.agents[1].pos)}")
print(f"Goal position: {goal}")
print(f"Reached goal: {tuple(env.agents[1].pos) == goal}")

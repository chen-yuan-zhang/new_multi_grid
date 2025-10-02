import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver, BEHAVIOR_TYPES
import numpy as np

# Create environment
env = AGREnv(size=8, num_goals=3, max_steps=50)
obs, info = env.reset()

print("Goals:", env.goals)
print("Number of goals:", len(env.goals))
print("BEHAVIOR_TYPES:", BEHAVIOR_TYPES)
print("len(BEHAVIOR_TYPES):", len(BEHAVIOR_TYPES))

# Check what gets passed to set_uniform_prob
initial_goal_belief = {g: 1/len(env.goals) for g in env.goals}
print("Initial goal belief:", initial_goal_belief)

for g in env.goals:
    total_prob_per_behavior = initial_goal_belief[g] / len(BEHAVIOR_TYPES)
    print(f"Goal {g}: total_prob_per_behavior = {total_prob_per_behavior}")

# Check number of free cells
free_cells = np.argwhere(env.base_grid == 0)
num_free_cells = len(free_cells)
print(f"Number of free cells: {num_free_cells}")

# Calculate what uniform probability should be
for g in env.goals:
    total_prob_per_behavior = initial_goal_belief[g] / len(BEHAVIOR_TYPES)
    uniform_prob = total_prob_per_behavior / (num_free_cells * 4) if num_free_cells > 0 else 0
    print(f"Goal {g}: uniform_prob per cell per direction = {uniform_prob}")

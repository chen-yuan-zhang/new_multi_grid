import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
import numpy as np

# Create environment
env = AGREnv(size=6, num_goals=2, max_steps=50)
obs, info = env.reset()

print("Creating BeliefUpdateObserver...")
belief_observer = BeliefUpdateObserver(env)

print("Goal belief:", belief_observer.goal_belief)
print("Actor belief keys:", list(belief_observer.actor_belief.keys()))

# Check if actor beliefs are actually non-zero
for goal, belief_list in belief_observer.actor_belief.items():
    print(f"\nGoal {goal}:")
    for i, grid in enumerate(belief_list):
        total = np.sum(grid)
        nonzero_count = np.count_nonzero(grid)
        print(f"  Behavior {i}: sum={total:.8f}, nonzero_cells={nonzero_count}")
        if nonzero_count > 0:
            print(f"    Min nonzero: {np.min(grid[grid > 0]):.8f}")
            print(f"    Max value: {np.max(grid):.8f}")

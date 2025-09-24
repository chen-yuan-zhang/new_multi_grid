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

print("Initial beliefs:")
for goal, belief_list in belief_observer.actor_belief.items():
    total_belief = sum(np.sum(grid) for grid in belief_list)
    print(f"Goal {goal}: total belief = {total_belief:.6f}")

print("\nAfter first compute_action:")
action = belief_observer.compute_action(obs[0])
print(f"Action: {action}")

print("Updated beliefs:")
for goal, belief_list in belief_observer.actor_belief.items():
    total_belief = sum(np.sum(grid) for grid in belief_list)
    print(f"Goal {goal}: total belief = {total_belief:.6f}")
    for i, grid in enumerate(belief_list):
        behavior_sum = np.sum(grid)
        print(f"  Behavior {i}: {behavior_sum:.6f}")

import random
import argparse
import pandas as pd

from multigrid.envs.goal_prediction import AGREnv
from .agents.target import AstarTarget
import numpy as np


sizes = [10,20,30]
initial_distances = [3, 5, 7]
num_layouts = 10
num_scenarios = 5

for size in sizes:
    for layout_id in range(num_layouts):
        # generate the base grid (e.g. layouts)
        env = AGREnv(size=size)
        obs, info = env.reset()
        base_grid = info['base_grid']

    
        # wall positions
        rows, cols = np.where(base_grid == 1)

        hidden_cost_matrix_1 = np.zeros((size, size))
        hidden_cost_matrix_2 = np.zeros((size, size))
        hidden_cost_matrix_3 = np.zeros((size, size))
        hidden_cost_matrix_4 = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                # cost for each cell equal to the distance to the closet wall
                if base_grid[i, j] == 0:
                    min_dist = np.min(np.abs(rows - i) + np.abs(cols - j))
                    # like wall
                    hidden_cost_matrix_1[i, j] = min_dist
                    # hate wall
                    hidden_cost_matrix_2[i, j] = 1 / (min_dist + 1)
                    # like edge
                    hidden_cost_matrix_3[i, j] = min(i, j, size - i - 1, size - j - 1)
                    # hate edge
                    hidden_cost_matrix_4[i, j] = 1 / (min(i, j, size - i - 1, size - j - 1) + 1)

        hidden_costs = [hidden_cost_matrix_1, hidden_cost_matrix_2, hidden_cost_matrix_3, hidden_cost_matrix_4]

        for initial_distance in initial_distances:
            
            for scenario_id in range(num_scenarios):
                # select start positions and goal positions
                env = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
                agent = AstarTarget(env)

                obs, info = env.reset()
                done = False
                while not done:
                    action = agent.compute_action(obs)
                    obs, reward, done, info = env.step(action)

                # Save the hidden costs and the base grid
                hidden_costs_df = pd.DataFrame({
                    'hidden_cost_1': hidden_cost_matrix_1.flatten(),
                    'hidden_cost_2': hidden_cost_matrix_2.flatten(),
                    'hidden_cost_3': hidden_cost_matrix_3.flatten(),
                    'hidden_cost_4': hidden_cost_matrix_4.flatten(),
                    'base_grid': base_grid.flatten()
                })

                filename = f"hidden_costs_size_{size}_distance_{initial_distance}_scenario_{scenario}.csv"
                hidden_costs_df.to_csv(filename, index=False)

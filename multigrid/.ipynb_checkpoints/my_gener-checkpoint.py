from .gr_pursuer.agents.target import AstarTarget,eGreedyTarget
from .gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.envs.goal_prediction import AGREnv

from matplotlib import pyplot as plt
import random
import argparse
import pandas as pd
import pickle
import gzip
import numpy as np



sizes = [10,20,30]
# initial_distances = [3, 5, 7]
initial_distance = 3
num_layouts = 10
num_scenarios = 5
results = []

        
for size in sizes:
    for layout_id in range(num_layouts):
        env = AGREnv(size=size)
        obs, info = env.reset()
        base_grid = info['base_grid']
        env.close()
        
        # wall positions
        rows, cols = np.where(base_grid == 1)
        # hidden_cost_matrix_1 = np.zeros((size, size))
        # hidden_cost_matrix_2 = np.zeros((size, size))
        # hidden_cost_matrix_3 = np.zeros((size, size))
        # hidden_cost_matrix_4 = np.zeros((size, size))

        hidden_cost_matrix_1 = 10 * np.ones((size, size))
        hidden_cost_matrix_2 = 10 * np.ones((size, size))
        hidden_cost_matrix_3 = 10 * np.ones((size, size))
        hidden_cost_matrix_4 = 10 * np.ones((size, size))

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

        # hidden_cost_matrix_1 = np.random.random((size, size))
        # hidden_cost_matrix_2 = np.random.random((size, size))
        # hidden_cost_matrix_3 = np.random.random((size, size))
        # hidden_cost_matrix_4 = np.random.random((size, size))

        hidden_costs = [hidden_cost_matrix_1, hidden_cost_matrix_2, hidden_cost_matrix_3, hidden_cost_matrix_4]

        
        for scenario_id in range(num_scenarios):
            env_grid = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
            obs, info = env_grid.reset()
            assert (info['base_grid'] == base_grid).all()
            goals = info['goals']
            goal = info['goal']
            agents_start_pos = info['agents_start_pos']
            agents_start_dir = info['agents_start_dir']
            env_grid.close()

            for style_id, hidden_cost in enumerate(hidden_costs):
                print(f"Scenario {scenario_id}, Style {style_id}, Size {size}, Layout {layout_id}, Initial Distance {initial_distance}")
                env_agents = AGREnv(size=size,initial_distance=initial_distance, base_grid=base_grid,
                                    goals=goals,goal=goal,
                                    enable_hidden_cost = True, hidden_cost=hidden_cost,
                                    agents_start_pos=agents_start_pos, agents_start_dir=agents_start_dir)
                obs, info = env_agents.reset()
                TargetAgent = AstarTarget(env_agents)
                while not env_agents.unwrapped.is_done():
                    actions = {agent.index: agent.action_space.sample() for agent in env_agents.unwrapped.agents}
                    actions[1] = TargetAgent.compute_action(obs)
                    observation, reward, terminated, truncated, info = env_agents.step(actions)            
                env_agents.close()

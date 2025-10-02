import random
import argparse
import pandas as pd
import pickle
import gzip

from multigrid.envs.goal_prediction import AGREnv
from .gr_pursuer.agents.target import AstarTarget,eGreedyTarget
import numpy as np


sizes = [10,20,30]
# initial_distances = [3, 5, 7]
initial_distance = 3
num_layouts = 10
num_scenarios = 5

results = []

for size in sizes:
    for layout_id in range(num_layouts):
        # generate the base grid (e.g. layouts)
        env = AGREnv(size=size)
        obs, info = env.reset()
        base_grid = info['base_grid']
        env.close()

    
        # wall positions
        rows, cols = np.where(base_grid == 1)

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

        hidden_costs = [hidden_cost_matrix_1, hidden_cost_matrix_2, hidden_cost_matrix_3, hidden_cost_matrix_4]

        
        for scenario_id in range(num_scenarios):
            # select start positions and goal positions
            env_grid = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid)
            obs, info = env_grid.reset()
            assert (info['base_grid'] == base_grid).all()

            goals = info['goals']
            goal = info['goal']

            agents_start_pos = info['agents_start_pos']
            agents_start_dir = info['agents_start_dir']
            env_grid.close()

            for style_id, hidden_cost in enumerate(hidden_costs):
                # create the target agent
                env_agents = AGREnv(size=size, initial_distance=initial_distance, base_grid=base_grid, goals = goals, 
                                    goal=goal, enable_hidden_cost = True, hidden_cost=hidden_cost, agents_start_dir = agents_start_dir, agents_start_pos = agents_start_pos)
                obs, info = env_agents.reset()
                assert info['agents_start_pos'] == agents_start_pos
                assert info['agents_start_dir'] == agents_start_dir
                assert info['goals'] == goals
                assert info['goal'] == goal

                all_actions = []
                all_imgs = []
                all_obs = []
                TargetAgent = AstarTarget(env_agents)
                print(f"Scenario {scenario_id}, Style {style_id}, Size {size}, Layout {layout_id}, Initial Distance {initial_distance}")
                while not env_agents.unwrapped.is_done():
                    all_obs.append(obs[1])
                    all_imgs.append(env_agents.grid.render(tile_size=32, agents=env_agents.unwrapped.agents[1:], highlight_mask=None))
                    actions = {agent.index: agent.action_space.sample() for agent in env_agents.unwrapped.agents}
                    actions[1] = TargetAgent.compute_action(obs)
                    all_actions.append(actions[1])
                    obs, reward, terminated, truncated, info = env_agents.step(actions)
                env_agents.close()

                # save the results
                results.append({
                    'size': size,
                    'layout_id': layout_id,
                    'initial_distance': initial_distance,
                    'scenario_id': scenario_id,
                    'hidden_cost_type': style_id,
                    'start_positions': agents_start_pos[1],
                    'start_directions': agents_start_dir[1],
                    'goals': goals,
                    'goal': goal,
                    'all_actions': all_actions,
                    'all_imgs': all_imgs,
                    'all_obs': all_obs
                })
                print(len(results), "results collected")

with gzip.open('results.pkl.gz', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

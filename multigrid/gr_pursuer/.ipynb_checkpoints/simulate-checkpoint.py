from .agents.target import Target
from .agents.observer import Observer

import os
import argparse
import numpy as np
import pandas as pd

from csv import writer
from time import sleep
from pathlib import Path
from .astar import astar2d
from multigrid.envs.goal_prediction import GREnv


def get_data(env, step, actions):
    base_grid = env.base_grid
    goals = env.goals
    target_goal = env.goal
    observer_action = int(actions.get(0, None))
    target_action = actions.get(1, None)

    cost = []
    for goal in goals:
            cost.append(len(astar2d(env.observer.pos, goal, base_grid)) - 1)

    data = {
        "step": step,
        "observer_pos": env.observer.pos, 
        "target_pos": env.target.pos, 
        "observer_dir": int(env.observer.dir), 
        "target_dir": int(env.target.dir),
        "observer_action": observer_action,
        "target_action": target_action, 
        "target_goal": target_goal,
        "cost": cost,
    }
    
    return data

class Writer:
   
    def __init__(self, file):
      self.file = file

    def write(self, data):
      
        # Create the file and add the columns if it does not exist
        if not os.path.exists(self.file):
            with open(self.file, 'w') as file:
                    epoch_log = writer(file)
                    epoch_log.writerow(
                        data.keys()
                    )

        with open(self.file, 'a') as file:
            epoch_log = writer(file) 
            epoch_log.writerow(
                data.values()
            )


def run(scenario=None, main_dir=None):  

    if scenario is not None:
        base_grid = np.array(eval(scenario["base_grid"]))
        goals = eval(scenario["goals"])
        hidden_cost = np.array(eval(scenario["hidden_cost"]))
        layout = scenario["layout"]
        nscenario = scenario["scenario"]
        agents_start_pos=scenario[["observer_pos", "target_pos"]].apply(eval).tolist()
        agents_start_dir = scenario[['observer_dir', 'target_dir']].tolist()
        
        env = GREnv(size=32, base_grid=base_grid, 
                    see_through_walls=[False, True],
                    agent_view_size=[5, 3],
                    goals=goals, hidden_cost=hidden_cost, 
                    agents_start_pos=agents_start_pos,
                    agents_start_dir=agents_start_dir,
                    # enable_hidden_cost=enable_hidden_cost, 
                    # render_mode='human'
        )
        writer_dir = main_dir / f"layout{layout}scenario{nscenario}.csv"

    else:
        env = GREnv(size=32,
                    see_through_walls=[False, True],
                    agent_view_size=[5, 3],
                    # render_mode='human
        )
        writer_dir = main_dir / f"scenario_random.csv"
        
    csv_writer = Writer(writer_dir)

    observations, infos = env.reset()
    observer = Observer(env) # Green
    target = Target(env) # Red
    agents = [observer, target]

    step = 0
    target_failed = False
    while not env.is_done():  
        actions = {}
        for i, agent in enumerate(agents):
            action = agent.compute_action(observations[i])
            if action is not None:
                actions[agent.agent.index] = int(action)
            else:
                if i==1:
                    target_failed = True

        data = get_data(env, step, actions)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        data["done"] = int(env.is_done())
        csv_writer.write(data)

        step+=1

    return step, target_failed


def main(input_dir=None):

    if input_dir is not None:

        main_dir = Path(input_dir)
        scenarios_file = main_dir / "scenarios.csv"
        scenarios = pd.read_csv(scenarios_file)

        # Get the name of the file
        for idx, scenario in scenarios.iterrows():
            nlayout = scenarios.loc[idx, 'layout']
            nscenario = scenarios.loc[idx, 'scenario']
            print(f"Scenario {idx} Layout {nlayout}, Scenario {nscenario}")
            nsteps, target_failed = run(scenario=scenario, main_dir=main_dir)
            scenarios.loc[idx, "nsteps"] = nsteps
            scenarios.loc[idx, "target_failed"] = target_failed

        scenarios.to_csv(scenarios_file, index=False)
        
    else:

        while True:
            run()
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for running the scenario")
    parser.add_argument("--input_dir", type=str, default=None, help="directory location that contains the file to run as scenarios")

    args = parser.parse_args()
    main(args.input_dir)
from multigrid.gr_pursuer.agents.target import AstarTarget
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver,Observer

import argparse
import numpy as np
import pandas as pd
from time import sleep
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
import json


def run(base_grid=None, goals=None, goal = None, hidden_cost=None, observer_pos=None, target_pos=None, TargetAgent_actions = None, observer_dir=None, target_dir=None, render_mode="human"):
    agents_start_pos = [observer_pos, target_pos]
    agents_start_dir = [observer_dir, target_dir] 
    env = AGREnv(base_grid=base_grid, 
                     goals=goals, hidden_cost=hidden_cost, goal = goal,
                     enable_hidden_cost=True, 
                     agents_start_pos=agents_start_pos,agents_start_dir = agents_start_dir,
                     render_mode=None)
    observation, infos = env.reset()
    #ObserverAgent = BeliefUpdateObserver(env)
    ObserverAgent = BeliefUpdateObserver(env)
    TargetAgent = AstarTarget(env)

    flag = False
    first_step = -1
    steps = 0
    while not env.unwrapped.is_done():
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        actions[0] = ObserverAgent.compute_action(observation[0])
        actions[1] = TargetAgent_actions[steps]
        #img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
        observation, reward, terminated, truncated, info = env.step(actions)
        steps += 1
        probs = ObserverAgent.goal_belief
        predicted_goal = None
        max_prob = 0
        for goal, prob in probs.items():
            if prob > max_prob:
                max_prob = prob
                predicted_goal = goal

        if predicted_goal == env.goal and max_prob > 0.5:
            if not flag:
                first_step = ObserverAgent.step
                flag = True
                print("Success")
                print(ObserverAgent.step)
        else:
             flag = False
         
    return flag, first_step


def main(dataset=None):
    if dataset is not None:
        scenarios = pd.read_csv(dataset)
        succ_count = 0
        step_count = 0
        for idx, scenario in scenarios.iterrows():
            print(f"Scenario {idx}")
            base_grid = np.array(eval(scenario["base_grid"]))
            goals = eval(scenario["goals"])
            hidden_cost = np.array(eval(scenario["hidden_cost"]))
            observer_pos = eval(scenario["observer_pos"])
            target_pos = eval(scenario["target_pos"])
            observer_dir = scenario["observer_dir"]
            target_dir = scenario["target_dir"]
            actions = [Action(v) for v in json.loads(scenario["all_actions"]) ] 
            goal = eval(scenario["goal"])
            
          
            flag, first_step = run(base_grid=base_grid, goals=goals, goal = goal,
                                   hidden_cost=hidden_cost, observer_pos=observer_pos, target_pos=target_pos, 
                                   observer_dir=observer_dir, target_dir=target_dir, 
                                   TargetAgent_actions = actions)
            
            print(f"Success: {flag} First step: {first_step}")
            scenarios.loc[idx, "success"] = flag
            scenarios.loc[idx, "first_step"] = first_step
            succ_count += 1 if flag else 0
            step_count += first_step if flag else 0
        print(f"Success rate: {succ_count}/{len(scenarios)}")
        print(f"Average first step: {step_count/succ_count}")
        scenarios.to_csv("results_2.csv", index=False)
    else:
        while True:
            run()
            break
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for running the scenario")
    parser.add_argument("--dataset", type=str, default=None, help="csv file to run as dataset")

    args = parser.parse_args()
    main(args.dataset)
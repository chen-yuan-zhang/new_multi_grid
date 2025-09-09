from .agents.target import Target
from .agents.observer import Observer, BeliefUpdateObserver

import argparse
import numpy as np
import pandas as pd
from time import sleep
from multigrid.envs.goal_prediction import GREnv
from multigrid.core.actions import Action


def run(base_grid=None, goals=None, hidden_cost=None, observer_pos=None, target_pos=None, observer_dir=None, target_dir=None, render_mode="human"):
   env = GREnv(size=32, base_grid=base_grid, 
               agent_view_size=[5, 3],
                     goals=goals, hidden_cost=hidden_cost, 
                     enable_hidden_cost=True, 
                     render_mode=None)
   observations, infos = env.reset()
   if not base_grid is None:
      print("Setting up the environment")
      env.base_grid = base_grid
      env.goals = goals
      env.hidden_cost = hidden_cost
      env.observer.pos = observer_pos
      env.target.pos = target_pos
      env.observer.dir = observer_dir
      env.target.dir = target_dir
   # Green
   observer = BeliefUpdateObserver(env)
   # Red
   target = Target(env)

   agents = [observer, target]
   observations, rewards, terminations, truncations, infos = env.step(actions={agent.agent.index:Action.stay for agent in agents})

   flag = False
   first_step = -1

   while not env.is_done():  
      actions = {}
      for i, agent in enumerate(agents):
         action = agent.compute_action(observations[i])
         if action is not None:
            actions[agent.agent.index] = action
      observations, rewards, terminations, truncations, infos = env.step(actions)

      probs = observer.goal_belief
      predicted_goal = None
      max_prob = 0
      for goal, prob in probs.items():
         if prob > max_prob:
            max_prob = prob
            predicted_goal = goal

      if predicted_goal == env.goal and max_prob > 0.5:
         if not flag:
            first_step = observer.step
            flag = True
            print("Success")
            print(observer.step)
      else:
         flag = False
         
   return flag, first_step
      # probs = observer.prob_dict
      # if probs is not None:
      #    probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      # else:
      #    probs = " None "

      # env.mission = probs

      # sleep(0.3)


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
         observer_pos = np.array(eval(scenario["observer_pos"]))
         target_pos = np.array(eval(scenario["target_pos"]))
         observer_dir = np.array(scenario["observer_dir"])
         target_dir = np.array(scenario["target_dir"])

         flag, first_step = run(base_grid=base_grid, goals=goals, hidden_cost=hidden_cost, observer_pos=observer_pos, target_pos=target_pos, observer_dir=observer_dir, target_dir=target_dir, render_mode=None)
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
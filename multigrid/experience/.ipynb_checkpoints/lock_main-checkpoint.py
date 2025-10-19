import argparse
import numpy as np
import pandas as pd
from multigrid.envs.new_locked import AGRlocked
from multigrid.core.actions import Action
from multigrid.gr_pursuer.agents.lock_target import LockTarget,OldLockTarget
from multigrid.gr_pursuer.agents.obs import GreedyObserver,Observer


import json

def run(base_grid=None, base_rooms=None, num_rows=None, num_cols=None,
        agents_start_pos=None, agents_start_dir=None, goals=None, goal=None,
        render_mode="human", model = None):
    env = AGRlocked(base_grid=base_grid,
                    base_rooms=base_rooms,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    agents_start_pos=agents_start_pos,
                    agents_start_dir=agents_start_dir,
                    goals=goals,
                    goal=goal,
                    render_mode=None)

    observation, infos = env.reset()
    #TargetAgent  = OldLockTarget(env)
    TargetAgent = LockTarget(env)
    GreedyObserverAgent = GreedyObserver(env)
    ObserverAgent = Observer(env)

    flag = False
    first_step = -1
    steps = 0
    while not env.unwrapped.is_done() and steps <= 200:
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        if model is None:
            actions[0] = Action.stay
        elif model == "Greedy":
            actions[0] = GreedyObserverAgent.compute_action(observation)
        else:
            actions[0] = ObserverAgent.compute_action(observation)
        
        actions[1] = TargetAgent.compute_action(observation)
        print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        steps += 1
        #input()

        predicted_goal = ObserverAgent.belif_goal
        if predicted_goal == env.goal:
            if not flag:
                first_step = steps
                flag = True
                print("Success")
                print(steps)
        else:
             flag = False

    if first_step == -1:
        first_step = steps

    env.close()
    return steps,first_step
    #return steps


def main(dataset=None):
    print(dataset)
    if dataset is not None:
        scenarios = pd.read_csv(dataset)
    for idx, scenario in scenarios.iterrows():
        print(f"Scenario {idx}")
        # idx = 162  # 第40个（1-based）→ 0-based 索引为 39
        # print(f"Scenario {idx}")
        # scenario = scenarios.iloc[idx]

        # if "steps_with_help_paper2" in scenario and pd.notna(scenario["steps_with_help_paper2"]):
        #     continue
        # if "steps_with_help_gr" in scenario and pd.notna(scenario["steps_with_help_gr"]):
        #     gr_val = float(scenario["steps_with_help_gr"])
        
        #     # uniform may be missing; if missing, do NOT skip (i.e., proceed to test)
        #     uni_exists = ("steps_with_help_upperbound" in scenario) and pd.notna(scenario["steps_with_help_upperbound"])
        #     uni_val = float(scenario["steps_with_help_upperbound"]) if uni_exists else None
        
        #     if uni_exists and gr_val > uni_val:
        #         print(f"🔁 Re-test because GR ({gr_val}) > upperbound ({uni_val}).")
        #         # do NOT continue; fall through to run the test again
        #     else:
        #         print(f"✅ steps_with_help_gr existed ({gr_val}) and not worse than upperbound ({uni_val}); skip.")
        #         continue

        # 将字符串字段转为对应结构
        base_grid = np.array(eval(scenario["base_grid"]))
        base_rooms = eval(scenario["base_rooms"])
        num_rows = int(scenario["num_rows"])
        num_cols = int(scenario["num_cols"])

        agents_start_pos = eval(scenario["agents_start_pos"])
        agents_start_dir = eval(scenario["agents_start_dir"])
        goals = eval(scenario["goals"])
        goal = eval(scenario["goal"])
        
        # steps = run(base_grid=base_grid,
        #             base_rooms=base_rooms,
        #             num_rows=num_rows,
        #             num_cols=num_cols,
        #             agents_start_pos=agents_start_pos,
        #             agents_start_dir=agents_start_dir,
        #             goals=goals,
        #             goal=goal)
        # scenarios.loc[idx, "steps_baseline"] = steps
        # print(f"Steps taken without help: {steps}")
        with_steps,first_step = run(base_grid=base_grid,
                    base_rooms=base_rooms,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    agents_start_pos=agents_start_pos,
                    agents_start_dir=agents_start_dir,
                    goals=goals,
                    goal=goal,
                    model='obs')
        print(f"Steps taken with help: {with_steps}")
        print(f"First step predict correct goal: {first_step}")
        scenarios.loc[idx, "steps_with_help_paper2"] = with_steps
        scenarios.loc[idx, "steps_with_help_paper2_goal"] = first_step

        #scenarios.loc[idx, "steps_with_help_paper"] = with_steps

        # with_steps = run(base_grid=base_grid,
        #             base_rooms=base_rooms,
        #             num_rows=num_rows,
        #             num_cols=num_cols,
        #             agents_start_pos=agents_start_pos,
        #             agents_start_dir=agents_start_dir,
        #             goals=goals,
        #             goal=goal,
        #             model="obs")
        # print(f"Steps taken with help: {with_steps}")
        # scenarios.loc[idx, "steps_with_help_paper2"] = with_steps
        scenarios.to_csv("result_new.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scenarios from dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="CSV file with environment info (base_grid, base_rooms, start_pos, start_dir, goals, goal, etc.)")
    args = parser.parse_args()
    main(args.dataset)
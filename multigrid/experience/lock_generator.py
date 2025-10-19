from multigrid.envs.new_locked import AGRlocked
from multigrid.gr_pursuer.agents.lock_target import LockTarget
from multigrid.gr_pursuer.agents.obs import Observer
import multigrid.envs


import csv
import random
import argparse
import pandas as pd
import pickle
import gzip
import os

def generate_env_info(num_rows, num_cols, n_samples, writer,num_goals):
    for i in range(n_samples):
        env = AGRlocked(render_mode=None, num_rows=num_rows, num_cols=num_cols,num_goals=num_goals)
        observations, infos = env.reset()
        base_grid = infos.get('base_grid')
        base_rooms = infos.get('base_rooms')
        agents_start_pos = infos.get('agents_start_pos')
        agents_start_dir = infos.get('agents_start_dir')
        goals = infos.get('goals')
        goal = infos.get('goal')

        writer.writerow([
            num_rows,
            num_cols,
            base_grid.tolist(),
            base_rooms,
            agents_start_pos,
            agents_start_dir,
            goals,
            goal
        ])
        env.close()
        print(f"Generated {num_rows}x{num_cols} #{i+1}")

# === 主程序 ===
file_exists = os.path.exists("result.csv")

with open("result.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["num_rows", "num_cols", "base_grid", "base_rooms",
                         "agents_start_pos", "agents_start_dir", "goals", "goal"])
    for size in [3, 4, 5]:
        for num_goals in [3,5]:
            generate_env_info(num_rows=size, num_cols=size, n_samples=20,
                              writer=writer, num_goals=num_goals)

print("✅ Saved 150 scenarios to result.csv")




#env.close()

# env = AGRlocked(base_grid=base_grid,base_rooms=base_rooms)
# observations, infos = env.reset()
# base_grid1 = infos['base_grid']
# base_rooms1 = infos['base_rooms']
# env.close()


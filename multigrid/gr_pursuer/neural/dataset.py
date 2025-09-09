import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def load_scenarios(path, window):
    scenarios = pd.read_csv(path / 'scenarios.csv')
    # Filter scenarios without where the target didn't failed
    scenarios = scenarios[~scenarios["target_failed"]]

    # Filter scenarios shorter that the observation window
    scenarios = scenarios[scenarios["nsteps"] > window].reset_index(drop=True)
    return scenarios

class TargetDataset(Dataset):
    def __init__(self, path, scenarios, window):
        self.path = path
        self.window = window
        self.scenarios = scenarios

        # Add the cum_steps column
        self.scenarios["cum_steps"] = (self.scenarios["nsteps"] - window).cumsum()
    

    def __len__(self):
        length = int(self.scenarios["nsteps"].sum() - self.scenarios.shape[0] * self.window)
        return length

    def __getitem__(self, idx):

        # Index the data
        if idx<self.scenarios["cum_steps"].iloc[0]:
            scenario_idx = 0
        else:
            scenario_idx = self.scenarios[self.scenarios["cum_steps"] <= idx].index[-1] + 1
        scenario = self.scenarios.loc[scenario_idx]

        # Get the step index
        prev_cum_steps = 0 if scenario_idx==0 else int(self.scenarios["cum_steps"].loc[scenario_idx-1])
        steps_idx = idx - prev_cum_steps + self.window

        scenario = dict(scenario)
        scenario["path"] = self.path
        scenario["idx"] = idx
        scenario["step_idx"] = steps_idx
        scenario["start_idx"] = steps_idx - self.window 


        return scenario
    
    
def collate_fn(batch, data_path, size, goal_gt):
    simulation_tuples = [(sample['layout'], sample['scenario']) for sample in batch]
    simulation_tuples = list(dict.fromkeys(simulation_tuples))
    simulation_files = [f"{data_path}/layout{sample[0]}scenario{sample[1]}.csv" for sample in simulation_tuples]

    simulated_data = {key: pd.read_csv(path) for key, path in zip(simulation_tuples, simulation_files)}
    new_batch = []
    for sample in batch:
        layout = sample['layout']
        scenario = sample['scenario']
        data_dir = sample['path']

        idx = sample['idx']
        step_idx = sample['step_idx']
        start_idx = sample['start_idx']

        grid = np.array(eval(sample['base_grid']), dtype=np.int8).flatten()
        target_goal = np.array(eval(sample['target_goal']), dtype=np.int8).flatten()/size
        goals = np.array(eval(sample['goals']), dtype=np.int8).flatten()/size
        goals = target_goal if goal_gt else goals
        
        sim_df = simulated_data[(layout, scenario)]

        one_hot_encode = lambda d: np.eye(4, dtype=np.int8)[d]
        sim_df['target_pos_encoded'] = sim_df['target_pos'].apply(lambda x: np.array(eval(x), dtype=np.int8).flatten()/size)
        sim_df['target_dir_encoded'] = sim_df['target_dir'].apply(one_hot_encode).tolist()
        one_hot_encode = lambda d: np.eye(3, dtype=np.int8)[d]
        sim_df['action_encoded'] = sim_df['target_action'].astype(int).apply(one_hot_encode).tolist()
        sim_df['done'] = sim_df['done'].astype(int)  # Indicate when an instance resets

        # print(idx, step_idx, sample["nsteps"], sim_df.shape[0], layout, scenario)
        step_row = sim_df.loc[step_idx]
        join_state = lambda row: list(row["target_pos_encoded"]) + row["target_dir_encoded"].tolist()
        
        # print(idx, start_idx, step_idx, sample["nsteps"])
        sequence_data = sim_df.loc[start_idx:step_idx]
        state = sequence_data.apply(join_state, axis=1).tolist()
        action = step_row['action_encoded']

        next_sequence_data = sim_df.loc[step_idx+1:]
        if next_sequence_data.empty:
            next_states = []
        else:
            next_states = sim_df.loc[step_idx+1:].apply(join_state, axis=1).tolist()

        item = {
            "state": torch.tensor(state, dtype=torch.float32),
            "grid": torch.tensor(grid, dtype=torch.float32),
            "goals": torch.tensor(goals, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "next_states": torch.tensor(next_states, dtype=torch.float32)
        }   
        new_batch.append(item)      

    # Concatenate the batch
    batch = {key: [item[key] for item in new_batch] for key in new_batch[0].keys()}
    batch = {key: torch.stack((value)) if key!="next_states" else value for key, value in batch.items()}

    return batch
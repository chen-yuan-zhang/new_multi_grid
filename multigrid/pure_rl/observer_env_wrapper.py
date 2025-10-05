# observer_env_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from multigrid.core.actions import Action
import numpy as np
import pandas as pd
import json
from time import sleep
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.pure_rl.obs_to_belief_image_array import obs_to_belief_image_array
# import gym action space Discrete
import gymnasium.spaces as spaces
from scipy.stats import entropy
import cv2

class ObserverEnvDirectRew(gym.Env):
    """
    Single-agent wrapper around AGREnv:
      - RLlib controls ONLY the observer.
      - Target acts via a scripted/random policy inside the wrapper.
      - Observation = features from belief (fixed-size Box).
      - Reward = custom observer reward (terminal + telescoping shaping).
    """
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        # ---- Rewarder ---- (gamma MUST match PPO gamma)
        self.gamma = float(config.get("gamma", 0.995))
        # ---- Base env ----
        # required: CSV path
        self.dataset = config.get("dataset")
        if self.dataset is None:
            raise ValueError("config['dataset'] is required (path to scenarios CSV).")

        self.scenarios = pd.read_csv(self.dataset)
        if len(self.scenarios) == 0:
            raise ValueError(f"Dataset CSV at '{self.dataset}' is empty.")
        
        # probe first row (same parsing style as your script) to get a real action_space
        first = self.scenarios.iloc[0]
        base_grid   = np.array(eval(first["base_grid"]))
        goals       = eval(first["goals"])
        hidden_cost = np.array(eval(first["hidden_cost"]))
        observer_pos = eval(first["observer_pos"])
        target_pos   = eval(first["target_pos"])
        observer_dir = int(first["observer_dir"])
        target_dir   = int(first["target_dir"])
        goal         = eval(first["goal"])

        probe_env = AGREnv(
            base_grid=base_grid,
            goals=goals,
            hidden_cost=hidden_cost,
            goal=goal,
            enable_hidden_cost=True,
            agents_start_pos=[observer_pos, target_pos],
            agents_start_dir=[observer_dir, target_dir],
            render_mode=self.render_mode,
        )
        self.action_space = spaces.Discrete(4) # observer has 4 discrete actions from 0 to 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(124, 124, 3), dtype=np.float32)  # belief image

        del probe_env
        
        # runtime state
        self.env = None
        self.belief_observer = None
        self._target_actions = []
        self._terminated = False
        self._truncated = False
        self._step_idx = 0
        

    # --- Helpers ---

    def _next_target_action(self):
        return self._target_actions[self._step_idx] if self._step_idx < len(self._target_actions) else self._target_actions[-1]

    # --- Gymnasium API ---
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._terminated = self._truncated = False
        self._step_idx = 0
        
        # --- Smart task rotation for balanced multi-task learning ---
        if not hasattr(self, '_task_rotation_counter'):
            self._task_rotation_counter = 0
        
        # Rotate through different hidden_cost_types to ensure balanced sampling
        task_types = self.scenarios['hidden_cost_type'].unique()
        if len(task_types) > 1:
            target_task_type = task_types[self._task_rotation_counter % len(task_types)]
            task_scenarios = self.scenarios[self.scenarios['hidden_cost_type'] == target_task_type]
            
            if len(task_scenarios) > 0:
                idx = np.random.randint(len(task_scenarios))
                scenario = task_scenarios.iloc[idx]
                self._task_rotation_counter += 1
                print(f"Task rotation: selected hidden_cost_type={target_task_type}")
            else:
                # Fallback to random selection
                idx = np.random.randint(len(self.scenarios))
                scenario = self.scenarios.iloc[idx]
        else:
            # Only one task type available
            idx = np.random.randint(len(self.scenarios))
            scenario = self.scenarios.iloc[idx]
        
        base_grid   = np.array(eval(scenario["base_grid"]))
        goals       = eval(scenario["goals"])
        hidden_cost = np.array(eval(scenario["hidden_cost"]))
        observer_pos = eval(scenario["observer_pos"])
        target_pos   = eval(scenario["target_pos"])
        observer_dir = int(scenario["observer_dir"])
        target_dir   = int(scenario["target_dir"])
        actions      = [Action(v) for v in json.loads(scenario["all_actions"])]
        goal         = eval(scenario["goal"])

        # fresh AGREnv for this episode
        self.env = AGREnv(
            base_grid=base_grid,
            goals=goals,
            hidden_cost=hidden_cost,
            goal=goal,
            enable_hidden_cost=True,
            agents_start_pos=[observer_pos, target_pos],
            agents_start_dir=[observer_dir, target_dir],
            render_mode=self.render_mode,
        )

        # reset scripted target action sequence
        self._target_actions = actions

        obs, info = self.env.reset()
        self.belief_observer = BeliefUpdateObserver(self.env)

        _ = self.belief_observer.compute_action(obs[0], render_and_save=False, get_action=False)  # init belief
        
        
        augmented_obs = self.belief_observer.augment_observation(obs)
        
        belief_img, log_belief_sum = obs_to_belief_image_array(self.belief_observer, None, obs[0])

        belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0

        
        # log_belief_sum is array (H,W), try to normalize it so max is 1 and min is 0
        # curr value is ranged from -20 to -5
        log_belief_sum = (log_belief_sum - np.min(log_belief_sum)) / (np.max(log_belief_sum) - np.min(log_belief_sum) + 1e-10)

        # Initial belief + rewarder reset
        init_belief = augmented_obs[0]['goal_belief']
        info = {}
        return belief_img, info
    
    
    def step(self, action):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode is done. Call reset().")

        assert isinstance(self.belief_observer, BeliefUpdateObserver), "Call reset() before step()."

        # Compose multi-agent actions dict: observer (index 0) + target (index 1).
        actions = {0: action, 1: self._next_target_action()}
        next_obs, _, _, _, _ = self.env.step(actions)
        self._step_idx += 1
        
        _ = self.belief_observer.compute_action(next_obs[0], render_and_save=False, get_action=False)  # update belief

        # Update belief & extract features
        augmented_obs = self.belief_observer.augment_observation(next_obs)
        goal_belief = augmented_obs[0]['goal_belief']
        
        
        belief_img, log_belief_sum = obs_to_belief_image_array(self.belief_observer, "/home/sukaih/Extrastorage/new_multi_grid_new_rl/multigrid/pure_rl/debug_belief_img.png", next_obs[0], add_noise=False, behavior_type=0)
        cvt_img = cv2.cvtColor(belief_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/sukaih/Extrastorage/new_multi_grid_new_rl/multigrid/pure_rl/debug_belief_img.png", cvt_img)


        belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0
        
        
        infos = {}
        
        # Custom reward for observer
        # r_t will be the neg entropy of log_belief_sum and neg entropy of goal_belief, that means, more concentrated belief, more reward
        # rl just reward when in view, otherwise 0 reward
        if "target_pos" in next_obs[0] or self.belief_observer.pos == self.env.target.pos:
            r_t = 1.0
            # auxiliary reward if argmax of goal_belief is correct
            goal_max = None 
            goal_belief_val = -1.0
            for g, v in goal_belief.items():
                if v > goal_belief_val:
                    goal_belief_val = v
                    goal_max = g
            if goal_max == self.env.goal:
                r_t += 1.0
        else:
            r_t = 0.0

        termination = truncation = self.env.unwrapped.is_done()
        self._terminated = termination
        self._truncated = truncation

        return belief_img, float(r_t), termination, truncation, infos


if __name__ == "__main__":
    config = {
        "dataset": "results.csv",
        "gamma": 0.995,
    }
    
    env = ObserverEnvDirectRew(config)
    breakpoint()
    obs, info = env.reset()
    breakpoint()
    max_steps = 1000
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
        if terminated or truncated:
            break
    
    print("Episode finished.")
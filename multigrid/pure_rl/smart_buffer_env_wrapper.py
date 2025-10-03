# smart_buffer_env_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque, defaultdict
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
from multigrid.core.actions import Action
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.pure_rl.obs_to_belief_image_array import obs_to_belief_image_array
import random


class TaskAwareExperienceBuffer:
    """
    Smart buffer that maintains balanced experiences across different task types.
    """
    
    def __init__(self, capacity_per_task: int = 1000, num_task_types: int = 4):
        self.capacity_per_task = capacity_per_task
        self.num_task_types = num_task_types
        
        # Separate buffers for each task type
        self.task_buffers = {
            task_id: deque(maxlen=capacity_per_task) 
            for task_id in range(num_task_types)
        }
        
        # Track task statistics
        self.task_counts = defaultdict(int)
        self.task_episode_lengths = defaultdict(list)
        
    def add_episode(self, episode_data: List[Dict], task_id: int):
        """Add a complete episode to the appropriate task buffer."""
        self.task_buffers[task_id].append(episode_data)
        self.task_counts[task_id] += 1
        self.task_episode_lengths[task_id].append(len(episode_data))
        
    def sample_balanced_batch(self, batch_size: int) -> List[Dict]:
        """Sample a balanced batch ensuring representation from all task types."""
        samples_per_task = batch_size // self.num_task_types
        remainder = batch_size % self.num_task_types
        
        batch = []
        
        for task_id in range(self.num_task_types):
            if len(self.task_buffers[task_id]) == 0:
                continue
                
            # Base samples per task
            n_samples = samples_per_task
            
            # Distribute remainder
            if remainder > 0:
                n_samples += 1
                remainder -= 1
            
            # Sample episodes from this task type
            available_episodes = list(self.task_buffers[task_id])
            
            transitions_collected = 0
            while transitions_collected < n_samples and available_episodes:
                episode = random.choice(available_episodes)
                
                # Sample transitions from this episode
                transitions_needed = min(n_samples - transitions_collected, len(episode))
                sampled_transitions = random.sample(episode, transitions_needed)
                
                batch.extend(sampled_transitions)
                transitions_collected += len(sampled_transitions)
        
        return batch
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics for monitoring."""
        stats = {
            'task_counts': dict(self.task_counts),
            'buffer_sizes': {task_id: len(buffer) for task_id, buffer in self.task_buffers.items()},
            'avg_episode_lengths': {
                task_id: np.mean(lengths) if lengths else 0 
                for task_id, lengths in self.task_episode_lengths.items()
            }
        }
        return stats


class SmartBufferObserverEnv(gym.Env):
    """
    Enhanced observer environment with task-aware experience buffering.
    """
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        
        # Initialize base environment components
        self.gamma = float(config.get("gamma", 0.995))
        self.dataset = config.get("dataset")
        if self.dataset is None:
            raise ValueError("config['dataset'] is required (path to scenarios CSV).")

        self.scenarios = pd.read_csv(self.dataset)
        if len(self.scenarios) == 0:
            raise ValueError(f"Dataset CSV at '{self.dataset}' is empty.")
        
        # Smart buffer configuration
        buffer_config = config.get("buffer_config", {})
        self.use_buffer = buffer_config.get("enabled", True)
        self.buffer_capacity = buffer_config.get("capacity_per_task", 500)
        self.buffer_sample_ratio = buffer_config.get("sample_ratio", 0.3)  # 30% from buffer, 70% fresh
        
        # Initialize smart buffer
        if self.use_buffer:
            self.experience_buffer = TaskAwareExperienceBuffer(
                capacity_per_task=self.buffer_capacity,
                num_task_types=4  # Based on hidden_cost_type: 0,1,2,3
            )
            self.current_episode_buffer = []
        
        # Task balancing
        self.task_rotation_enabled = config.get("task_rotation", True)
        self.task_counter = 0
        
        # Standard gym setup
        first = self.scenarios.iloc[0]
        base_grid = np.array(eval(first["base_grid"]))
        goals = eval(first["goals"])
        hidden_cost = np.array(eval(first["hidden_cost"]))
        observer_pos = eval(first["observer_pos"])
        target_pos = eval(first["target_pos"])
        observer_dir = int(first["observer_dir"])
        target_dir = int(first["target_dir"])
        goal = eval(first["goal"])

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
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(124, 124, 3), dtype=np.float32)
        
        del probe_env
        
        # Runtime state
        self.env = None
        self.belief_observer = None
        self._target_actions = []
        self._terminated = False
        self._truncated = False
        self._step_idx = 0
        self.current_task_id = None

    def _get_task_id(self, scenario) -> int:
        """Extract task ID from scenario for smart buffering."""
        return int(scenario["hidden_cost_type"])
    
    def _select_scenario(self):
        """Smart scenario selection with task balancing."""
        if self.task_rotation_enabled:
            # Rotate through task types to ensure balance
            task_types = [0, 1, 2, 3]
            target_task_type = task_types[self.task_counter % len(task_types)]
            
            # Find scenarios of target task type
            task_scenarios = self.scenarios[self.scenarios['hidden_cost_type'] == target_task_type]
            if len(task_scenarios) > 0:
                scenario = task_scenarios.sample(n=1).iloc[0]
                self.task_counter += 1
                return scenario
        
        # Fallback to random selection
        idx = np.random.randint(len(self.scenarios))
        return self.scenarios.iloc[idx]

    def _next_target_action(self):
        return self._target_actions[self._step_idx] if self._step_idx < len(self._target_actions) else self._target_actions[-1]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Store previous episode in buffer if it exists
        if (self.use_buffer and hasattr(self, 'current_episode_buffer') and 
            len(self.current_episode_buffer) > 0 and self.current_task_id is not None):
            self.experience_buffer.add_episode(self.current_episode_buffer, self.current_task_id)
        
        # Reset episode state
        self._terminated = self._truncated = False
        self._step_idx = 0
        self.current_episode_buffer = []
        
        # Smart scenario selection
        scenario = self._select_scenario()
        self.current_task_id = self._get_task_id(scenario)
        
        print(f"Selected scenario with task_id={self.current_task_id}, hidden_cost_type={scenario['hidden_cost_type']}")
        
        # Initialize environment
        base_grid = np.array(eval(scenario["base_grid"]))
        goals = eval(scenario["goals"])
        hidden_cost = np.array(eval(scenario["hidden_cost"]))
        observer_pos = eval(scenario["observer_pos"])
        target_pos = eval(scenario["target_pos"])
        observer_dir = int(scenario["observer_dir"])
        target_dir = int(scenario["target_dir"])
        actions = [Action(v) for v in json.loads(scenario["all_actions"])]
        goal = eval(scenario["goal"])

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

        self._target_actions = actions
        obs, info = self.env.reset()
        self.belief_observer = BeliefUpdateObserver(self.env)
        _ = self.belief_observer.compute_action(obs[0], render_and_save=False, get_action=False)
        
        augmented_obs = self.belief_observer.augment_observation(obs)
        belief_img, log_belief_sum = obs_to_belief_image_array(self.belief_observer, None, obs[0])
        belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0
        log_belief_sum = (log_belief_sum - np.min(log_belief_sum)) / (np.max(log_belief_sum) - np.min(log_belief_sum) + 1e-10)

        # Store initial observation in episode buffer
        if self.use_buffer:
            initial_experience = {
                'obs': belief_img.copy(),
                'task_id': self.current_task_id,
                'step': self._step_idx
            }
            self.current_episode_buffer.append(initial_experience)

        return belief_img, info

    def step(self, action):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode is done. Call reset().")

        # Execute step
        actions = {0: action, 1: self._next_target_action()}
        next_obs, _, _, _, _ = self.env.step(actions)
        self._step_idx += 1
        
        _ = self.belief_observer.compute_action(next_obs[0], render_and_save=False, get_action=False)
        augmented_obs = self.belief_observer.augment_observation(next_obs)
        goal_belief = augmented_obs[0]['goal_belief']
        
        belief_img, log_belief_sum = obs_to_belief_image_array(self.belief_observer, None, next_obs[0], add_noise=True)
        belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0
        log_belief_sum = (log_belief_sum - np.min(log_belief_sum)) / (np.max(log_belief_sum) - np.min(log_belief_sum) + 1e-10)
        
        # Calculate reward
        if "target_pos" in next_obs[0] or self.belief_observer.pos == self.env.target.pos:
            r_t = 1.0
            goal_max = max(goal_belief.items(), key=lambda x: x[1])[0]
            if goal_max == self.env.goal:
                r_t += 1.0
        else:
            r_t = 0.0

        termination = truncation = self.env.unwrapped.is_done()
        self._terminated = termination
        self._truncated = truncation

        # Store experience in episode buffer
        if self.use_buffer:
            experience = {
                'obs': belief_img.copy(),
                'action': action,
                'reward': r_t,
                'next_obs': belief_img.copy(),
                'terminated': termination,
                'truncated': truncation,
                'task_id': self.current_task_id,
                'step': self._step_idx
            }
            self.current_episode_buffer.append(experience)

        return belief_img, float(r_t), termination, truncation, {}

    def get_buffer_statistics(self):
        """Get smart buffer statistics for monitoring."""
        if not self.use_buffer:
            return {}
        return self.experience_buffer.get_statistics()

    def sample_buffer_experiences(self, batch_size: int):
        """Sample experiences from the smart buffer."""
        if not self.use_buffer or len(self.experience_buffer.task_buffers) == 0:
            return []
        return self.experience_buffer.sample_balanced_batch(batch_size)
# smart_buffer_env_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque, defaultdict
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional
from multigrid.core.actions import Action
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.pure_rl.obs_to_belief_image_array import obs_to_belief_image_array
import random


class CurriculumAwareExperienceBuffer:
    """
    Smart buffer that maintains balanced experiences across different curriculum levels.
    Supports curriculum learning based on scenario size and initial distance.
    """
    
    def __init__(self, capacity_per_task: int = 1000, sizes: Optional[List[int]] = None, initial_distances: Optional[List[int]] = None):
        self.capacity_per_task = capacity_per_task
        self.sizes = sizes if sizes is not None else [10, 12, 15]
        self.initial_distances = initial_distances if initial_distances is not None else [3, 5, 7]
        
        # Create curriculum levels from combinations of size and distance
        self.curriculum_levels = []
        self.level_to_id = {}
        level_id = 0
        
        for size in self.sizes:
            for distance in self.initial_distances:
                level = (size, distance)
                self.curriculum_levels.append(level)
                self.level_to_id[level] = level_id
                level_id += 1
        
        self.num_task_types = len(self.curriculum_levels)
        
        # Separate buffers for each curriculum level
        self.task_buffers = {
            level_id: deque(maxlen=capacity_per_task) 
            for level_id in range(self.num_task_types)
        }
        
        # Curriculum progression tracking
        self.current_curriculum_stage = 0  # Start with easiest tasks
        self.convergence_threshold = 0.8  # Threshold for moving to next stage
        self.episode_success_rates = defaultdict(deque)  # Track recent success rates
        self.success_window_size = 100  # Number of episodes to consider for convergence
        
        # Track task statistics
        self.task_counts = defaultdict(int)
        self.task_episode_lengths = defaultdict(list)
        
        
    def add_episode(self, episode_data: List[Dict], task_id: int, success: bool = False):
        """Add a complete episode to the appropriate curriculum level buffer."""
        self.task_buffers[task_id].append(episode_data)
        self.task_counts[task_id] += 1
        self.task_episode_lengths[task_id].append(len(episode_data))
        
        # Track success rate for curriculum progression
        if len(self.episode_success_rates[task_id]) >= self.success_window_size:
            self.episode_success_rates[task_id].popleft()
        self.episode_success_rates[task_id].append(success)
        
    def get_curriculum_difficulty(self, size: int, distance: int) -> float:
        """Calculate difficulty score for a curriculum level (lower = easier)."""
        # Normalize both dimensions and combine them
        size_difficulty = (size - min(self.sizes)) / (max(self.sizes) - min(self.sizes)) if len(set(self.sizes)) > 1 else 0
        distance_difficulty = (distance - min(self.initial_distances)) / (max(self.initial_distances) - min(self.initial_distances)) if len(set(self.initial_distances)) > 1 else 0
        return size_difficulty + distance_difficulty
    
    def get_current_curriculum_levels(self) -> List[Tuple[int, int]]:
        """Get the curriculum levels that should be trained on currently."""
        # Sort curriculum levels by difficulty
        sorted_levels = sorted(self.curriculum_levels, key=lambda x: self.get_curriculum_difficulty(x[0], x[1]))
        
        # Determine how many levels to include based on current stage
        max_stage = min(self.current_curriculum_stage + 1, len(sorted_levels))
        return sorted_levels[:max_stage]
    
    def should_advance_curriculum(self) -> bool:
        """Check if we should advance to the next curriculum stage."""
        current_levels = self.get_current_curriculum_levels()
        
        # Check if all current levels have sufficient success rate
        for level in current_levels:
            level_id = self.level_to_id[level]
            if level_id not in self.episode_success_rates:
                return False
            
            success_rates = list(self.episode_success_rates[level_id])
            if len(success_rates) < self.success_window_size:
                return False
                
            recent_success_rate = sum(success_rates) / len(success_rates)
            if recent_success_rate < self.convergence_threshold:
                return False
        
        return True
    
    def advance_curriculum(self):
        """Advance to the next curriculum stage."""
        if self.current_curriculum_stage < len(self.curriculum_levels) - 1:
            self.current_curriculum_stage += 1
            print(f"Advanced curriculum to stage {self.current_curriculum_stage}")
            print(f"Current levels: {self.get_current_curriculum_levels()}")
        
    def sample_balanced_batch(self, batch_size: int) -> List[Dict]:
        """Sample a balanced batch from current curriculum levels."""
        current_levels = self.get_current_curriculum_levels()
        current_level_ids = [self.level_to_id[level] for level in current_levels]
        
        # Filter to only include levels with data
        available_level_ids = [lid for lid in current_level_ids if len(self.task_buffers[lid]) > 0]
        
        if not available_level_ids:
            return []
            
        samples_per_task = batch_size // len(available_level_ids)
        remainder = batch_size % len(available_level_ids)
        
        batch = []
        
        for level_id in available_level_ids:
            # Base samples per task
            n_samples = samples_per_task
            
            # Distribute remainder
            if remainder > 0:
                n_samples += 1
                remainder -= 1
            
            # Sample episodes from this curriculum level
            available_episodes = list(self.task_buffers[level_id])
            
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
        current_levels = self.get_current_curriculum_levels()
        current_level_ids = [self.level_to_id[level] for level in current_levels]
        
        stats = {
            'curriculum_stage': self.current_curriculum_stage,
            'current_levels': current_levels,
            'task_counts': dict(self.task_counts),
            'buffer_sizes': {level_id: len(buffer) for level_id, buffer in self.task_buffers.items()},
            'avg_episode_lengths': {
                level_id: np.mean(lengths) if lengths else 0 
                for level_id, lengths in self.task_episode_lengths.items()
            },
            'success_rates': {
                level_id: np.mean(list(success_rates)) if success_rates else 0
                for level_id, success_rates in self.episode_success_rates.items()
            },
            'should_advance': self.should_advance_curriculum()
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
        
        # Initialize render mode before other components
        self.render_mode = config.get("render_mode", None)
        
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
            # Get curriculum configuration
            curriculum_config = buffer_config.get("curriculum", {})
            sizes = curriculum_config.get("sizes", [10, 12, 15])
            initial_distances = curriculum_config.get("initial_distances", [3, 5, 7])
            
            self.experience_buffer = CurriculumAwareExperienceBuffer(
                capacity_per_task=self.buffer_capacity,
                sizes=sizes,
                initial_distances=initial_distances
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
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)
        self.single_action_space = self.action_space 

        
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
        """Extract curriculum task ID from scenario based on size and initial_distance."""
        size = int(scenario["size"])
        initial_distance = int(scenario["initial_distance"])
        
        # Find the corresponding curriculum level
        level = (size, initial_distance)
        if level in self.experience_buffer.level_to_id:
            return self.experience_buffer.level_to_id[level]
        else:
            # If not found, use the closest level (fallback)
            print(f"Warning: Curriculum level {level} not found, using closest match")
            closest_level = min(self.experience_buffer.curriculum_levels, 
                              key=lambda x: abs(x[0] - size) + abs(x[1] - initial_distance))
            return self.experience_buffer.level_to_id[closest_level]
    
    def _select_scenario(self):
        """Smart scenario selection with curriculum learning."""
        if self.use_buffer and hasattr(self, 'experience_buffer'):
            # Check if we should advance curriculum
            if self.experience_buffer.should_advance_curriculum():
                self.experience_buffer.advance_curriculum()
            
            # Get current curriculum levels
            current_levels = self.experience_buffer.get_current_curriculum_levels()
            
            # Filter scenarios to only include current curriculum levels
            if current_levels:
                valid_scenarios = []
                for _, scenario in self.scenarios.iterrows():
                    size = int(scenario["size"])
                    initial_distance = int(scenario["initial_distance"])
                    if (size, initial_distance) in current_levels:
                        valid_scenarios.append(scenario)
                
                if valid_scenarios:
                    # Sample from valid scenarios
                    selected_scenario = valid_scenarios[np.random.randint(len(valid_scenarios))]
                    print(f"Selected curriculum scenario: size={selected_scenario['size']}, "
                          f"initial_distance={selected_scenario['initial_distance']}")
                    return selected_scenario
        
        # Fallback to random selection if buffer not available or no valid scenarios
        idx = np.random.randint(len(self.scenarios))
        scenario = self.scenarios.iloc[idx]
        print(f"Selected random scenario: size={scenario['size']}, "
              f"initial_distance={scenario['initial_distance']}")
        return scenario

    def _next_target_action(self):
        return self._target_actions[self._step_idx] if self._step_idx < len(self._target_actions) else self._target_actions[-1]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Store previous episode in buffer if it exists
        if (self.use_buffer and hasattr(self, 'current_episode_buffer') and 
            len(self.current_episode_buffer) > 0 and self.current_task_id is not None):
            
            # Calculate episode success based on whether goal was achieved
            episode_success = getattr(self, '_last_episode_success', False)
            self.experience_buffer.add_episode(
                self.current_episode_buffer, 
                self.current_task_id,
                success=episode_success
            )
        
        # Reset episode state
        self._terminated = self._truncated = False
        self._step_idx = 0
        self.current_episode_buffer = []
        self._last_episode_success = False  # Track success for curriculum
        
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
        self._at_least_once_see_in_view = False

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
        
        belief_img, log_belief_sum = obs_to_belief_image_array(self.belief_observer, None, next_obs[0], add_noise=False)
        belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0
        log_belief_sum = (log_belief_sum - np.min(log_belief_sum)) / (np.max(log_belief_sum) - np.min(log_belief_sum) + 1e-10)
        
        # Calculate reward
        goal_max = -1
        if "target_pos" in next_obs[0] or self.belief_observer.pos == self.env.target.pos:
            if not self._at_least_once_see_in_view:
                self._at_least_once_see_in_view = True
            r_t = 1.0
        else:
            r_t = 0.0
        if self._at_least_once_see_in_view:
            goal_max = max(goal_belief.items(), key=lambda x: x[1])[0]

            # auxiliary rewards -> when agent is confident, also reward
            goal_prob_max = max(goal_belief.values())
            if goal_prob_max > 0.66: 
                if goal_max == self.env.goal:
                    goal_prob_rew = goal_prob_max - (1/len(goal_belief))
                    r_t += goal_prob_rew

        termination = truncation = self.env.unwrapped.is_done()
        self._terminated = termination
        self._truncated = truncation
        
        # Track episode success for curriculum learning
        if termination:
            if goal_max == self.env.goal:
                r_t += 5.0 # one-off reward for correct goal identification
            # Episode is successful if goal was correctly identified
            self._last_episode_success = (goal_max == self.env.goal)  # Substantial reward indicates success
            self._at_least_once_see_in_view = False

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
    
    def advance_curriculum_stage(self):
        """Manually advance curriculum stage."""
        if self.use_buffer and hasattr(self, 'experience_buffer'):
            self.experience_buffer.advance_curriculum()
            
    def get_current_curriculum_info(self):
        """Get current curriculum information."""
        if self.use_buffer and hasattr(self, 'experience_buffer'):
            return {
                'current_stage': self.experience_buffer.current_curriculum_stage,
                'current_levels': self.experience_buffer.get_current_curriculum_levels(),
                'should_advance': self.experience_buffer.should_advance_curriculum()
            }
        return {}
    


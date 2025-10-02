from .base import BaseAgent
from ..astar import astar, get_successor, execute_action, get_obs_successor, get_reverse_successor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction
from multigrid.core.actions import Action
from multigrid.utils.obs import gen_obs_grid_encoding
from multigrid.core.constants import Type
import os
from collections import deque
from time import sleep

# from .neuro_predictor import neuro_predict

BETA = 1

# Behavior type indices (extend if more behaviors are needed)
BEHAVIOR_TYPES = [0, 1, 2, 3]

class BeliefUpdateObserver(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None, use_neural_predictor = False, use_log_space = True, use_neural_when_in_view_only = False):
        # For Sukai: set use_neural_predictor = True to use neural predictor
        
        super().__init__(env.observer)


        if use_neural_predictor:
            from .neuro_predictor import neuro_predict
            self.neuro_predict = neuro_predict
            print("Using neural predictor for transition probabilities.")

        self.env = env
        self.agent.name = "BeliefUpdateObserver"
        self.agent.can_overlap = True

        self.goals = env.goals
        self.step = -1
        self.pos = env.observer.pos
        self.dir = env.observer.dir
        self.use_neural_predictor = use_neural_predictor
        self.use_neural_when_in_view_only = use_neural_when_in_view_only
        self.use_log_space = use_log_space

        if init_goal_belief:
            self.goal_belief = init_goal_belief
        else:
            self.goal_belief = { g:1/len(self.goals) for g in self.goals}

        # actor_belief now: {goal: [belief_grid_behavior0, belief_grid_behavior1, ...]}
        if init_actor_belief:
            self.actor_belief = init_actor_belief  # assume already in list format per goal
        else:
            if self.use_log_space:
                self.actor_belief = {
                    g: [
                        set_uniform_log_prob(env.base_grid, self.goal_belief[g] / len(BEHAVIOR_TYPES))
                        for _ in BEHAVIOR_TYPES
                    ]
                    for g in self.goals
                }
            else:
                self.actor_belief = {
                    g: [
                        set_uniform_prob(env.base_grid, self.goal_belief[g] / len(BEHAVIOR_TYPES))
                        for _ in BEHAVIOR_TYPES
                    ]
                    for g in self.goals
                }

        self.dist_matrix = self.compute_pairwise_distances()
        # self.behavior_type_belief = {b: 1/len(BEHAVIOR_TYPES) for b in BEHAVIOR_TYPES}  # uniform prior over behavior types
        
        # Cache for transition probabilities to avoid redundant calculations
        self.transition_prob_cache = {}
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def logsumexp(self, log_probs):
        """Numerically stable log-sum-exp operation."""
        if not log_probs:
            return -np.inf
        log_probs = np.array(log_probs)
        max_val = np.max(log_probs)
        if max_val == -np.inf:
            return -np.inf
        return max_val + np.log(np.sum(np.exp(log_probs - max_val)))
    
    def logaddexp(self, log_a, log_b):
        """Numerically stable log addition: log(exp(log_a) + exp(log_b))."""
        # Handle NaN cases
        if np.isnan(log_a) or np.isnan(log_b):
            return -np.inf  # Treat NaN as zero probability
        
        # Handle -inf cases (zero probability)
        if log_a == -np.inf:
            return log_b
        if log_b == -np.inf:
            return log_a
        
        # Use numpy's stable implementation for all other cases (including +inf)
        return np.logaddexp(log_a, log_b)
    
    def normalize_actor_beliefs_to_one(self):
        """
        Normalize actor beliefs so that the total probability across all goals and behaviors sums to 1.
        This ensures proper probability distribution after each update step.
        """
        # Calculate total probability mass across all goals and behaviors
        total_mass = 0.0
        
        if self.use_log_space:
            # Collect all valid log probabilities
            all_log_probs = []
            for goal in self.goals:
                for behavior_grid in self.actor_belief[goal]:
                    valid_probs = behavior_grid[behavior_grid > -np.inf]
                    if len(valid_probs) > 0:
                        all_log_probs.extend(valid_probs)
            
            if all_log_probs:
                # Calculate total mass in regular space
                total_mass = np.sum(np.exp(np.clip(all_log_probs, -700, 700)))
            
            # Normalize in log-space
            if total_mass > 1e-15:  # Avoid division by very small numbers
                log_normalization_factor = np.log(total_mass)
                
                for goal in self.goals:
                    for behavior_grid in self.actor_belief[goal]:
                        # Subtract log normalization factor (equivalent to division in regular space)
                        behavior_grid[behavior_grid > -np.inf] -= log_normalization_factor
            else:
                # If total mass is too small, reset to uniform distribution
                self._reset_to_uniform_log_distribution()
        else:
            # Regular space - calculate total mass
            for goal in self.goals:
                for behavior_grid in self.actor_belief[goal]:
                    total_mass += np.sum(behavior_grid)
            
            # Normalize to sum to 1
            if total_mass > 1e-15:
                for goal in self.goals:
                    for behavior_grid in self.actor_belief[goal]:
                        behavior_grid /= total_mass
            else:
                # If total mass is too small, reset to uniform distribution
                self._reset_to_uniform_regular_distribution()
    
    def _reset_to_uniform_log_distribution(self):
        """Reset actor beliefs to uniform log distribution."""
        for goal in self.goals:
            goal_prob = self.goal_belief[goal]
            for behavior_idx in range(len(self.actor_belief[goal])):
                self.actor_belief[goal][behavior_idx] = set_uniform_log_prob(
                    self.env.base_grid, goal_prob / len(BEHAVIOR_TYPES)
                )
    
    def _reset_to_uniform_regular_distribution(self):
        """Reset actor beliefs to uniform regular distribution."""
        for goal in self.goals:
            goal_prob = self.goal_belief[goal]
            for behavior_idx in range(len(self.actor_belief[goal])):
                self.actor_belief[goal][behavior_idx] = set_uniform_prob(
                    self.env.base_grid, goal_prob / len(BEHAVIOR_TYPES)
                )


    def compute_pairwise_distances(self):
        """
        Compute all pairwise distances from each state (position and direction) to the goal locations using BFS.
        """
        free_cells = np.argwhere(self.env.base_grid == 0)
        
        num_cells = len(free_cells)
        num_directions = 4  # Number of possible directions (east, south, west, north)
        num_states = num_cells * num_directions

        cell_to_index = {tuple(cell): idx for idx, cell in enumerate(free_cells)}

        # Initialize distance matrix for distances to goal locations
        dist_matrix = {goal: np.full(num_states, np.inf) for goal in self.goals}
        for goal in self.goals:
            goal_idx = cell_to_index[tuple(goal)]
            queue = deque([(goal_idx, dir, 0) for dir in range(num_directions)])  # (cell_index, direction, distance)
            visited = set()

            while queue:
                current_idx, current_dir, current_dist = queue.popleft()
                state = (current_idx, current_dir)
                if state in visited:
                    continue
                visited.add(state)

                dist_matrix[goal][current_idx * num_directions + current_dir] = current_dist

                pos_state = (free_cells[current_idx], current_dir)
                for action, next_pos_state in get_reverse_successor(self.env, pos_state):
                    next_pos, next_dir = next_pos_state
                    if tuple(next_pos) in cell_to_index:
                        next_idx = cell_to_index[tuple(next_pos)]
                        queue.append((next_idx, next_dir, current_dist + 1))

        adjusted_dist_matrix = dict()

        for i in range(num_states):
            for goal in self.goals:
                cell = free_cells[i // num_directions]
                pos_state = (tuple(cell), i % num_directions)
                adjusted_dist_matrix[(pos_state, tuple(goal))] = dist_matrix[goal][i]

        return adjusted_dist_matrix
    
    def get_cached_transition_probs(self, pos_state, goal, behavior_idx, successors, beta=BETA, use_neural=False, use_neural_when_in_view_only=False, target_visible=False):
        """
        Get transition probabilities with caching to avoid redundant calculations.
        
        Parameters:
        pos_state: Current position and direction state
        goal: Goal position
        behavior_idx: Behavior type index
        successors: List of possible successor states
        beta: Temperature parameter
        use_neural: Whether to use neural predictor (if available)
        
        Returns:
        dict: Transition probabilities for each successor state
        """
        # Create cache key from inputs (include use_neural flag in key)
        successors_key = tuple(sorted([(action, ((next_pos[0], next_pos[1]), next_dir)) 
                                     for action, (next_pos, next_dir) in successors]))
        # Convert goal to tuple, handling numpy arrays
        goal_key = tuple(goal) if hasattr(goal, '__iter__') and not isinstance(goal, str) else goal
        cache_key = (pos_state, goal_key, behavior_idx, successors_key, beta, use_neural)
        
        # Check if result is already cached
        if cache_key in self.transition_prob_cache:
            self.cache_hits += 1
            return self.transition_prob_cache[cache_key]
        
        self.cache_misses += 1
        
        # Calculate transition probabilities
        tran_probs = {}
        
        if use_neural and use_neural_when_in_view_only:
            # check if actor agent in view
            if target_visible:
                use_neural = True
            else:
                use_neural = False
        
        if use_neural:
            # Neural predictor version
            formatted_successors = []
            for action, succ in successors:
                next_pos, next_dir = succ
                formatted_successors.append((action, ((next_pos[0], next_pos[1]), next_dir)))
            # Uncomment the following line when neural predictor is available
            tran_probs = self.neuro_predict(self.env, goal, behavior_idx, formatted_successors, pos_state)
            
        
        if not use_neural:
            # Symbolic model - compute in log space to avoid underflow
            log_tran_probs = {}
            for action, succ in successors:
                next_pos, next_dir = succ
                succ_state = ((next_pos[0], next_pos[1]), next_dir)
                if (succ_state, goal) in self.dist_matrix:
                    # Store log probability: log(exp(-beta * (1 + dist))) = -beta * (1 + dist)
                    log_tran_probs[succ_state] = -beta * (1 + self.dist_matrix[(succ_state, goal)])
                else:
                    print("should not happen")
                    input()
                    log_tran_probs[succ_state] = -np.inf
            
            # Normalize in log space using logsumexp
            log_values = list(log_tran_probs.values())
            if log_values:
                log_total = self.logsumexp(log_values)
                # Convert to regular space with normalization
                for succ_state in log_tran_probs:
                    normalized_log_prob = log_tran_probs[succ_state] - log_total
                    # Convert back to regular space for compatibility
                    tran_probs[succ_state] = np.exp(normalized_log_prob) if normalized_log_prob > -700 else 0.0
            else:
                # Fallback: uniform distribution if no valid probabilities
                uniform_prob = 1.0 / len(successors) if successors else 0.0
                for action, succ in successors:
                    next_pos, next_dir = succ
                    succ_state = ((next_pos[0], next_pos[1]), next_dir)
                    tran_probs[succ_state] = uniform_prob
                    
        assert len(tran_probs) == len(successors), "Transition probabilities not computed for all successors"
        # Cache the result
        self.transition_prob_cache[cache_key] = tran_probs
        return tran_probs

    def update_actor_belief_multi_cached(self, actor_belief, goals, target_visible, beta=BETA):
        """
        Cached version of update_actor_belief_multi that uses the transition probability cache.
        
        Parameters:
        actor_belief: {goal: [belief_grid_behavior0, belief_grid_behavior1, ...]}
        goals: List of goal positions
        beta: Temperature parameter
        
        Returns:
        Updated actor_belief dictionary with same structure
        """
        # Initialize new belief grids (log-space needs -inf, regular space needs 0)
        if self.use_log_space:
            new_actor_belief = {goal: [np.full_like(grid, -np.inf) for grid in actor_belief[goal]] for goal in goals}
        else:
            new_actor_belief = {goal: [np.zeros_like(grid) for grid in actor_belief[goal]] for goal in goals}

        for goal in goals:
            for behavior_idx, current_grid in enumerate(actor_belief[goal]):
                # Find non-zero probability cells (handle both log-space and regular space)
                if self.use_log_space:
                    nonzero_cells = np.argwhere(current_grid > -np.inf)
                else:
                    nonzero_cells = np.argwhere(current_grid > 0)
                if nonzero_cells.size == 0:
                    continue
                    
                for cell in nonzero_cells:
                    pos, direction = cell[:2], cell[2]
                    # Convert numpy arrays to hashable types for cache keys
                    pos_state = ((int(pos[0]), int(pos[1])), int(direction))
                    prob = current_grid[tuple(cell)]
                    successors = get_successor(self.env, pos_state)

                    # If at goal, only allow staying
                    if pos[0] == goal[0] and pos[1] == goal[1]:
                        successors = list(filter(lambda x: x[0] == Action.stay, successors))

                    # Get cached transition probabilities (already normalized)
                    tran_probs = self.get_cached_transition_probs(pos_state, goal, behavior_idx, successors, beta, self.use_neural_predictor, self.use_neural_when_in_view_only, target_visible)
                    
                    # Verify transition probabilities sum to ~1 (sanity check)
                    total_prob = sum(tran_probs.values())
                    if total_prob < 1e-10:
                        # All transition probabilities are effectively zero, skip this cell
                        continue
                    
                    if self.use_log_space:
                        # Transition probabilities are already normalized, convert directly to log-space
                        # Update belief for each successor in log-space
                        for action, succ in successors:
                            next_pos, next_dir = succ
                            succ_state = ((next_pos[0], next_pos[1]), next_dir)
                            if succ_state in tran_probs and tran_probs[succ_state] > 0:
                                # Convert normalized transition prob to log-space
                                log_transition_prob = np.log(tran_probs[succ_state])
                                # Add in log-space: log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
                                current_log_belief = new_actor_belief[goal][behavior_idx][next_pos[0], next_pos[1], next_dir]
                                new_log_belief = prob + log_transition_prob
                                new_actor_belief[goal][behavior_idx][next_pos[0], next_pos[1], next_dir] = self.logaddexp(current_log_belief, new_log_belief)
                    else:
                        # Regular space normalization
                        total_prob = sum(float(v) for v in tran_probs.values())
                        if total_prob <= 0:
                            continue
                            
                        # Update belief for each successor
                        for action, succ in successors:
                            next_pos, next_dir = succ
                            succ_state = ((next_pos[0], next_pos[1]), next_dir)
                            transition_prob = float(tran_probs.get(succ_state, 0.0)) / total_prob
                            new_actor_belief[goal][behavior_idx][next_pos[0], next_pos[1], next_dir] += prob * transition_prob

        return new_actor_belief

    def clear_transition_cache(self):
        """Clear the transition probability cache."""
        self.transition_prob_cache.clear()
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring performance."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.transition_prob_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

        
    def compute_action(self, obs, render_and_save=False, get_action=True):
        self.step += 1
        
        # Access observer's observation (agent 0) from the multi-agent observation dict
        observer_obs = obs[0] if isinstance(obs, dict) and 0 in obs else obs
        
        self.pos = observer_obs["observer_pos"]
        self.dir = observer_obs["observer_dir"]
        
        target_visible = "target_pos" in observer_obs or self.pos == self.env.target.pos
        
        self.update_belief(observer_obs) 
        # update the belief based on current observation, each entry is the joint prob P(state, goal, obs history)
        
        self.update_goal_belief() 
        # update the goal belief based on the belief of the observer, each entry is the conditional prob P(goal|obs history)
        # assume goal directed behavior, predict next step belief based on current belief
        self.actor_belief = self.update_actor_belief_multi_cached(self.actor_belief, self.goals, target_visible) 
        # update the actor belief based on the goal belief, each entry is the joint prob P(state, goal, obs history)
        
        # Normalize actor beliefs to ensure probability sums to 1
        self.normalize_actor_beliefs_to_one()
        
        if render_and_save:
            self.render_and_save(f'belief_update_test/actor_belief_step_{self.step}.png', obs)

        # Use greedy action selection instead of MCTS
        if get_action:
            return self.greedy()
        else:
            return None

    def augment_observation(self, obs):
        """
        Augment observation with belief distributions.
        This method calls the environment's augment_obs_with_beliefs method.
        
        Parameters:
        obs: Current observation to augment
        
        Returns:
        Augmented observation with belief distributions
        """
        return self.env.augment_obs_with_beliefs(obs, self.goal_belief, self.actor_belief)




    def greedy(self):
        """
        Simple greedy action selection:
        1. If moving forward reduces distance to target, move forward
        2. Otherwise, turn toward the direction that points to the target
        """
        # Get the most likely actor position across all goals and behaviors
        most_likely_actor_pos = self.get_most_likely_actor_position()
        
        if most_likely_actor_pos is None:
            # If no actor position can be determined, stay in place
            return Action.stay
        
        # Current position and direction
        current_pos = tuple(self.pos)
        current_dir = int(self.dir)
        
        # Calculate current distance to target
        current_distance = abs(current_pos[0] - most_likely_actor_pos[0]) + abs(current_pos[1] - most_likely_actor_pos[1])
        
        # Check if moving forward reduces distance
        # Get forward direction vector
        forward_vec = Direction(current_dir).to_vec()
        forward_pos = (current_pos[0] + forward_vec[0], current_pos[1] + forward_vec[1])
        forward_distance = abs(forward_pos[0] - most_likely_actor_pos[0]) + abs(forward_pos[1] - most_likely_actor_pos[1])
        
        # If moving forward reduces distance, do it
        if forward_distance < current_distance:
            return Action.forward
        
        # Otherwise, determine which direction would be most useful
        # Calculate the direction vector from current position to target
        target_vec = (most_likely_actor_pos[0] - current_pos[0], most_likely_actor_pos[1] - current_pos[1])
        
        # Determine the best direction to face
        best_direction = None
        min_angle_diff = float('inf')
        
        # Check all 4 directions to find the one most aligned with target vector
        for direction in range(4):
            dir_vec = Direction(direction).to_vec()
            
            # Calculate dot product to measure alignment (higher is better)
            dot_product = dir_vec[0] * target_vec[0] + dir_vec[1] * target_vec[1]
            
            # Convert to angle difference (lower is better)
            # We want maximum dot product, so minimum negative dot product
            angle_diff = -dot_product
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_direction = direction
        
        # Determine how to turn to face the best direction
        if best_direction is None or best_direction == current_dir:
            # Already facing the right direction or couldn't determine, stay
            return Action.stay
        
        # Calculate the shortest turn (left or right) to reach best direction
        # Direction values: 0=right, 1=down, 2=left, 3=up
        turn_diff = (best_direction - current_dir) % 4
        
        if turn_diff == 1 or turn_diff == -3:
            # Turn right (clockwise)
            return Action.right
        elif turn_diff == 3 or turn_diff == -1:
            # Turn left (counter-clockwise)  
            return Action.left
        else:
            # 180 degree turn needed, choose left arbitrarily
            return Action.left
    
    def get_most_likely_actor_position(self):
        """
        Find the most likely position of the actor by aggregating probabilities across:
        - All goals (actor_belief already contains joint probabilities)
        - All behavior types 
        - All directions at each position
        Returns the position with the highest aggregated belief mass.
        """
        if not self.actor_belief:
            return None
            
        # Get grid dimensions from any belief grid
        sample_goal = next(iter(self.actor_belief))
        sample_grid = self.actor_belief[sample_goal][0]
        height, width, num_directions = sample_grid.shape
        
        # Create aggregated position belief map
        aggregated_belief = np.zeros((height, width))
        
        # Aggregate across all goals, behavior types, and directions
        for goal in self.goals:
            # Sum across all behavior types for this goal
            for behavior_grid in self.actor_belief[goal]:
                if self.use_log_space:
                    # Convert from log space to regular space, sum across directions
                    regular_space_grid = np.exp(np.clip(behavior_grid, -700, 700))
                    position_belief = np.sum(regular_space_grid, axis=2)  # Sum over direction dimension
                else:
                    # Sum across all directions at each position
                    position_belief = np.sum(behavior_grid, axis=2)  # Sum over direction dimension
                
                # Add to aggregated belief (no goal_weight needed - actor_belief already contains joint probabilities)
                aggregated_belief += position_belief
        
        # Find position with maximum aggregated belief
        if np.max(aggregated_belief) == 0:
            return None
            
        max_pos = np.unravel_index(np.argmax(aggregated_belief), aggregated_belief.shape)
        return max_pos
    


    def render_and_save(self, filename, obs):
        """
        Render the environment and save the visualization.
        
        Parameters:
        filename (str): The name of the file to save the visualization.
        obs (dict): The observation dictionary containing the observer and goal positions.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Get a sample grid shape from the first behavior type of the first goal
        sample_goal = next(iter(self.actor_belief))
        sample_grid = self.actor_belief[sample_goal][0]
        
        if self.use_log_space:
            # Handle log space: convert to regular space first, then sum
            total_belief = np.zeros_like(sample_grid)
            for goal, belief_list in self.actor_belief.items():
                for behavior_grid in belief_list:
                    # Convert from log space to regular space (clipped to avoid overflow)
                    regular_space_grid = np.exp(np.clip(behavior_grid, -700, 700))
                    total_belief += regular_space_grid
            
            # Sum across directions for position beliefs
            belief_sum = np.sum(total_belief, axis=2)
            # Take log for visualization (already in regular space)
            log_belief_sum = np.log(belief_sum + 1e-10)
        else:
            # Regular space: sum directly
            total_belief = np.zeros_like(sample_grid)
            for goal, belief_list in self.actor_belief.items():
                for behavior_grid in belief_list:
                    total_belief += behavior_grid
            
            # Sum across directions for position beliefs
            belief_sum = np.sum(total_belief, axis=2)
            # Take log for visualization
            log_belief_sum = np.log(belief_sum + 1e-10)
        
        vmin = np.min(log_belief_sum)
        vmax = np.max(log_belief_sum)
        
        plt.imshow(log_belief_sum, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar()
        goal_colors = ['yellow', 'green', 'cyan', 'magenta', 'orange']
        goal_probs = [self.goal_belief[goal] for goal in self.goals]
        goal_text = '\n'.join([f'Goal {i+1} ({goal_colors[i % len(goal_colors)]}): {prob:.2f}' for i, prob in enumerate(goal_probs)])
        plt.title(goal_text)

        # Overlay obstacles
        obstacles = np.where(self.env.base_grid != 0)
        plt.scatter(obstacles[1], obstacles[0], c='black', marker='s', label='Obstacle')

        # Overlay observer position
        observer_pos = obs["observer_pos"]
        plt.scatter(observer_pos[1], observer_pos[0], c='blue', marker='o', label='Observer')

        # Overlay goal positions
        
        for i, goal in enumerate(self.goals):
            plt.scatter(goal[1], goal[0], c=goal_colors[i % len(goal_colors)], marker='*', label='Goal')

        # Overlay target position if observed
        if "target_pos" in obs:
            target_pos = obs["target_pos"]
            plt.scatter(target_pos[1], target_pos[0], c='red', marker='x', label='Target')


        plt.savefig(filename)
        plt.close()

    def update_belief(self, obs):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        if "target_pos" in obs or self.pos == self.env.target.pos:
            print(self.step)
            print("in view")
            target_pos = self.env.target.pos
            target_dir = self.env.target.dir # 0-3 denote east south west north respectively
            
            for goal in self.goals:
                # Keep original values at observed position, zero out everything else
                new_list = []
                for behavior_grid in self.actor_belief[goal]:
                    if self.use_log_space:
                        new_grid = np.full_like(behavior_grid, -np.inf)  # Log(0) = -inf for other positions
                        # Keep the original log probability at the observed position
                        new_grid[tuple(target_pos)][target_dir] = behavior_grid[tuple(target_pos)][target_dir]
                    else:
                        new_grid = np.zeros_like(behavior_grid)  # Zero for other positions
                        # Keep the original probability at the observed position
                        new_grid[tuple(target_pos)][target_dir] = behavior_grid[tuple(target_pos)][target_dir]
                    new_list.append(new_grid)
                self.actor_belief[goal] = new_list
            
            # Normalize actor beliefs to maintain proper probability mass after observation
            
                

        else:
            print(self.step)
            print("not in view")
   
            obs_shape = self.agent.observation_space['image'].shape[:-1]
            vis_mask = np.zeros_like(obs_shape, dtype=bool)
            vis_mask = (self.env.gen_obs()[0]['image'][..., 0] != Type.unseen) # 0 denotes the observer
  

            highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)


            # of the agent's view area
            f_vec = self.agent.state.dir.to_vec()
            r_vec = np.array((-f_vec[1], f_vec[0]))
            top_left = (
                self.agent.state.pos
                + f_vec * (self.agent.view_size - 1)
                - r_vec * (self.agent.view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent.view_size):
                for vis_i in range(0, self.agent.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.env.width:
                        continue
                    if abs_j < 0 or abs_j >= self.env.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True
            # highlight_mask = obs['fov']
            for goal in self.goals:
                for cell in np.argwhere(highlight_mask == 1):
                    for behavior_idx in range(len(self.actor_belief[goal])):
                        if self.use_log_space:
                            self.actor_belief[goal][behavior_idx][tuple(cell)] = -np.inf  # Log(0) = -inf
                        else:
                            self.actor_belief[goal][behavior_idx][tuple(cell)] = 0

        self.normalize_actor_beliefs_to_one()
               
               
    def update_goal_belief(self):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        if self.use_log_space:
            # Log-space goal belief update
            for goal in self.goals:
                # Sum across all behavior types and then across state dimensions in log-space
                log_sums = []
                for behavior_grid in self.actor_belief[goal]:
                    # Convert log probabilities to regular space for summing, then back to log
                    grid_sum = np.sum(np.exp(np.clip(behavior_grid, -700, 700)))  # Clip to avoid overflow
                    if grid_sum > 0:
                        log_sums.append(np.log(grid_sum))
                
                if log_sums:
                    self.goal_belief[goal] = np.exp(self.logsumexp(log_sums))
                else:
                    self.goal_belief[goal] = 1e-10  # Small positive value
        else:
            # Regular space goal belief update
            for goal in self.goals:
                # Sum across all behavior types and then across state dimensions
                self.goal_belief[goal] = sum(np.sum(behavior_grid) for behavior_grid in self.actor_belief[goal])

        total = sum(self.goal_belief.values())
        if total == 0:
            # If all goal beliefs are zero, reset to uniform distribution
            print("Warning: All goal beliefs are zero, resetting to uniform distribution")
            for goal in self.goals:
                self.goal_belief[goal] = 1.0 / len(self.goals)
        else:
            # Normalize goal beliefs to sum to 1
            for goal in self.goals:
                self.goal_belief[goal] /= total
            #print(goal,self.goal_belief[goal])




def set_uniform_prob(grid, total_prob = 1.0):
    """
    Set a uniform probability for all free cells in the grid.
    
    Parameters:
    grid (np.array): The grid to be analyzed.
    total_prob (float): Total probability mass to distribute
    
    Returns:
    np.array: A grid with uniform probabilities for all free cells.
    """
    dir = 4
    free_cells = np.argwhere(grid == 0)
    num_free_cells = len(free_cells)
    uniform_prob = total_prob / (num_free_cells * dir) if num_free_cells > 0 else 0

    prob_grid = np.zeros((*grid.shape, dir), dtype=float)
    for cell in free_cells:
        prob_grid[tuple(cell)] = uniform_prob

    return prob_grid


def set_uniform_log_prob(grid, total_prob = 1.0):
    """
    Set a uniform log-probability for all free cells in the grid.
    
    Parameters:
    grid (np.array): The grid to be analyzed.
    total_prob (float): Total probability mass to distribute
    
    Returns:
    np.array: A grid with uniform log-probabilities for all free cells.
    """
    dir = 4
    free_cells = np.argwhere(grid == 0)
    num_free_cells = len(free_cells)
    
    if num_free_cells > 0:
        uniform_prob = total_prob / (num_free_cells * dir)
        uniform_log_prob = np.log(uniform_prob) if uniform_prob > 0 else -np.inf
    else:
        uniform_log_prob = -np.inf

    log_prob_grid = np.full((*grid.shape, dir), -np.inf, dtype=float)
    for cell in free_cells:
        log_prob_grid[tuple(cell)] = uniform_log_prob

    return log_prob_grid









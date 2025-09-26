from .base import BaseAgent
from ..astar import astar, get_successor, execute_action, get_obs_successor, get_reverse_successor
import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction
from multigrid.core.actions import Action
from multigrid.utils.obs import gen_obs_grid_encoding
from multigrid.core.constants import Type
import random
import os
from collections import deque
from copy import deepcopy

# from .neuro_predictor import neuro_predict


# MODES
TRACK = 0
MOVE2GOAL = 1
BETA = 1

# Behavior type indices (extend if more behaviors are needed)
BEHAVIOR_TYPES = [0, 1, 2, 3]

class Observer(BaseAgent):

    def __init__(self, env):

        super().__init__(env.observer)

        self.agent.name = "Observer"
        self.goals = env.goals
        self.goal_costs = None
        self.start = None
        self.prob_dict = None
        self.agent.can_overlap = True
        self.grid = env.base_grid
        self.env = env
        self.start = (env.observer.state.pos,env.observer.state.dir)
        self.goal_belief = { g:1/len(self.goals) for g in self.goals}

        self.step = -1
        self.target_observations = []

        self.infer_goal = None
        self.mode = TRACK

        self.agent.reported_goal = None

    def compute_gr(self, evader_pos, evader_dir,cost):
        current_dis = len(astar(self.start, evader_pos, self.env,cost)) - 1

        probs = []
        for goal in self.goals:
            opt_cost = len(astar(self.start, goal, self.env,cost)) - 1
            real_cost = current_dis + len(astar((evader_pos,evader_dir), goal, self.env,cost)) - 1
            prob = np.exp(-(real_cost - opt_cost))/(1+np.exp(-(real_cost - opt_cost)))
            probs.append(prob)

        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        largest_index = np.argmax(normalized_probs)
        infer_goal = self.goals[largest_index]

        probs = { g:p for g, p in zip(self.goals, normalized_probs)}

        return infer_goal, probs
    
    def compute_target_paths(self, target_pos,target_dir, cost):
        paths = []
        costs = []

        for goal in self.goals:
            path = astar((target_pos,target_dir), goal, self.env,cost)
            costs.append(len(path))

        return paths, costs

    def compute_action(self, obs):

        self.step += 1
        pos = list(obs['observer_pos'])
        dir = np.array(obs['observer_dir'])
        dir_vec = DIR_TO_VEC[dir]

        cost = np.ones((self.grid.shape))


        # STACK OBSERVATIONS
        if 10 in obs["image"]: # Check if the target is in the image
            target_pos = obs["target_pos"]
            target_dir = obs["target_dir"]
            self.infer_goal, self.prob_dict = self.compute_gr(target_pos, target_dir,cost)
            self.goal_belief = self.prob_dict
            target_paths, target_costs = self.compute_target_paths(target_pos, target_dir,cost)

            self.target_observations.append((self.step, target_pos, target_dir, target_paths, target_costs))

            if len(self.target_observations) > 3 and max((self.prob_dict).values())>0.8:
                max_idx = np.argmax((self.prob_dict).values())
                self.agent.reported_goal = self.goals[max_idx]
                
                return Action.done


        path = None
        dir_vec_ = None
        # EXE MODE BEHAVIOUR
        if self.mode == TRACK:
            if len(self.target_observations) > 0:
                last_target_obs = self.target_observations[-1]
                step, target_pos, target_dir, target_paths, target_costs = last_target_obs
                dir_vec_ = DIR_TO_VEC[target_dir]
                path = astar((pos,dir), target_pos, self.env,cost)
            else:
                print("Target: Lost Track")
                return Action.right

        # elif self.mode == MOVE2GOAL:
        #     path = astar((pos,dir), self.infer_goal, self.env,cost)

        
        if len(path)<=1 and dir_vec_ is not None:
            n_dir = len(DIR_TO_VEC)
            dir_vec_curr = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec_curr).all():
                action = Action.right
            else:
                action = Action.left

        elif len(path)<2 or path is None:
            print("Target: Path not processed")
            return Action.right


        else:            
            action = np.array(path[1][0])     
        return action



class BeliefUpdateObserver(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None, use_neural_predictor = False):
        # For Sukai: set use_neural_predictor = True to use neural predictor
        
        super().__init__(env.observer)

        self.env = env
        self.agent.name = "BeliefUpdateObserver"
        self.agent.can_overlap = True

        self.goals = env.goals
        self.step = -1
        self.pos = env.observer.pos
        self.dir = env.observer.dir
        self.use_neural_predictor = use_neural_predictor

        if init_goal_belief:
            self.goal_belief = init_goal_belief
        else:
            self.goal_belief = { g:1/len(self.goals) for g in self.goals}

        # actor_belief now: {goal: [belief_grid_behavior0, belief_grid_behavior1, ...]}
        if init_actor_belief:
            self.actor_belief = init_actor_belief  # assume already in list format per goal
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
    
    def get_cached_transition_probs(self, pos_state, goal, behavior_idx, successors, beta=BETA, use_neural=False):
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
        
        if use_neural:
            # Neural predictor version
            formatted_successors = []
            for action, succ in successors:
                next_pos, next_dir = succ
                formatted_successors.append((action, ((next_pos[0], next_pos[1]), next_dir)))
            # Uncomment the following line when neural predictor is available
            # tran_probs = neuro_predict(self.env, goal, behavior_idx, formatted_successors, pos_state)
            
        
        if not use_neural:
            # Symbolic model
            for action, succ in successors:
                next_pos, next_dir = succ
                succ_state = ((next_pos[0], next_pos[1]), next_dir)
                if (succ_state, goal) in self.dist_matrix:
                    tran_probs[succ_state] = math.exp(-beta * (1 + self.dist_matrix[(succ_state, goal)]))
                else:
                    print("should not happen")
                    input()
                    tran_probs[succ_state] = 0
        
        # Cache the result
        self.transition_prob_cache[cache_key] = tran_probs
        return tran_probs

    def update_actor_belief_multi_cached(self, actor_belief, goals, beta=BETA):
        """
        Cached version of update_actor_belief_multi that uses the transition probability cache.
        
        Parameters:
        actor_belief: {goal: [belief_grid_behavior0, belief_grid_behavior1, ...]}
        goals: List of goal positions
        beta: Temperature parameter
        
        Returns:
        Updated actor_belief dictionary with same structure
        """
        new_actor_belief = {goal: [np.zeros_like(grid) for grid in actor_belief[goal]] for goal in goals}

        for goal in goals:
            for behavior_idx, current_grid in enumerate(actor_belief[goal]):
                # Find non-zero probability cells
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

                    # Get cached transition probabilities
                    tran_probs = self.get_cached_transition_probs(pos_state, goal, behavior_idx, successors, beta, self.use_neural_predictor)
                    
                    # Normalize probabilities
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

        
    def compute_action(self, obs):
        self.step += 1
        self.pos = obs["observer_pos"]
        self.dir = obs["observer_dir"]
        self.update_belief(obs) 
        # update the belief based on current observation, each entry is the joint prob P(state, goal, obs history)
        
        self.update_goal_belief() 
        # update the goal belief based on the belief of the observer, each entry is the conditional prob P(goal|obs history)
        # assume goal directed behavior, predict next step belief based on current belief
        self.actor_belief = self.update_actor_belief_multi_cached(self.actor_belief, self.goals) 
        # update the actor belief based on the goal belief, each entry is the joint prob P(state, goal, obs history)
        self.render_and_save(f'belief_update_test/actor_belief_step_{self.step}.png', obs)

        # Use greedy action selection instead of MCTS
        return self.greedy()

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
        Greedy action selection: move directly towards the most likely position of the actor.
        """
        # Convert numpy arrays to hashable types
        current_pos_state = (tuple(self.pos), int(self.dir))
        
        # Get the most likely actor position across all goals and behaviors
        most_likely_actor_pos = self.get_most_likely_actor_position()
        
        if most_likely_actor_pos is None:
            # If no actor position can be determined, stay in place
            return Action.stay
        
        # Get possible observer actions
        possible_actions = get_obs_successor(self.env, current_pos_state)
        
        if not possible_actions:
            return Action.stay
        
        # Find the action that gets us closest to the most likely actor position
        best_action = None
        best_distance = float('inf')
        
        for action, next_pos_state in possible_actions:
            next_pos = next_pos_state[0]
            # Calculate Manhattan distance to the most likely actor position
            distance = abs(next_pos[0] - most_likely_actor_pos[0]) + abs(next_pos[1] - most_likely_actor_pos[1])
            
            if distance < best_distance:
                best_distance = distance
                best_action = action
        
        return best_action if best_action is not None else Action.stay
    
    def evaluate_action_utility(self, current_pos_state, next_pos_state, action):
        """
        Evaluate the utility of taking a specific action.
        Combines information gain potential and goal-directed movement.
        """
        # Weight factors for different utility components
        info_gain_weight = 1.0
        goal_approach_weight = 0.5
        entropy_reduction_weight = 2.0
        
        # 1. Information gain potential - how much new area will be observed
        info_gain_score = self.calculate_information_gain(next_pos_state)
        
        # 2. Goal approach score - how close we get to likely target locations
        goal_approach_score = self.calculate_goal_approach_score(current_pos_state, next_pos_state)
        
        # 3. Entropy reduction potential - preference for actions that reduce goal uncertainty
        entropy_score = self.calculate_entropy_reduction_potential(next_pos_state)
        
        total_utility = (info_gain_weight * info_gain_score + 
                        goal_approach_weight * goal_approach_score +
                        entropy_reduction_weight * entropy_score)
        
        return total_utility
    
    def calculate_information_gain(self, next_pos_state):
        """
        Calculate potential information gain from observing from next_pos_state.
        Higher scores for positions that can observe areas with high belief mass.
        """
        next_pos, next_dir = next_pos_state
        
        # Simulate the field of view from the next position
        fov_mask = self.get_fov_mask(next_pos, Direction(next_dir))
        
        # Calculate total belief mass in the field of view
        total_observable_belief = 0.0
        for goal in self.goals:
            goal_weight = self.goal_belief[goal]
            for behavior_grid in self.actor_belief[goal]:
                # Sum belief in the field of view area
                for cell in np.argwhere(fov_mask):
                    x, y = cell[0], cell[1]
                    if x < behavior_grid.shape[0] and y < behavior_grid.shape[1]:
                        total_observable_belief += goal_weight * np.sum(behavior_grid[x, y, :])
        
        return total_observable_belief
    
    def calculate_goal_approach_score(self, current_pos_state, next_pos_state):
        """
        Calculate how much closer the next position gets us to likely target locations.
        """
        current_pos = current_pos_state[0]
        next_pos = next_pos_state[0]
        
        approach_score = 0.0
        
        for goal in self.goals:
            goal_prob = self.goal_belief[goal]
            
            # Calculate average target position for this goal weighted by belief
            expected_target_pos = self.get_expected_target_position(goal)
            
            if expected_target_pos is not None:
                # Distance improvement score
                current_dist = np.linalg.norm(np.array(current_pos) - np.array(expected_target_pos))
                next_dist = np.linalg.norm(np.array(next_pos) - np.array(expected_target_pos))
                
                distance_improvement = current_dist - next_dist
                approach_score += goal_prob * distance_improvement
        
        return approach_score
    
    def calculate_entropy_reduction_potential(self, next_pos_state):
        """
        Estimate how much this action could reduce goal belief entropy.
        Prefer actions that have potential to disambiguate between goals.
        """
        current_entropy = compute_entropy(self.goal_belief)
        
        # Simple heuristic: actions that can observe multiple goals simultaneously
        # have higher potential for entropy reduction
        next_pos, next_dir = next_pos_state
        fov_mask = self.get_fov_mask(next_pos, Direction(next_dir))
        
        observable_goals = 0
        for goal in self.goals:
            if fov_mask[goal[0], goal[1]]:
                observable_goals += 1
        
        # Higher score for positions that can observe multiple goals
        entropy_potential = observable_goals * current_entropy
        
        return entropy_potential
    
    def get_most_likely_actor_position(self):
        """
        Find the most likely position of the actor by aggregating probabilities across:
        - All goals (weighted by goal belief)
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
            goal_weight = self.goal_belief[goal]
            
            # Sum across all behavior types for this goal
            for behavior_grid in self.actor_belief[goal]:
                # Sum across all directions at each position
                position_belief = np.sum(behavior_grid, axis=2)  # Sum over direction dimension
                # Add to aggregated belief, weighted by goal probability
                aggregated_belief += goal_weight * position_belief
        
        # Find position with maximum aggregated belief
        if np.max(aggregated_belief) == 0:
            return None
            
        max_pos = np.unravel_index(np.argmax(aggregated_belief), aggregated_belief.shape)
        return max_pos
    
    def get_expected_target_position(self, goal):
        """
        Calculate the expected position of the target for a specific goal,
        weighted by the belief distribution across all behavior types.
        """
        total_belief = 0.0
        weighted_pos = np.array([0.0, 0.0])
        
        for behavior_grid in self.actor_belief[goal]:
            for x in range(behavior_grid.shape[0]):
                for y in range(behavior_grid.shape[1]):
                    for d in range(behavior_grid.shape[2]):
                        belief_mass = behavior_grid[x, y, d]
                        if belief_mass > 0:
                            total_belief += belief_mass
                            weighted_pos += belief_mass * np.array([x, y])
        
        if total_belief > 0:
            return weighted_pos / total_belief
        else:
            return None
    
    def get_fov_mask(self, pos, direction):
        """
        Get the field of view mask for a given position and direction.
        Uses the same logic as the environment's visibility calculation.
        """
        fov_mask = np.zeros((self.env.width, self.env.height), dtype=bool)
        
        # Get field of view parameters
        view_size = self.agent.view_size
        
        # Calculate field of view vectors
        f_vec = direction.to_vec()
        r_vec = np.array((-f_vec[1], f_vec[0]))
        
        # Calculate top-left corner of the field of view
        top_left = np.array(pos) + f_vec * (view_size - 1) - r_vec * (view_size // 2)
        
        # Mark all cells in the field of view
        for vis_j in range(view_size):
            for vis_i in range(view_size):
                # Calculate world coordinates
                world_pos = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                abs_i, abs_j = int(world_pos[0]), int(world_pos[1])
                
                # Check bounds and mark visible
                if 0 <= abs_i < self.env.width and 0 <= abs_j < self.env.height:
                    # Check if cell is not blocked by walls
                    if self.env.base_grid[abs_i, abs_j] == 0:  # 0 means free space
                        fov_mask[abs_i, abs_j] = True
        
        return fov_mask
        
    def mcts(self, iterations = 100, exploration_weight = 1):
        # Convert numpy arrays to hashable types for cache keys
        start_pos_state = (tuple(self.pos), int(self.dir))
        # Keep multi-behavior structure for planning
        start_actor_belief = deepcopy(self.actor_belief)
        start_goal_belief = deepcopy(self.goal_belief)
        root = MCTSNode(self.agent, start_pos_state, start_actor_belief, start_goal_belief, self.env, self.dist_matrix, observer_cache=self)
        
        for _ in range(iterations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(exploration_weight)
            
            if not node.is_terminal():
                node = node.expand()
            result = node.rollout()
            node.backpropagate(result)

        return root.best_child(0).action

        

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
        total_belief = np.zeros_like(sample_grid)
        
        # Sum across all goals and all behavior types
        for goal, belief_list in self.actor_belief.items():
            for behavior_grid in belief_list:
                total_belief += behavior_grid

        belief_sum = np.sum(total_belief, axis=2)
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
                new_list = []
                for behavior_grid in self.actor_belief[goal]:
                    new_grid = np.zeros_like(behavior_grid)
                    new_grid[tuple(target_pos)][target_dir] = behavior_grid[tuple(target_pos)][target_dir]
                    new_list.append(new_grid)
                self.actor_belief[goal] = new_list
                

        else:
            print(self.step)
            print("not in view")
   
            obs_shape = self.agent.observation_space['image'].shape[:-1]
            vis_mask = np.zeros_like(obs_shape, dtype=bool)
            vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer
  

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
                        self.actor_belief[goal][behavior_idx][tuple(cell)] = 0

                
               
               
    def update_goal_belief(self):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        for goal in self.goals:
            # Sum across all behavior types and then across state dimensions
            self.goal_belief[goal] = sum(np.sum(behavior_grid) for behavior_grid in self.actor_belief[goal])

        total = sum(self.goal_belief.values())
        if total == 0:
            print("should not happen")
            print(self.goal_belief)
            for goal in self.goals:
                for behavior_idx, behavior_grid in enumerate(self.actor_belief[goal]):
                    print(f"Goal {goal}, behavior {behavior_idx}:", np.where(behavior_grid > 0))
            input()
        for goal in self.goals:
            self.goal_belief[goal] /= total
            #print(goal,self.goal_belief[goal])



# def update_actor_belief(actor_belief, goals, env, dist_matrix, beta = BETA):
#     new_actor_belief = {goal: np.zeros_like(actor_belief[goal]) for goal in goals}

#     for goal in goals:
#         current_actor_belief = actor_belief[goal]
#         for cell in np.argwhere(current_actor_belief > 0): # select no zero prob
#             pos, dir = cell[:2], cell[2]
#             pos_state = (pos, dir)
#             prob = current_actor_belief[tuple(cell)]
#             successors = get_successor(env, pos_state)

#             tran_probs = {}

#             if pos[0] == goal[0] and pos[1] == goal[1]:
#                 successors = list(filter(lambda x: x[0] == Action.stay, successors))

#             # tran_probs[succ]  = neural_predict(current_state, goal, behaviour_type)
#             # * Dummy case for enumerate Behavior Type
#             # ! make sure run `source env.sh` before running 
#             # ! make sure run this code in ansr-nectar-2 server 
#             BEHAVIOR_TYPE_LIST = [0, 1, 2, 3]
            
#             behaviour_type = np.random.choice(BEHAVIOR_TYPE_LIST, p=[0.25, 0.25, 0.25, 0.25])
            
#             # reformat successor 
#             new_successors = []
#             for action, succ in successors:
#                 next_pos, next_dir = succ
#                 new_successors.append((action, ((next_pos[0], next_pos[1]), next_dir)))
                
#             successors = new_successors
            
#             tran_probs = neuro_predict(env, goal, behaviour_type, successors, pos_state)
            
#             # ! Note that the deep learning model cause the processing time to be long, you may want to shorten size of testing environments. it takes approx 2 mins to run 1 episode. 
            
#             # >>> tran_probs
#             # >>> {'stay': np.float32(8.090865e-09), 'left': np.float32(0.46875), 'forward': np.float32(0.53125), 'right': np.float32(1.1995435e-06)}
            
#             #
#             # --- Comment out to test deep learning model ---
#             # for action, succ in successors:

#             #     next_pos, next_dir = succ

#             #     succ = ((next_pos[0], next_pos[1]), next_dir)

#             #     if (succ, goal) in dist_matrix:
#             #         tran_probs[succ] = math.exp(- beta * (1 + dist_matrix[(succ, goal)]))

                    
#             #     else:
#             #         print("should not happen")
#             #         input()
#             #         tran_probs[succ] = 0
#             # --- End of comment out ---
    
#             total_prob = sum(tran_probs.values())
#             if total_prob > 0:
#                 for succ in tran_probs:
#                     tran_probs[succ] /= total_prob

#             for action, succ in successors:
#                 new_actor_belief[goal][succ[0][0],succ[0][1],succ[1]] += prob*tran_probs[((succ[0][0],succ[0][1]),succ[1])]

 
#     return new_actor_belief

def update_actor_belief_multi(actor_belief, goals, env, dist_matrix, beta=BETA):
    """
    Update actor belief for multiple behavior types.
    
    Parameters:
    actor_belief: {goal: [belief_grid_behavior0, belief_grid_behavior1, ...]}
    goals: List of goal positions
    env: The environment
    dist_matrix: Distance matrix for optimal paths
    beta: Temperature parameter
    
    Returns:
    Updated actor_belief dictionary with same structure
    """
    new_actor_belief = {goal: [np.zeros_like(grid) for grid in actor_belief[goal]] for goal in goals}

    for goal in goals:
        for behavior_idx, current_grid in enumerate(actor_belief[goal]):
            # Find non-zero probability cells
            nonzero_cells = np.argwhere(current_grid > 0)
            if nonzero_cells.size == 0:
                continue
                
            for cell in nonzero_cells:
                pos, direction = cell[:2], cell[2]
                pos_state = (pos, direction)
                prob = current_grid[tuple(cell)]
                successors = get_successor(env, pos_state)

                # If at goal, only allow staying
                if pos[0] == goal[0] and pos[1] == goal[1]:
                    successors = list(filter(lambda x: x[0] == Action.stay, successors))

                # Format successors for neural predictor
                formatted_successors = []
                for action, succ in successors:
                    next_pos, next_dir = succ
                    formatted_successors.append((action, ((next_pos[0], next_pos[1]), next_dir)))

                # # Get transition probabilities from neural predictor
                # # uncomment the following line and symbolic model part to use the neural predictor
                # tran_probs = neuro_predict(env, goal, behavior_idx, formatted_successors, pos_state)

                # symbolic model for testing
                tran_probs = {}
                for action, succ in successors:
                    next_pos, next_dir = succ
                    succ_state = ((next_pos[0], next_pos[1]), next_dir)
                    if (succ_state, goal) in dist_matrix:
                        tran_probs[succ_state] = math.exp(- beta * (1 + dist_matrix[(succ_state, goal)]))
                    else:
                        print("should not happen")
                        input()
                        tran_probs[succ_state] = 0
                
                # Normalize probabilities
                total_prob = sum(float(v) for v in tran_probs.values())
                if total_prob <= 0:
                    continue
                    
                # Update belief for each successor
                for action, succ in formatted_successors:
                    succ_state = succ  # succ is already in format ((x, y), dir)
                    transition_prob = float(tran_probs.get(succ_state, 0.0)) / total_prob
                    new_actor_belief[goal][behavior_idx][succ[0][0], succ[0][1], succ[1]] += prob * transition_prob

    return new_actor_belief


def set_uniform_prob(grid, total_prob = 1):
    """
    Set a uniform probability for all free cells in the grid.
    
    Parameters:
    grid (np.array): The grid to be analyzed.
    
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



class MCTSNode:
    def __init__(self, agent, pos_state, actor_belief, goal_belief, env, dist_matrix, action = None, parent=None, observer_cache=None):
        self.agent = agent
        self.dist_matrix = dist_matrix
        self.pos_state = pos_state  # The current game state
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.actor_belief = {goal: actor_belief[goal] for goal in actor_belief}
        self.goal_belief = {goal: goal_belief[goal] for goal in goal_belief}
        self.env = env
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.value = 0  # Total value of the node
        self.observer_cache = observer_cache  # Reference to observer for caching

    def is_fully_expanded(self):
        return len(self.children) == len(get_obs_successor(self.env, self.pos_state))

    def best_child(self, exploration_weight=1.0):
        """Selects the best child using UCT for decision nodes and expectation for chance nodes."""

        return max(
            self.children, 
            key=lambda child: (child.value / (child.visits + 1e-6)) + 
                              exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
        )

    def expand(self):
        """Expands the node by adding a new child node."""
        tried_moves = {child.action for child in self.children}
        possible_succs = get_obs_successor(self.env, self.pos_state)

        for action, next_pos_state in possible_succs:
            if action not in tried_moves:
                g = self.sample_goal()
 
                # Aggregate belief across all behavior types for sampling
                aggregated_belief = aggregate_actor_belief(self.actor_belief[g])
                actor_pos_state = self.sample_from_3d_belief(aggregated_belief)

                new_actor_belief = self.update_actor_belief_from_obs(actor_pos_state, next_pos_state)

                new_goal_belief = self.update_goal_belief(new_actor_belief)
      
                # goal directed update of the actor belief
                if self.observer_cache:
                    new_actor_belief = self.observer_cache.update_actor_belief_multi_cached(new_actor_belief, self.env.goals)
                else:
                    new_actor_belief = update_actor_belief_multi(new_actor_belief, self.env.goals, self.env, self.dist_matrix)


                new_node = MCTSNode(self.agent, next_pos_state, new_actor_belief, new_goal_belief, self.env, self.dist_matrix, action = action, parent=self, observer_cache=self.observer_cache)
                self.children.append(new_node)
                return new_node

    def sample_goal(self):
        """Samples a goal based on the probability distribution in self.goal_belief."""
        goals = list(self.goal_belief.keys())  # Extract possible goals
        probabilities = np.array(list(self.goal_belief.values()))  # Extract probabilities

        if probabilities.sum() == 0:
            print("should not happen")
            print(self.goal_belief)
            input()
        # Normalize probabilities to ensure they sum to 1
        probabilities /= probabilities.sum()

        # Sample a goal based on the normalized probability distribution
        sampled_goal = np.random.choice(len(goals), p=probabilities)
        return goals[sampled_goal]
            
    def update_goal_belief(self, actor_belief):
        new_goal_belief = {}
        for goal in self.goal_belief:
            new_goal_belief[goal] = np.sum(actor_belief[goal])

        total = sum(new_goal_belief.values())
        for goal in self.goal_belief:
            new_goal_belief[goal] /= total

        return new_goal_belief
            
    def update_actor_belief_from_obs(self, actor_pos_state, observer_pos_state):
        
        # Initialize new actor belief with same structure as current (list of grids per goal)
        new_actor_belief = {goal: [np.zeros_like(grid) for grid in self.actor_belief[goal]] for goal in self.actor_belief}

        actor_pos = actor_pos_state[0], actor_pos_state[1]
        actor_dir = actor_pos_state[2]

        observer_pos = observer_pos_state[0]
        observer_dir = Direction(observer_pos_state[1])
        obs_shape = self.agent.observation_space['image'].shape[:-1]
        vis_mask = np.zeros_like(obs_shape, dtype=bool)
        vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer


        highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)


        # of the agent's view area
        f_vec = observer_dir.to_vec()
        r_vec = np.array((-f_vec[1], f_vec[0]))
        top_left = (
            observer_pos
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
                highlight_mask[abs_i, abs_j] = True # means FoV


        if highlight_mask[actor_pos]: # if the actor is in the observer's view
            for g in self.goal_belief:
                # Update each behavior grid separately
                for behavior_idx in range(len(self.actor_belief[g])):
                    new_actor_belief[g][behavior_idx][actor_pos][actor_dir] = self.actor_belief[g][behavior_idx][actor_pos][actor_dir] 
        else:
            for g in self.goal_belief: # not in observer's view: for each grid in FoV = 0, otherwise use past actor_belief
                for behavior_idx in range(len(self.actor_belief[g])):
                    # Zero out cells in field of view (all direction layers)
                    for cell in np.argwhere(highlight_mask == True):
                        new_actor_belief[g][behavior_idx][cell[0], cell[1], :] = 0

                    # Keep belief for cells not in field of view (all direction layers)
                    for cell in np.argwhere(highlight_mask == False):
                        new_actor_belief[g][behavior_idx][cell[0], cell[1], :] = self.actor_belief[g][behavior_idx][cell[0], cell[1], :]
                

        return new_actor_belief

        
    def is_terminal(self):
        return self.env.is_done()

    def rollout(self):
        """Simulates the game to the end from the current state and returns the result."""
        return -compute_entropy(self.goal_belief)


    def backpropagate(self, result, action_penalty=0):
        """Updates the tree nodes based on the result of the rollout."""
        self.visits += 1
        self.value += result - action_penalty
        if self.parent:
            self.parent.backpropagate(result)

    

    def sample_from_3d_belief(self, actor_belief):
        """Samples a location from the 3D belief map using probability distribution."""
        # depth, height, width = actor_belief.shape  # Get dimensions # ! TODO raise ValueError: not enough values to unpack (expected 3, got 2), it seems that actor_belief shape is 2D array
        if len(actor_belief.shape) == 2:
            height, width = actor_belief.shape
        elif len(actor_belief.shape) == 3:
            depth, height, width = actor_belief.shape
        # Flatten the 3D belief map into a 1D array
        flattened_belief = np.copy(actor_belief).ravel()

        if flattened_belief.sum() == 0:
            print("should not happen")
            print(actor_belief)
            input()
        # Normalize probabilities to ensure they sum to 1
        flattened_belief /= flattened_belief.sum()

        # Sample an index based on the belief distribution
        sampled_index = np.random.choice(len(flattened_belief), p=flattened_belief)

        # Convert the 1D index back to 3D coordinates
        sampled_depth, rem = divmod(sampled_index, height * width)
        sampled_row, sampled_col = divmod(rem, width)

        return sampled_depth, sampled_row, sampled_col  # Return sampled (z, y, x) coordinates

def compute_entropy(goal_belief):
    """Computes the Shannon entropy of the goal belief distribution."""
    probabilities = np.array(list(goal_belief.values()))
    # probabilities = goal_belief
    
    # Ensure the probabilities sum to 1
    probabilities /= probabilities.sum()
    
    # Compute entropy, avoiding log(0) by filtering out zero probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Small offset to avoid log(0)
    
    return entropy


def aggregate_actor_belief(belief_list):
    """
    Aggregate a list of belief grids into a single grid by summing them.
    
    Parameters:
    belief_list: List of belief grids (one per behavior type)
    
    Returns:
    np.array: Aggregated belief grid
    """
    # if belief_list is None or len(belief_list) == 0: # ! TODO if direct call if not belief_list, it will raise ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    #     return None
    aggregated = np.zeros_like(belief_list[0])
    for grid in belief_list:
        aggregated += grid
    return aggregated



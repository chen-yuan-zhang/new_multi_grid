from .base import BaseAgent
from ..new_astar import astar, get_successor, execute_action, get_obs_successor, get_reverse_successor
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

# MODES
TRACK = 0
MOVE2GOAL = 1
BETA = 1

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
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        
        super().__init__(env.observer)

        self.env = env
        self.agent.name = "BeliefUpdateObserver"
        self.agent.can_overlap = True

        self.goals = env.goals
        self.step = -1
        self.pos = env.observer.pos
        self.dir = env.observer.dir

        if init_goal_belief:
            self.goal_belief = init_goal_belief
        else:
            self.goal_belief = { g:1/len(self.goals) for g in self.goals}

        if init_actor_belief:
            self.actor_belief = init_actor_belief
        else:
            self.actor_belief = {g: set_uniform_prob(env.base_grid, self.goal_belief[g]) for g in self.goals}

        self.dist_matrix = self.compute_pairwise_distances()


    def compute_pairwise_distances(self):
        """
        Compute all pairwise distances from each state (position and direction) to the goal locations using BFS.
        """
        #free_cells = np.argwhere(self.env.base_grid == 0)
        free_cells = np.argwhere(self.env.base_grid != 2)
        
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
    

        
    def compute_action(self, obs):
        self.step += 1
        self.pos = obs["observer_pos"]
        self.dir = obs["observer_dir"]
        self.update_belief(obs) 
        # update the belief based on current observation, each entry is the joint prob P(state, goal, obs history)

        self.update_goal_belief() 
        # update the goal belief based on the belief of the observer, each entry is the conditional prob P(goal|obs history)
        # assume goal directed behavior, predict next step belief based on current belief
        self.actor_belief = update_actor_belief(self.actor_belief, self.goals, self.env, self.dist_matrix) 
        # update the actor belief based on the goal belief, each entry is the joint prob P(state, goal, obs history)
        self.render_and_save(f'belief_update_test/actor_belief_step_{self.step}.png', obs)

        #return self.greedy()

        print(self.goal_belief)

        return self.mcts()

    def greedy(self, iterations = 100, exploration_weight = 1):
        start_pos_state = (self.pos, self.dir)
        start_actor_belief = deepcopy(self.actor_belief)
        start_goal_belief = deepcopy(self.goal_belief)
        root = MCTSNode(self.agent, start_pos_state, start_actor_belief, start_goal_belief, self.env, self.dist_matrix)

        children = []
        while not root.is_terminal() and not root.is_fully_expanded():
            children.append(root.expand())
    
        scored_children = []
        for child in children:
            result = child.rollout()
            scored_children.append((result, child))

        best_result, best_child = max(scored_children, key=lambda x: x[0])
        return best_child.action
        
    def mcts(self, iterations = 100, exploration_weight = 1):
        start_pos_state = (self.pos, self.dir)
        start_actor_belief = deepcopy(self.actor_belief)
        start_goal_belief = deepcopy(self.goal_belief)
        root = MCTSNode(self.agent, start_pos_state, start_actor_belief, start_goal_belief, self.env, self.dist_matrix)
        
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

        total_belief = np.zeros_like(next(iter(self.actor_belief.values())))
        for goal, belief in self.actor_belief.items():
            total_belief += belief

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
        obstacles = np.where(self.env.base_grid == 2)
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
                new_actor_belief = np.zeros_like(self.actor_belief[goal])
                new_actor_belief[tuple(target_pos)][target_dir] = self.actor_belief[goal][tuple(target_pos)][target_dir]
                self.actor_belief[goal] = new_actor_belief
                

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
                    self.actor_belief[goal][tuple(cell)] = 0
                #print(self.actor_belief[goal])
               
               
    def update_goal_belief(self):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        for goal in self.goals:
            self.goal_belief[goal] = np.sum(self.actor_belief[goal])

        total = sum(self.goal_belief.values())
        if total == 0:
            print("should not happen,1")
            print(self.goal_belief)
            for goal in self.goals:
                print(np.where(self.actor_belief[goal]>0))
            input()
        for goal in self.goals:
            self.goal_belief[goal] /= total
            #print(goal,self.goal_belief[goal])



def update_actor_belief(actor_belief, goals, env, dist_matrix, beta = BETA):
    new_actor_belief = {goal: np.zeros_like(actor_belief[goal]) for goal in goals}

    for goal in goals:
        current_actor_belief = actor_belief[goal]
        for cell in np.argwhere(current_actor_belief > 0): # select no zero prob
            pos, dir = cell[:2], cell[2]
            pos_state = (pos, dir)
            prob = current_actor_belief[tuple(cell)]
            successors = get_successor(env, pos_state)

            tran_probs = {}

            if pos[0] == goal[0] and pos[1] == goal[1]:
                successors = list(filter(lambda x: x[0] == Action.stay, successors))

            for action, succ in successors:

                next_pos, next_dir = succ

                succ = ((next_pos[0], next_pos[1]), next_dir)

                if (succ, goal) in dist_matrix:
                    tran_probs[succ] = math.exp(- beta * (1 + dist_matrix[(succ, goal)]))

                else:
                    print("should not happen,2")
                    input()
                    tran_probs[succ] = 0

    
            total_prob = sum(tran_probs.values())
            if total_prob > 0:
                for succ in tran_probs:
                    tran_probs[succ] /= total_prob

            for action, succ in successors:
                new_actor_belief[goal][succ[0][0],succ[0][1],succ[1]] += prob*tran_probs[((succ[0][0],succ[0][1]),succ[1])]

 
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
    free_cells = np.argwhere(grid != 2)
    num_free_cells = len(free_cells)
    uniform_prob = total_prob / (num_free_cells * dir) if num_free_cells > 0 else 0

    prob_grid = np.zeros((*grid.shape, dir), dtype=float)
    for cell in free_cells:
        prob_grid[tuple(cell)] = uniform_prob

    return prob_grid



class MCTSNode:
    def __init__(self, agent, pos_state, actor_belief, goal_belief, env, dist_matrix, action = None, parent=None):
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
 
                actor_pos_state = self.sample_from_3d_belief(self.actor_belief[g])

                new_actor_belief = self.update_actor_belief_from_obs(actor_pos_state, next_pos_state)

                new_goal_belief = self.update_goal_belief(new_actor_belief)
      
                # goal directed update of the actor belief
                new_actor_belief = update_actor_belief(new_actor_belief, self.env.goals, self.env, self.dist_matrix)


                new_node = MCTSNode(self.agent, next_pos_state, new_actor_belief, new_goal_belief, self.env, self.dist_matrix, action = action, parent=self)
                self.children.append(new_node)
                return new_node

    def sample_goal(self):
        """Samples a goal based on the probability distribution in self.goal_belief."""
        goals = list(self.goal_belief.keys())  # Extract possible goals
        probabilities = np.array(list(self.goal_belief.values()))  # Extract probabilities

        if probabilities.sum() == 0:
            print("should not happen,4")
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
        
        new_actor_belief = {goal: np.zeros_like(self.actor_belief[goal]) for goal in self.actor_belief}

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
                new_actor_belief[g][actor_pos][actor_dir] = self.actor_belief[g][actor_pos][actor_dir] 
        else:
            for g in self.goal_belief: # not in observer's view: for each grid in FoV = 0, otherwise use past actor_belief
                for cell in np.argwhere(highlight_mask == True):
   
                    new_actor_belief[g][tuple(cell)] = 0

                for cell in np.argwhere(highlight_mask == False):
                    new_actor_belief[g][tuple(cell)] = self.actor_belief[g][tuple(cell)]
                

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
        depth, height, width = actor_belief.shape  # Get dimensions
        # Flatten the 3D belief map into a 1D array
        flattened_belief = np.copy(actor_belief).ravel()

        if flattened_belief.sum() == 0:
            print("should not happen,3")
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
from __future__ import annotations

from typing import Literal
from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction, Type, IDX_TO_COLOR
from multigrid.core.world_object import Goal, Wall

import numpy as np
import random


NEIGHBOURS = (((1,0), (-1, 0), (0, 1), (0, -1)),((-1, 0), (-1, 1), (0, 2), (1, 2), (2, 1), (2, 0), (0, -1), (1, -1)))

def get_neighbours(cell, size, cell_size):
    """
    Get the neighboring cells for a given cell in a grid.
    
    Parameters
    ----------
    cell : tuple[int, int]
        The current cell position
    size : int
        The size of the grid
    cell_size : int
        The size of each cell
        
    Returns
    -------
    list[tuple[int, int]]
        List of neighboring cell positions
    """
    x, y = cell
    neighbors = []
    
    # Check all four directions
    for dx, dy in NEIGHBOURS[cell_size - 1]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < size and 0 <= new_y < size:
            neighbors.append((new_x, new_y))
    
    return neighbors

class AGREnv(MultiGridEnv):
    """
    .. image:: https://i.imgur.com/wY0tT7R.gif
        :width: 200

    ***********
    Description
    ***********

    This environment is an empty room, and the goal for each agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is subtracted
    for the number of steps to reach the goal.

    The standard setting is competitive, where agents race to the goal, and
    only the winner receives a reward.

    This environment is useful with small rooms, to validate that your RL algorithm
    works correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agents starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    *************
    Mission Space
    *************

    "get to the green goal square"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent reaches the goal
    * Timeout (see ``max_steps``)

    """

    def __init__(
        self,
        size: int | None = 8,
        base_grid: np.ndarray | None = None,
        num_goals: int | None = 3,
        enable_hidden_cost: bool = False,
        hidden_cost: np.ndarray | None = None,
        initial_distance: int | None = 3,
        max_steps: int | None = None,
        **kwargs):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        base_grid : np.array, optional
            A pre-defined grid to use as the base grid. If None, a random grid will
            be generated. If this is provided, the `size` parameter will be ignored.
        goals : list[tuple[int, int]], optional
            A list of goal positions in the grid. If None, goals will be generated randomly.
        goal : tuple[int, int], optional
            True goal position to use. If None, a random goal will be selected from the
            list of goals.
        num_goals : int, default=3
            Number of goals to generate in the grid. If `goals` is provided, this
            parameter is ignored.
        enable_hidden_cost : bool, default=False
            Whether to enable hidden costs in the environment. If True, hidden_cost will be used.
            If False, a zero grid will be used and hidden_cost will be ignored.
        hidden_cost : np.array, optional
            A pre-defined hidden cost grid, if enable_hidden_cost is False this parameter will be ignored. 
            If None, a random hidden cost grid will be generated
        initial_distance : int, default=3
            Initial separation distance between the two agents. This is used to ensure that
            the agents start at a reasonable distance from each other.
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent reaches the goal
            or after all agents reach the goal
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """

        if base_grid is not None:
            assert base_grid.shape[0] == base_grid.shape[1], "base_grid must be square"
            size = base_grid.shape[0]
        
        # Ensure size is not None
        if size is None:
            size = 8  # default size

        self.grid_size = size

        self.agents_start_pos = None
        self.agents_start_dir = None
        self.base_grid = base_grid
        self.num_goals = num_goals
        self.initial_distance = initial_distance
        self.enable_hidden_cost = enable_hidden_cost

        if self.enable_hidden_cost:
            if hidden_cost is None:
                self.hidden_cost = np.random.random((size, size))
            else:
                self.hidden_cost = hidden_cost
        else:
            self.hidden_cost = np.zeros((size, size))

        self.goals = []
        self.goal = None

        super().__init__(
            mission_space="predicte the goal of the actor",
            grid_size=size,
            agents=2,
            agent_view_size=[5, 5],
            see_through_walls=[False, False],
            allow_agent_overlap=[True, False],
            max_steps=max_steps or (4 * size**2),
            **kwargs,
        )

        self.mission = self.mission_space.sample()

        self.observer = self.agents[0]
        self.target = self.agents[1]


    # def reset(self, all_reset = True, observer_pos=None, target_pos=None, observer_dir=None, target_dir=None):
    #     """
    #     Reset the environment
    #     """
    #     if all_reset:
    #         obs, info = super().reset()
    #     else:
    #         self.agents_start_dir = [observer_dir, target_dir]
    #         self.agents_start_pos = [observer_pos, target_pos]
    #         obs, info = super().reset()
            
            
    #     obs = self.mod_obs(obs)
        
    #     return obs, info

    def reset(self, reset_grid=True, reset_agents = True):
        """
        Reset the environment
        """
        if reset_grid:
            # Reset the grid and agents
            self.base_grid = None
            self.agents_start_pos = None
            self.agents_start_dir = None
            self.goals = []
            self.goal = None
            

            if self.enable_hidden_cost:
                self.hidden_cost = np.random.random((self.grid_size, self.grid_size))

            
            
        elif reset_agents:
            # Reset only the agents
            self.agents_start_pos = None
            self.agents_start_dir = None
            self.goals = []
            self.goal = None
            
        
        self._gen_grid(self.grid_size, self.grid_size)
        for agent in self.agents:
            agent.state.terminated = False

        self.step_count = 0
        observation = self.gen_obs()
        obs = self.mod_obs(observation)
        # Add initial information of this episode
        infos = {
            'base_grid': self.base_grid,
            'initial_distance': self.initial_distance,
            'enable_hidden_cost': self.enable_hidden_cost,
            'hidden_cost': self.hidden_cost,
            'goals': self.goals,
            'goal': self.goal,
            'agents_start_pos': self.agents_start_pos,
            'agents_start_dir': self.agents_start_dir,
        }
        

        return obs, infos

    def _gen_goals(self, num_goals):
        """
        Generate a list of goal positions in the grid
        """
        

        for i in range(num_goals):
            obj = Goal(IDX_TO_COLOR[i])  # Use the color from the mapping
            pos = self.place_obj(obj)
            self.goals.append(pos)

        # Select a random goal as the true goal
        self.goal = self.goals[np.random.randint(0, len(self.goals))]


    def generate_base_grid(self, size):
        # generate a random square grid with walls and empty spaces

        def unfill(grid, coord, cell_size=1):
            max_coord = grid.shape[0] - 1 # maximum coordinate
            x1 = np.clip(coord[0], 1, max_coord)
            x2 = np.clip(x1+cell_size, 1, max_coord)
            y1 = np.clip(coord[1], 1, max_coord)
            y2 = np.clip(y1+cell_size, 1, max_coord)
            grid[x1:x2, y1:y2] = 0
            return grid

        grid = np.ones((size, size), dtype=int)
        cell_size = 1  # Initialize cell_size

        #Choose 2 random points
        start = tuple(np.random.randint(1, size-1, 2))  # Avoid edges
        grid = unfill(grid, start, 1)
        explored = set()
        queue = [start]

        while len(queue)>0:
            
            # idx = random.randint(0, len(queue)-1)
            idx = min(3, len(queue)-1) if len(queue)>3 else 0
            cell = queue[idx]
            explored.add(cell)
            neighbours = get_neighbours(cell, size, cell_size)
            filled_neighbours = [neighbour for neighbour in neighbours 
                                if grid[neighbour] == 1]

            # The cell doesn't have too many explored neighbours
            if len(filled_neighbours) > 2:
                grid = unfill(grid, cell, cell_size)
                queue += [neighbour for neighbour in filled_neighbours
                        if neighbour not in explored]
                
            queue.pop(idx)
            # Change the cell size randomly
            cell_size = np.random.randint(1, 3) if cell[0]%2==0 and cell[1]%2==0 else 1

        return grid
    
    def position_agents(self, grid, init_sep, size):
        """
        Search for a random position where all the cells are unfilled in a initial_separation distance
        """
        while True:
            # Find all empty cells for target placement
            row, col = np.where(grid == 0)
            if len(row) == 0:  # Safety check - no empty cells
                break
                
            # Randomly select target position from empty cells
            idx = np.random.randint(0, len(row))
            target_x, target_y = row[idx], col[idx]
            
            # Randomly choose one of the four directions for observer placement
            directions = [
                # (observer_pos, target_pos, observer_dir, target_dir)
                ((target_x, target_y + init_sep), (target_x, target_y), Direction.up, np.random.randint(0, 4)),  # Observer north of target
                ((target_x + init_sep, target_y), (target_x, target_y), Direction.left, np.random.randint(0, 4)),   # Observer east of target  
                ((target_x, target_y - init_sep), (target_x, target_y), Direction.down, np.random.randint(0, 4)),     # Observer south of target
                ((target_x - init_sep, target_y), (target_x, target_y), Direction.right, np.random.randint(0, 4))   # Observer west of target
            ]
            
            # Filter valid directions (observer position must be within grid bounds)
            valid_directions = []
            for observer_pos, target_pos, observer_dir, target_dir in directions:
                obs_x, obs_y = observer_pos
                if 1 <= obs_x < size-1 and 1 <= obs_y < size-1:
                    valid_directions.append((observer_pos, target_pos, observer_dir, target_dir))
            
            # If we have valid directions, randomly select one
            if valid_directions:
                return random.choice(valid_directions)

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        # Create an empty grid if not existing
        if self.base_grid is None:
            self.base_grid = self.generate_base_grid(width)
            
        width, height = self.base_grid.shape
        self.grid = Grid(width, height)

        rows, cols = np.where(self.base_grid==1)
        self.grid.state[rows, cols] = Wall()

        # Place the agent
        if self.agents_start_pos is None and self.agents_start_dir is None:
            observer_pos, target_pos, observer_dir, target_dir = self.position_agents(self.base_grid, self.initial_distance, width)
            self.agents_start_pos = [observer_pos, target_pos]
            self.agents_start_dir = [observer_dir, target_dir]

        for i, agent in enumerate(self.agents):
            agent.state.pos = self.agents_start_pos[i]
            agent.state.dir = self.agents_start_dir[i]

        # Generate Goals
        if self.goal is None:
            self._gen_goals(self.num_goals)

    def mod_obs(self, obs):
        obs_observations = obs[0]
        if 10 in obs_observations["image"]: # Check if the target is in the image
            obs_observations["target_pos"] = self.target.state.pos
            obs_observations["target_dir"] = self.target.state.dir

        # Add observer's position and direction to the observations
        obs_observations["observer_pos"] = self.observer.state.pos
        obs_observations["observer_dir"] = self.observer.state.dir

        # Add target's position and direction to the actor's observations
        obs[1]["target_pos"] = self.target.state.pos
        obs[1]["target_dir"] = self.target.state.dir
        
        obs[0] = obs_observations
        return obs

    
    def step(self, actions):
        """
        :meta private:
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        # add observations modification
        observations = self.mod_obs(observations)
        
        return observations, rewards, terminations, truncations, infos
    
    def is_done(self) -> bool:
        """
        Return whether the current episode is finished (for all agents).
        """
        truncated = self.step_count >= self.max_steps
        return truncated or self.target.state.terminated
    
    def on_success(
        self,
        agent,
        rewards,
        terminations):
        """
        Callback for when an agent completes its mission.

        Parameters
        ----------
        agent : Agent
            Agent that completed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        terminations : dict[AgentID, bool]
            Termination dictionary to be updated
        """
        # If the agent is the target, it has completed its mission
        if agent == self.target:
            agent.state.terminated = True # terminate this agent only
            terminations[agent.index] = True



        
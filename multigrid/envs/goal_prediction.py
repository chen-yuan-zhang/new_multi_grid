from __future__ import annotations

from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction
from multigrid.core.world_object import Goal

import numpy as np

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
        goals: list[tuple[int, int]] | None = None,
        goal: tuple[int, int] | None = None,
        num_goals: int | None = 3,
        enable_hidden_cost: bool = False,
        hidden_cost: np.ndarray | None = None,
        initial_distance: int | None = 3,
        max_steps: int | None = None,
        joint_reward: bool = False,
        success_termination_mode: str = 'any',
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

        self.agents_start_pos = None
        self.agents_start_dir = None
        self.base_grid = base_grid
        self.num_goals = num_goals
        self.initial_distance = initial_distance

        if enable_hidden_cost:
            if hidden_cost is None:
                self.hidden_cost = np.random.random((size, size))
            else:
                self.hidden_cost = hidden_cost
        else:
            self.hidden_cost = np.zeros((size, size))

        self.goals = []
        self.goal = None
        if goals is not None:
            self.goals = goals
            self.num_goals = len(goals) # set the number of goals to the length of the provided list

        if goal is not None:
            self.goal = goal
        
            assert self.goal in self.goals, "The goal must be in the list of goals"

        super().__init__(
            mission_space="predicte the goal of the actor",
            grid_size=size,
            agents=2,
            max_steps=max_steps or (4 * size**2),
            see_through_walls=[False, False],
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            **kwargs,
        )

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

    # def _gen_goals(self, num_goals):
    #     """
    #     Generate a list of goal positions in the grid
    #     """
    #     for i in range(num_goals):

    #         obj = Goal(IDX_TO_COLOR[i])
    #         if len(self.goals) == num_goals:
    #             pos = self.goals[i]
    #             self.grid.set(pos[0], pos[1], obj)
    #         else:
    #             pos = self.place_obj(obj)
    #             self.goals.append(pos)

    #         self.POS2COLOR[pos] = str(IDX_TO_COLOR[i]).split(".")[1]  

    #     # goals_costs = get_cost(self.base_grid, self.goals, self.target.pos, self.hidden_cost)
    #     # print(self.goals)
    #     # print(goals_costs)
    #     self.goal = self.goals[np.random.randint(0, len(self.goals)-1)]


    # def generate_base_grid(self, size, cell_size=1):

    #     def unfill(grid, coord, size, cell_size):
    #         x1 = np.clip(coord[0], 1, size-1)
    #         x2 = np.clip(x1+cell_size, 1, size-1)
    #         y1 = np.clip(coord[1], 1, size-1)
    #         y2 = np.clip(y1+cell_size, 1, size-1)
    #         grid[x1:x2, y1:y2] = 0
    #         return grid

    #     grid = np.ones((size, size), dtype=int)

    #     #Choose 2 random points
    #     start = np.random.randint(0, size//cell_size, 2)
    #     grid = unfill(grid, start, size, cell_size)
    #     explored = {tuple(start): 1}
    #     queue = get_neighbours(start, size, cell_size)

    #     while len(queue)>0:
            
    #         # idx = random.randint(0, len(queue)-1)
    #         idx = 3 if len(queue)>3 else 0
    #         cell = queue[idx]
    #         explored[tuple(cell)] = 1
    #         neighbours = get_neighbours(cell, size, cell_size)
    #         filled_neighbours = [neighbour for neighbour in neighbours 
    #                             if grid[tuple(neighbour)] == 1]

    #         # The cell doesn't have 2 explored neighbours
    #         if ((cell_size==1) and (len(filled_neighbours) > 2) or (cell_size==2) and (len(filled_neighbours) > 2)):
    #             # grid[tuple(cell)] = 0
    #             grid = unfill(grid, cell, size, cell_size)
    #             queue += [neighbour for neighbour in filled_neighbours
    #                     if tuple(neighbour) not in explored]
                
    #         queue.pop(idx)
    #         # Change the cell size randomly
    #         cell_size = random.randint(1, 2) if cell[0]%2==0 and cell[1]%2==0 else 1

    #     return grid
    
    # def position_agents(self, grid, init_sep, size):
    #     # Search for a random position where all the cells are unfilled in a initial_separation distance

    #     while True:
    #         #  np.where(grid == 0)
    #         row, col = np.where(grid == 0)
    #         idx = random.randint(0, len(row)-1)
    #         x, y = row[idx], col[idx]

    #         # Check if the col or row is empty
    #         if y+init_sep < size :
    #             return (x, y+init_sep), (x, y), Direction.up, Direction.down 
    #         elif x+init_sep < size:
    #             return (x+init_sep, y), (x, y), Direction.left, Direction.down
    #         elif y-init_sep >= 0 :
    #             return (x, y-init_sep), (x, y), Direction.down, Direction.down
    #         elif x-init_sep >= 0 :
    #             return (x-init_sep, y), (x, y), Direction.right, Direction.down

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        # Create an empty grid
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
            if self.agents_start_pos is not None and self.agents_start_dir is not None:
                agent.state.pos = self.agents_start_pos[i]
                agent.state.dir = self.agents_start_dir[i]

        # Generate Goals
        if self.goal is None:
            self._gen_goals(self.num_goals)

    def mod_obs(self, obs):
        # Pursuer
        mod_observations = {}
        mod_observations = [{"fov": obs[0]["image"], "grid": self.grid.state, 
                             "pos": self.observer.pos, "dir": self.observer.dir}]
        if 10 in obs[0]["image"]:
            mod_observations[0]["target_pos"] = self.agents[1].pos
            mod_observations[0]["target_dir"] = self.agents[1].dir

        # Target
        mod_observations.append({"fov": obs[1]["image"], "grid": self.grid.state, 
                                 "pos": self.target.pos, "dir": self.target.dir})
        return mod_observations
    
    def is_success(self, fwd_obj, agent):
        # if fwd_obj is None:
        #     return True if agent.reported_goal == self.goal else False
        # else:
        #     return fwd_obj.type == Type.goal
        return False
        
    def on_success(self, agent, rewards, terminations):
        super().on_success(agent, rewards, terminations)
        self.mission = f"{agent.name} Success"
        print('\t' + self.mission)

    def on_failure(self, agent, rewards, terminations):
        super().on_success(agent, rewards, terminations)
        self.mission = f"{agent.name} Failure"
        print('\t' + self.mission)

    def is_done(self):
        """
        Check if the episode is done
        """

        # base_done = super().is_done()
        # done = self.target.pos == self.goal or self.agent_states.terminated[0]# self.observer.pos == self.goal

        truncated = self.step_count >= self.max_steps
        done = truncated or self.target.pos == self.goal
        return done
    
    def step(self, actions):
        """
        :meta private:
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        observations = self.mod_obs(observations)
        
        return observations, rewards, terminations, truncations, infos
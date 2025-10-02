from .base import BaseAgent
from ..astar import astar,get_successor


import random
import numpy as np


class AstarTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals

        self.path = None

        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.grid_size, env.grid_size), dtype=np.float32)

    def compute_action(self, obs):
        pos = self.agent.state.pos
        dir = self.agent.state.dir


        if self.path is None:
            self.path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
            self.index = 0


        self.index += 1
        return self.path[self.index][0]

class eGreedyTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals

        self.path = None

        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.grid_size, env.grid_size), dtype=np.float32)

    def compute_action(self, obs, epsilon=0.2):
        pos = self.agent.state.pos
        dir = self.agent.state.dir
        successors = get_successor(self.env,(pos, dir))
        legal_actions = []
        for action, _ in successors:
            legal_actions.append(action)
        
        # ε-greedy
        if random.random() < epsilon:
            self.path = None
            #print("choose random action")
            return random.choice(legal_actions)
            
        if self.path is None:
            #print(self.env.goal)
            #print(self.goal)
            self.path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
            self.index = 0


        self.index += 1
        return self.path[self.index][0]



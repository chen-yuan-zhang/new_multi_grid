from .base import BaseAgent
from ..lock_astar import astar_key,astar_open,get_successor
from ..new_astar import astar

from multigrid.core.actions import Action

import random
import numpy as np



class LockTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals
        self.plans = env.plans
        self.goal_room = env.goal_room

        self.path = None

        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)
    
    def compute_action(self, obs):
        # 当前真实位置/朝向每步更新
        pos = self.agent.state.pos
        dir = self.agent.state.dir
    
        # ——— 初始化一次性的运行时状态 ———
        if not hasattr(self, "_rt_inited") or not getattr(self, "_rt_inited"):
            self.plan_events   = list(self.plans.get(self.goal_room, []))  # 事件序列
            self.plan_idx      = 0             # 当前处理到第几个事件
            self.held_color    = None          # 手里拿的钥匙颜色（字符串）
            self.final_phase   = False         # 是否已经开始“最后到 goal”的阶段
            self.path          = None          # 当前子路径（动作序列）
            self.index         = 0             # 当前子路径内的动作索引
            self._rt_inited    = True
    
        # ——— 如果没有子路径或走完了，就生成下一段子路径 ———
        def _start_next_subpath():
            # 还有事件没消费：生成该事件的子路径
            if (not self.final_phase) and self.plan_idx < len(self.plan_events):
                evt = self.plan_events[self.plan_idx]
                self.plan_idx += 1
                if evt["type"] == "pickup":
                    key_xy = tuple(evt["pos"]["value"])  # 目标钥匙坐标
                    self.path = astar_key((pos, dir), key_xy, self.env, self.hidden_cost)
                    self.index = 1
                    return True
                elif evt["type"] == "open":
                    door_xy = tuple(evt["pos"]["value"])  # 门坐标
                    self.path = astar_open((pos, dir), door_xy, self.env, self.hidden_cost)
                    self.index = 1
                    return True
            # 没有事件了：进入最终阶段，从当前位置直接到 self.goal
            print("Yeah!")
            self.final_phase = True
            self.path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
            self.index = 1
            return True
    
        # 若没有路径或已走完，创建下一段
        if self.path is None or self.index >= len(self.path):
            ok = _start_next_subpath()
            if not ok or self.path is None or len(self.path) == 0:
                # 防御式：没有可走的动作就停一下（按你环境的“无动作/Done”定义替换）
                return self.actions.done if hasattr(self, "actions") and hasattr(self.actions, "done") else 0
    
        # ——— 输出当前子路径的下一个动作，并前进一步 ———
        #print(self.path)
        act = self.path[self.index][0]   # 你的 path 结构是 [(action, ...), ...]
        self.index += 1
        return act

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



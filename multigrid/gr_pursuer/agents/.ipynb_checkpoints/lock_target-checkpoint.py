from .base import BaseAgent
from ..lock_astar import astar_key,astar_open,get_successor#,astar_unified_to_goal
from ..new_astar import astar

from multigrid.core.actions import Action

import random
import numpy as np
from math import ceil
from collections import deque, defaultdict
from typing import Literal,Dict, Tuple, List, Optional, Set, Any
from multigrid.envs.new_locked import _plan_onekey_persist_open,_build_neighbors,_canon,_initial_open_mask
from multigrid.envs.new_locked import _iter_room_keys,_key_position,_edge_position
from copy import deepcopy,copy


class GoalLockTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals
        self.goal_room = env.goal_room
        self.abs = deepcopy(env.abs)


        self.path = None
        self.hallway_col = env.num_cols // 2
        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)

    def compute_action(self, obs):
        pos = self.env.target.state.pos
        dir = self.env.target.state.dir
        held = self.env.target.carrying
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)

        plan = astar_unified_to_goal(
            pos_state=(pos, dir),  # 或 ((x,y),dir,held_color)
            goal_xy=self.goal,
            env=self.env,
            cost=self.hidden_cost,  # 可给代价图
            max_iter=20000,
            agent_idx=1,           # 如果需要读取手中物体
            ActionEnum_=Action # 替换为你自己的 Action
        )
        print(plan)
        return plan[0][0]


class LockTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals
        self.goal_room = env.goal_room
        self.abs = deepcopy(env.abs)


        self.path = None
        self.hallway_col = env.num_cols // 2
        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)

    def compute_action(self, obs):
        pos = self.env.target.state.pos
        dir = self.env.target.state.dir
        held = self.env.target.carrying
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        self.update_door_state()
        if held:
            rid = _rid_from_xy(self.env, pos[0], pos[1], self.hallway_col)
            self.abs["rooms"][rid]["keys"].append({
                                        'color': held.color,
                                        'pos': (pos[0],pos[1])})
        plan_events = _plan_onekey_persist_open(self.abs, start_rid,self.goal_room,goal=self.goal ,held = held)
        #print(plan_events)
        if plan_events != []:
            evt = plan_events[0]
            ety = plan_events[0]["type"] 
            if ety == "pickup":
                kpos = tuple(evt["pos"]["value"])
                path = astar_key((pos, dir), kpos, self.env, self.hidden_cost)
                if path and path[1][0] is not None:
                    return path[1][0]
                else:
                    return Action.left
            elif ety == "open":
                dpos = tuple(evt["pos"]["value"])
                path = astar_open((pos, dir), dpos, self.env, self.hidden_cost)
                if path and path[1][0] is not None:
                    return path[1][0]
                else:
                    return Action.stay
        # --- 如果所有事件都完成，走终点 ---
        path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
        if path and path[1][0] is not None:
            return path[1][0]
        else:
            return Action.left

    def update_door_state(self):
        for room in self.abs["rooms"]:
            self.abs["rooms"][room]["keys"] = []
        for x in range(self.env.width):
            for y in range(self.env.height):
                obj = self.env.grid.get(x,y)
                if obj and obj.type == "key":
                    room = _rid_from_xy(self.env,x,y,self.hallway_col)
                    self.abs["rooms"][room]["keys"].append({'color':obj.color,'pos':(x,y)})
        
        for door_idx in range(len(self.abs["edges"])):
            door = self.abs["edges"][door_idx]
            door["locked"] = False

            obj = self.env.grid.get(*door["pos"])
            if obj:
                door["locked"] = (self.env.grid.get(*door["pos"]).state == "locked" or 
                                  self.env.grid.get(*door["pos"]).state == "closed")
            else:
                door["locked"] = False


class OldLockTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals
        self.hallway_col = env.num_cols // 2
        self.goal_room = env.goal_room
        self.abs = env.abs
        pos = self.env.target.state.pos
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        self.plan_events   = _plan_onekey_persist_open(self.abs, start_rid,self.goal_room,goal=self.goal)
        self.path = None

        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)



    def _key_exists_at(self, pos, color,env):
        x, y = pos
        obj = env.grid.get(x, y)
        # 1) 这个格子上有东西，且是 Key
        if obj is not None and obj.type == "key":
            # 2) 比较颜色（兼容 Enum 和 str）
            obj_color = getattr(obj.color, "value", obj.color)
            target_color = getattr(color, "value", color)
            return str(obj_color) == str(target_color)
        return False
    
    def _door_is_open(self, pos,env):
        x, y = pos
        obj = env.grid.get(x, y)
        if obj is not None and obj.type == "door":
            return obj.is_open
        elif obj is not None:
            return True
        return False
        
    def compute_action(self, obs):
        pos = self.env.target.state.pos
        dir = self.env.target.state.dir
        pos = self.env.target.state.pos
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        if self.plan_events   == []:
            self.plan_events = _plan_onekey_persist_open(self.abs, start_rid,self.goal_room,goal=self.goal)
            
        # 如果没有事件了 → 直接去终点
        while self.index < len(self.plan_events):
            evt = self.plan_events[self.index]
            ety = evt["type"]
            # --- 事件类型判断 ---
            if ety == "pickup":
                kpos = tuple(evt["pos"]["value"])
                # 如果钥匙已经不存在 → 跳过
                if not self._key_exists_at(kpos, evt["key"],self.env):
                    #print("here")
                    self.index += 1
                    continue
                # 否则去拿钥匙
                path = astar_key((pos, dir), kpos, self.env, self.hidden_cost)
                return path[1][0]
    
            elif ety == "open":
                dpos = tuple(evt["pos"]["value"])
                # 如果门已经开了 → 跳过
                if self._door_is_open(dpos,self.env):
                    self.index += 1
                    continue
                # 否则去开门
                path = astar_open((pos, dir), dpos, self.env, self.hidden_cost)
                return path[1][0] 

            else:
                path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
                return path[1][0]
                
        return Action.stay


def _map_rid(c: int, r: int, hallway_col: int | None):
    if hallway_col is not None and c == hallway_col:
        return ("HALL",)   # 统一的走廊节点
    return (c, r)

def _rid_from_xy(env, x, y, hallway_col):
    # 找到 (x,y) 落在哪个原始 room (c,r)
    for r in range(env.num_rows):
        for c in range(env.num_cols):
            room = env.get_room(c, r)
            x0, y0 = room.top
            w,  h  = room.size
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                return _map_rid(c, r, hallway_col)  # 走廊列合并成 ('HALL',)
    # 万一没命中，保守返回 None（调用处做兜底）
    return None


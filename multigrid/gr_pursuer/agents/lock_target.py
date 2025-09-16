from .base import BaseAgent
from ..lock_astar import astar_key,astar_open,get_successor
from ..new_astar import astar

from multigrid.core.actions import Action

import random
import numpy as np
from math import ceil
from collections import deque, defaultdict
from typing import Literal,Dict, Tuple, List, Optional, Set, Any



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


        self.plan_events = list(self.plans.get(self.goal_room, []))  # [{'type': 'pickup'|'open', ...}, ...]

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
        
    def compute_action(self, obs,env):
        pos = self.env.target.state.pos
        dir = self.env.target.state.dir
        print()
    
        # 如果没有事件了 → 直接去终点
        while self.index < len(self.plan_events):
            evt = self.plan_events[self.index]
            ety = evt["type"]
            print(ety,self.index)
            # --- 事件类型判断 ---
            if ety == "pickup":
                kpos = tuple(evt["pos"]["value"])
                # 如果钥匙已经不存在 → 跳过
                if not self._key_exists_at(kpos, evt["key"],env):
                    print("here")
                    self.index += 1
                    continue
                # 否则去拿钥匙
                path = astar_key((pos, dir), kpos, env, self.hidden_cost)
                return path[1][0]
    
            elif ety == "open":
                dpos = tuple(evt["pos"]["value"])
                # 如果门已经开了 → 跳过
                if self._door_is_open(dpos,env):
                    self.index += 1
                    continue
                # 否则去开门
                path = astar_open((pos, dir), dpos, env, self.hidden_cost)
                print("here 11")
                return path[1][0] 
    
        # --- 如果所有事件都完成，走终点 ---
        path = astar((pos, dir), self.goal, env, self.hidden_cost)
        return path[1][0]


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


        
        
        # # 1) locate start/goal rooms (rid)
        # pos = self.agent.state.pos
        # start_rid = xy_to_rid(self.env, *pos)
        # goal_rid  = xy_to_rid(self.env, *tuple(self.goal))
        
        # # 2) build/reuse high-level plan (events)
        # #self._build_or_reuse_plan(start_rid, goal_rid)
        # events = _plan_onekey_persist_open(self.env.abs, start_rid, goal_rid)
        # print(events)

        # if not events:
        #     path = astar((pos, self.agent.state.dir), self.goal, self.env, self.hidden_cost)
        #     return path[1][0]
    
        # evt = events[0]
        # ety = evt.get("type")
        # exy = tuple(evt.get("pos", {}).get("value")) if evt.get("pos") else None
    
        # # --- 根据事件类型执行 ---
        # if ety == "pickup":
        #     # 路径规划到钥匙
        #     path = astar_key((pos, self.agent.state.dir), exy, self.env, self.hidden_cost)
        #     return path[1][0]
    
        # elif ety == "open":
        #     path = astar_open((pos, self.agent.state.dir), exy, self.env, self.hidden_cost)
        #     return path[1][0]

        # else:
        #     return 0
        
        # # -------- one-time runtime init --------
        # if not hasattr(self, "_rt_inited") or not getattr(self, "_rt_inited"):
        #     self.plan_events = list(self.plans.get(self.goal_room, []))  # [{'type': 'pickup'|'open', ...}, ...]
        #     self.plan_idx    = 0
        #     self.final_phase = False
        #     self.held_color  = None
        #     self._rt_inited  = True
    
        # # -------- helpers (MiniGrid/MultiGrid-style) --------
        # def _tile_at(xy):
        #     grid = getattr(self.env, "grid", None)
        #     if grid is None or xy is None:
        #         return None
        #     x, y = int(xy[0]), int(xy[1])
        #     try:
        #         return grid.get(x, y)
        #     except Exception:
        #         return None
    
        # # def _is_key_tile(tile, color=None):
        # #     if tile is None:
        # #         return False
        # #     return (getattr(tile, "type", None) == "key" and
        # #             (color is None or getattr(tile, "color", None) == color))
    
        # def _door_is_open(xy):
        #     t = _tile_at(xy)
        #     return (getattr(t, "type", None) == "door") and bool(getattr(t, "is_open", False))
            
        # def _pickup_color_if_on_key():
        #     """Update held_color by checking what the agent is actually carrying."""
        #     carried = getattr(getattr(self.agent, "state", None), "_carried_obj", None)
        
        #     # unwrap numpy array case: array(None) -> None
        #     if isinstance(carried, np.ndarray):
        #         try:
        #             carried = carried.item()
        #         except Exception:
        #             carried = None
        
        #     if carried is None:
        #         self.held_color = None
        #     else:
        #         # assume carried has attribute 'color' (like Key object in MiniGrid)
        #         self.held_color = getattr(carried, "color", None)
    
        # # # update held key by observation of current tile
        # _pickup_color_if_on_key()
    
        # # # -------- online event skipping (back-to-front) --------
        # # if (not self.final_phase) and self.plan_idx < len(self.plan_events):
        # #     # new_idx 至少是当前 plan_idx；遇到已开的门就推进到该门的后一个事件
        # #     new_idx = self.plan_idx
        # #     j = len(self.plan_events) - 1
        # #     while j >= self.plan_idx:
        # #         evtj = self.plan_events[j]
        # #         if evtj.get("type") == "open":
        # #             exyj = tuple(evtj.get("pos", {}).get("value")) if evtj.get("pos") else None
        # #             if exyj is not None and _door_is_open(exyj):
        # #                 # 跳到“门后面的下一个事件”，并保留更大的推进结果
        # #                 new_idx = max(new_idx, j + 1)
        # #         j -= 1
        # #     # 统一推进（一次性跳过一串“已开门”和其前导的 pickup）
        # #     self.plan_idx = new_idx
    
        # # -------- decide target now (fresh planning each step) --------
        # path = None
        # #print("plan_idx: ",self.plan_idx)
        # if (not self.final_phase) and self.plan_idx < len(self.plan_events):
        #     evt = self.plan_events[self.plan_idx]
        #     ety = evt.get("type")
        #     exy = tuple(evt.get("pos", {}).get("value")) if evt.get("pos") else None
        #     print(exy)
    
        #     if ety == "pickup" and exy is not None:
        #         target_color = evt.get("key") or evt.get("color")
            
        #         # 1) 若手上已经有对应颜色的钥匙 → 跳过
        #         if target_color is not None and self.held_color == target_color:
        #             self.plan_idx += 1
        #             path = None
            
        #         else:
        #             # 2) 检查目标格子是否仍然是该颜色的钥匙
        #             t = _tile_at(exy)
        #             is_key_here = (
        #                 t is not None
        #                 and getattr(t, "type", None) == "key"
        #                 and (target_color is None or getattr(t, "color", None) == target_color)
        #             )
            
        #             if not is_key_here:
        #                 self.plan_idx += 1
        #                 path = None
        #             else:
        #                 # 3) 仍然存在目标钥匙 → 规划去捡
        #                 path = astar_key((pos, dir), exy, self.env, self.hidden_cost,
        #                                  agent_idx=0, version=True)
    
        #     elif ety == "open" and exy is not None:
        #         # if door turned open since last check, consume and fall back to final-phase/next event
        #         if _door_is_open(exy):
        #             self.plan_idx += 1
        #             path = None
        #         else:
        #             path = astar_open((pos, dir), exy, self.env, self.hidden_cost)
        #     # consume the event only when its precondition becomes true on arrival,
        #     # which is naturally handled by the next-step skip loop.
        # else:
        #     # no more events: final phase -> go straight to goal
        #     self.final_phase = True
        #     path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
    
        # # -------- safe fallback --------
        # if not path or len(path) == 0:
        #     return Action.stay
    
        # # -------- execute ONLY ONE action; replan next step --------
        # #print(path)
        # act = path[1][0]  # path like [(action, ...), ...]
        # return act

    
    # def compute_action(self, obs):
    #     # 当前真实位置/朝向每步更新
    #     pos = self.agent.state.pos
    #     dir = self.agent.state.dir
    
    #     # ——— 初始化一次性的运行时状态 ———
    #     if not hasattr(self, "_rt_inited") or not getattr(self, "_rt_inited"):
    #         self.plan_events   = list(self.plans.get(self.goal_room, []))  # 事件序列
    #         self.plan_idx      = 0             # 当前处理到第几个事件
    #         self.held_color    = None          # 手里拿的钥匙颜色（字符串）
    #         self.final_phase   = False         # 是否已经开始“最后到 goal”的阶段
    #         self.path          = None          # 当前子路径（动作序列）
    #         self.index         = 0             # 当前子路径内的动作索引
    #         self._rt_inited    = True
    #     print(self.plan_events)
    
    #     # ——— 如果没有子路径或走完了，就生成下一段子路径 ———
    #     def _start_next_subpath():
    #         # 还有事件没消费：生成该事件的子路径
    #         if (not self.final_phase) and self.plan_idx < len(self.plan_events):
    #             evt = self.plan_events[self.plan_idx]
    #             self.plan_idx += 1
    #             if evt["type"] == "pickup":
    #                 key_xy = tuple(evt["pos"]["value"])  # 目标钥匙坐标
    #                 self.path = astar_key((pos, dir), key_xy, self.env, self.hidden_cost)
    #                 self.index = 1
    #                 return True
    #             elif evt["type"] == "open":
    #                 door_xy = tuple(evt["pos"]["value"])  # 门坐标
    #                 self.path = astar_open((pos, dir), door_xy, self.env, self.hidden_cost)
    #                 self.index = 1
    #                 return True
    #         # 没有事件了：进入最终阶段，从当前位置直接到 self.goal
    #         print("Yeah!")
    #         self.final_phase = True
    #         self.path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
    #         self.index = 1
    #         return True
    
    #     # 若没有路径或已走完，创建下一段
    #     if self.path is None or self.index >= len(self.path):
    #         ok = _start_next_subpath()
    #         if not ok or self.path is None or len(self.path) == 0:
    #             # 防御式：没有可走的动作就停一下（按你环境的“无动作/Done”定义替换）
    #             return self.actions.done if hasattr(self, "actions") and hasattr(self.actions, "done") else 0
    
    #     # ——— 输出当前子路径的下一个动作，并前进一步 ———
    #     #print(self.path)
    #     act = self.path[self.index][0]   # 你的 path 结构是 [(action, ...), ...]
    #     self.index += 1
    #     return act



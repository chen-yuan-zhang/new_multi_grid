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
        self.abs = env.abs


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

        for room in self.abs["rooms"]:
            self.abs["rooms"][room]["keys"] = []
        for x in range(self.env.width):
            for y in range(self.env.height):
                obj = self.env.grid.get(x,y)
                if obj and obj.type == "key":
                    room = _rid_from_xy(self.env,x,y,self.hallway_col)
                    self.abs["rooms"][room]["keys"].append({'color':obj.color,'pos':(x,y)})
                   
        for door in self.abs["edges"]:
            obj = self.env.grid.get(*door["pos"])
            if obj:
                door["locked"] = (self.env.grid.get(*door["pos"]).state == "locked")
            else:
                door["locked"] = False

        plan_events = _plan_onekey_persist_open(self.abs, start_rid,self.goal_room, held = held)
        if plan_events != []:
            evt = plan_events[0]
            ety = plan_events[0]["type"] 
            if ety == "pickup":
                kpos = tuple(evt["pos"]["value"])
                path = astar_key((pos, dir), kpos, self.env, self.hidden_cost)
                return path[1][0]
    
            elif ety == "open":
                dpos = tuple(evt["pos"]["value"])
                path = astar_open((pos, dir), dpos, self.env, self.hidden_cost)
                return path[1][0] 
    
        # --- 如果所有事件都完成，走终点 ---
        path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
        return path[1][0]

class OldLockTarget(BaseAgent):

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
        
    def compute_action(self, obs):
        pos = self.env.target.state.pos
        dir = self.env.target.state.dir
    
        # 如果没有事件了 → 直接去终点
        while self.index < len(self.plan_events):
            evt = self.plan_events[self.index]
            ety = evt["type"]
            #print(ety,self.index)
            # --- 事件类型判断 ---
            if ety == "pickup":
                kpos = tuple(evt["pos"]["value"])
                # 如果钥匙已经不存在 → 跳过
                if not self._key_exists_at(kpos, evt["key"],env):
                    #print("here")
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
                #print("here 11")
                return path[1][0] 
    
        # --- 如果所有事件都完成，走终点 ---
        path = astar((pos, dir), self.goal, env, self.hidden_cost)
        return path[1][0]




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



# 邻接：adj[r] = [(nbr, color, eid)] 
def _build_neighbors(abs_graph) -> Dict[Any, List[Tuple[Any, Any, int]]]: 
    adj = defaultdict(list) 
    for eid, e in enumerate(abs_graph["edges"]): 
        u, v = e["u"], e["v"] 
        col = e.get("color", None) 
        adj[u].append((v, col, eid)) 
        adj[v].append((u, col, eid)) 
    return adj


# —— 颜色统一（Enum/字符串都支持）—— #
def _canon(c):
    if c is None: return None
    return c.name if hasattr(c, "name") else str(c)

# —— 取钥匙位置（容错：可能没有pos）—— #
def _key_position(k):
    # new format: {"color":..., "pos":(x,y)}
    if isinstance(k, dict) and "pos" in k:
        return ("xy", tuple(k["pos"]))
    # fallback: 无位置信息
    return (None, None)

# —— 统一遍历房间内钥匙（返回列表[(color, kobj)]，kobj可为原字典或颜色字符串）—— #
def _iter_room_keys(room_dict):
    keys_raw = list(room_dict.get("keys", []))
    out = []
    for k in keys_raw:
        if isinstance(k, dict):
            color = _canon(k.get("color", None))
            out.append((color, k))
        else:
            # legacy plain color
            out.append((_canon(k), k))
    return out

# —— 帮助：安全取门位置信息 —— #
def _edge_position(abs_graph, eid):
    e = abs_graph["edges"][eid]
    # 优先使用显式坐标/位置
    for key in ("pos", "door_xy", "xy", "at"):
        if key in e:
            return {"type": "xy", "value": e[key]}
    # 退化：用房间端点描述
    return {"type": "rooms", "value": (e.get("u"), e.get("v"))}

# —— 初始已开门mask（无色门或 locked=False 视为已开）—— #
def _initial_open_mask(abs_graph) -> int:
    #print(abs_graph)
    mask = 0
    for eid, e in enumerate(abs_graph["edges"]):
        col = e.get("color", None)
        locked = e.get("locked", True)
        if _canon(col) is None or not locked:
            mask |= (1 << eid)
    return mask


# —— 单目标：单钥匙 + 持久开门 —— #
def _plan_onekey_persist_open(abs_graph, start_rid, goal_rid, held = None,open_mask = None, ):
    """
    返回：events（按时间顺序）
    events 元素两类：
      1) {'type':'pickup', 'room': rid, 'key': 'red', 'pos': {'type':'xy','value':(x,y)}|{'type':None,'value':None}}
      2) {'type':'open',   'eid': eid, 'color':'red',
          'from': u, 'to': v, 'pos': {...}}
    若不可达：返回 None
    """
    rooms = abs_graph["rooms"]
    adj   = _build_neighbors(abs_graph)
    if held is not None and hasattr(held, "color"):
        held = held.color

    if open_mask is None:
        open_mask = _initial_open_mask(abs_graph)

    start_state = (start_rid, held, open_mask)
    q = deque([start_state])

    prev  = {start_state: None}
    prev_evt: Dict[Tuple[Any, Optional[Any], int], Dict[str, Any]] = {}

    while q:
        rid, held, open_mask = q.popleft()
        if rid == goal_rid:
            # 回溯事件
            path_events: List[Dict[str, Any]] = []
            cur = (rid, held, open_mask)
            while prev[cur] is not None:
                evt = prev_evt.get(cur)
                if evt:
                    path_events.append(evt)
                cur = prev[cur]
            path_events.reverse()
            return path_events

        # ---- 先移动（可能开门） ----
        for nb, col, eid in adj[rid]:
            ccol = _canon(col)
            opened = (open_mask >> eid) & 1

            if opened or ccol is None or ccol == _canon(held):
                next_open = open_mask
                evt = None
                if (not opened) and (ccol is not None) and (ccol == _canon(held)):
                    next_open |= (1 << eid)
                    pos_info = _edge_position(abs_graph, eid)
                    evt = {
                        "type":  "open",
                        "eid":   eid,
                        "color": ccol,
                        "from":  rid,
                        "to":    nb,
                        "pos":   pos_info
                    }

                ns = (nb, held, next_open)
                if ns not in prev:
                    prev[ns] = (rid, held, open_mask)
                    prev_evt[ns] = evt
                    q.append(ns)

        # ---- 在当前房间拿/换钥匙（零代价扩展） ----
        room_keys = _iter_room_keys(rooms[rid])
        if room_keys:
            if held is None:
                for kcolor, kraw in room_keys:
                    ns = (rid, kcolor, open_mask)
                    if ns not in prev:
                        prev[ns] = (rid, held, open_mask)
                        kpos_type, kpos_val = _key_position(kraw)
                        prev_evt[ns] = {
                            "type": "pickup",
                            "room": rid,
                            "key":  kcolor,
                            "pos":  {"type": kpos_type, "value": kpos_val}
                        }
                        q.append(ns)
            else:
                for kcolor, kraw in room_keys:
                    if kcolor != _canon(held):
                        ns = (rid, kcolor, open_mask)
                        if ns not in prev:
                            prev[ns] = (rid, held, open_mask)
                            kpos_type, kpos_val = _key_position(kraw)
                            prev_evt[ns] = {
                                "type": "pickup",
                                "room": rid,
                                "key":  kcolor,  # 换到的新钥匙
                                "pos":  {"type": kpos_type, "value": kpos_val}
                            }
                            q.append(ns)

    return []
from __future__ import annotations

from typing import Literal,Dict, Tuple, List, Optional, Set, Any
from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction, Type, IDX_TO_COLOR,Color
from multigrid.core.world_object import Goal, Wall, Door, Key,WorldObj
from multigrid.core.actions import Action
from multigrid.core.mission import MissionSpace
from multigrid.core.roomgrid import Room, RoomGrid

import numpy as np
import random
from math import ceil
from collections import deque, defaultdict
import heapq
from itertools import count




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

def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def _heuristic(abs_graph: Dict, rid: Any, goal_xy: Optional[Tuple[int,int]]) -> int:
    """
    启发式函数：直接使用当前房间的 pos['value'] 与目标坐标的曼哈顿距离。
    """
    if goal_xy is None:
        return 0

    # 从 abs_graph 中找到当前房间的坐标
    rooms = abs_graph["rooms"]
    rinfo = rooms.get(rid, {})

    # 取 pos.value 作为房间代表坐标
    if "pos" in rinfo and isinstance(rinfo["pos"], dict) and "value" in rinfo["pos"]:
        room_xy = rinfo["pos"]["value"]
    else:
        # 若没有 pos.value，则启发式退化为 0
        return 0

    gx, gy = goal_xy
    rx, ry = room_xy
    return abs(rx - gx) + abs(ry - gy)

# def _plan_onekey_persist_open(
#     abs_graph,
#     start_rid,
#     goal_rid,
#     held=None,
#     open_mask=None,
#     goal=None,
#     max_expansions=500000
# ):
#     """
#     返回：events（按时间顺序）
#     events 元素两类：
#       1) {'type':'pickup', 'room': rid, 'key': 'red',
#           'pos': {'type':'xy','value':(x,y)}|{'type':None,'value':None}}
#       2) {'type':'open',   'eid': eid, 'color':'red',
#           'from': u, 'to': v, 'pos': {...}}
#     若不可达：返回 None
#     """
#     rooms = abs_graph["rooms"]
#     adj = _build_neighbors(abs_graph)

#     if held is not None and hasattr(held, "color"):
#         held = held.color

#     if open_mask is None:
#         open_mask = _initial_open_mask(abs_graph)

#     start_state = (start_rid, held, open_mask)
#     q = deque([start_state])

#     prev = {start_state: None}
#     prev_evt: Dict[Tuple[Any, Optional[Any], int], Dict[str, Any]] = {}
#     expansions = 0

#     while q:
#         rid, held, open_mask = q.popleft()
#         expansions += 1

#         # 扩展上限保护
#         if (max_expansions is not None) and (expansions > max_expansions):
#             return []  # 触发扩展上限，放弃

#         # 命中目标
#         if rid == goal_rid:
#             # 回溯事件路径
#             path_events: List[Dict[str, Any]] = []
#             cur = (rid, held, open_mask)
#             while prev[cur] is not None:
#                 evt = prev_evt.get(cur)
#                 if evt:
#                     path_events.append(evt)
#                 cur = prev[cur]
#             path_events.reverse()

#             if goal is not None:
#                 path_events.append({
#                     "type": "move",
#                     "room": goal_rid,
#                     "pos": {"type": "xy", "value": goal}
#                 })

#             return path_events

#         # ---- 先移动（可能开门） ----
#         for nb, col, eid in adj[rid]:
#             ccol = _canon(col)
#             opened = (open_mask >> eid) & 1

#             if opened or ccol is None or ccol == _canon(held):
#                 next_open = open_mask
#                 evt = None

#                 # 若此门未开但颜色匹配当前钥匙 → 开门事件
#                 if (not opened) and (ccol is not None) and (ccol == _canon(held)):
#                     next_open |= (1 << eid)
#                     pos_info = _edge_position(abs_graph, eid)
#                     evt = {
#                         "type": "open",
#                         "eid": eid,
#                         "color": ccol,
#                         "from": rid,
#                         "to": nb,
#                         "pos": pos_info
#                     }

#                 ns = (nb, held, next_open)
#                 if ns not in prev:
#                     prev[ns] = (rid, held, open_mask)
#                     prev_evt[ns] = evt
#                     q.append(ns)

#         # ---- 在当前房间拿/换钥匙（零代价扩展） ----
#         room_keys = _iter_room_keys(rooms[rid])
#         if room_keys:
#             if held is None:
#                 for kcolor, kraw in room_keys:
#                     ns = (rid, kcolor, open_mask)
#                     if ns not in prev:
#                         prev[ns] = (rid, held, open_mask)
#                         kpos_type, kpos_val = _key_position(kraw)
#                         prev_evt[ns] = {
#                             "type": "pickup",
#                             "room": rid,
#                             "key": kcolor,
#                             "pos": {"type": kpos_type, "value": kpos_val}
#                         }
#                         q.append(ns)
#             else:
#                 for kcolor, kraw in room_keys:
#                     if kcolor != _canon(held):
#                         ns = (rid, kcolor, open_mask)
#                         if ns not in prev:
#                             prev[ns] = (rid, held, open_mask)
#                             kpos_type, kpos_val = _key_position(kraw)
#                             prev_evt[ns] = {
#                                 "type": "pickup",
#                                 "room": rid,
#                                 "key": kcolor,  # 换到的新钥匙
#                                 "pos": {"type": kpos_type, "value": kpos_val}
#                             }
#                             q.append(ns)

#     return []

def _plan_onekey_persist_open(abs_graph, start_rid, goal_rid,
                              held=None, open_mask=None, goal=None,
                              max_expansions=50000, pickup_cost=0):
    """
    返回：events（按时间顺序）
      pickup: {'type':'pickup','room':rid,'key':color,'pos':{...}}
      open:   {'type':'open','eid':eid,'color':color,'from':u,'to':v,'pos':{...}}
      move:   {'type':'move','room':goal_rid,'pos':{'type':'xy','value':goal}}
    不可达：返回 []
    """
    rooms = abs_graph["rooms"]
    adj   = _build_neighbors(abs_graph)

    # 规范 held
    if held is not None and hasattr(held, "color"):
        held = held.color


    if open_mask is None:
        open_mask = _initial_open_mask(abs_graph)

    start_state = (start_rid, held, open_mask)

    # PQ 元素结构：(f, g, h, tie, rid, held, open_mask)
    pq = []
    tie_counter = count()

    g_cost: Dict[Tuple[Any, Optional[str], int], float] = {start_state: 0.0}
    prev: Dict[Tuple[Any, Optional[str], int], Optional[Tuple[Any, Optional[str], int]]] = {start_state: None}
    prev_evt: Dict[Tuple[Any, Optional[str], int], Optional[Dict[str, Any]]] = {}

    h0 = _heuristic(abs_graph, start_rid, goal)  # 必须是数字
    heapq.heappush(pq, (h0 + 0, 0, h0, next(tie_counter), start_rid, held, open_mask))

    expansions = 0
    while pq:
        f, g, h, _, rid, held, open_mask = heapq.heappop(pq)
        state = (rid, held, open_mask)

        # 过期条目跳过
        if g_cost.get(state, float("inf")) < g:
            continue

        expansions += 1
        if (max_expansions is not None) and (expansions > max_expansions):
            return []

        # 命中目标
        if rid == goal_rid:
            path_events: List[Dict[str, Any]] = []
            cur = state
            while prev[cur] is not None:
                evt = prev_evt.get(cur)
                if evt:
                    path_events.append(evt)
                cur = prev[cur]
            path_events.reverse()
            if goal is not None:
                path_events.append({"type":"move","room":goal_rid,"pos":{"type":"xy","value":goal}})
            return path_events

        # ---- 1) 跨房间移动（代价 1）----
        for nb, col, eid in adj[rid]:
            ccol = _canon(col)
            opened = (open_mask >> eid) & 1
            if opened or ccol is None or ccol == _canon(held):
                next_open = open_mask
                evt = None
                if (not opened) and (ccol is not None) and (ccol == _canon(held)):
                    next_open |= (1 << eid)
                    pos_info = _edge_position(abs_graph, eid)
                    evt = {"type":"open","eid":eid,"color":ccol,"from":rid,"to":nb,"pos":pos_info}

                ns = (nb, held, next_open)
                new_g = g + 1
                if new_g < g_cost.get(ns, float("inf")):
                    g_cost[ns] = new_g
                    prev[ns] = state
                    prev_evt[ns] = evt
                    nh = _heuristic(abs_graph, nb, goal)
                    heapq.heappush(pq, (new_g + nh, new_g, nh, next(tie_counter), nb, held, next_open))
        # ---- 2) 拾取 / 换钥匙（代价 pickup_cost）----
        room_keys = _iter_room_keys(rooms[rid])
        if room_keys:
            if held is None:
                for kcolor, kraw in room_keys:
                    ns = (rid, kcolor, open_mask)
                    new_g = g + pickup_cost
                    if new_g < g_cost.get(ns, float("inf")):
                        g_cost[ns] = new_g
                        prev[ns] = state
                        kpos_type, kpos_val = _key_position(kraw)
                        prev_evt[ns] = {
                            "type":"pickup","room":rid,"key":kcolor,
                            "pos":{"type":kpos_type,"value":kpos_val}
                        }
                        nh = _heuristic(abs_graph, rid, goal)
                        heapq.heappush(pq, (new_g + nh, new_g, nh, next(tie_counter), rid, kcolor, open_mask))
            else:
                for kcolor, kraw in room_keys:
                    if kcolor != _canon(held):
                        ns = (rid, kcolor, open_mask)
                        new_g = g + pickup_cost
                        if new_g < g_cost.get(ns, float("inf")):
                            g_cost[ns] = new_g
                            prev[ns] = state
                            kpos_type, kpos_val = _key_position(kraw)
                            prev_evt[ns] = {
                                "type":"pickup","room":rid,"key":kcolor,
                                "pos":{"type":kpos_type,"value":kpos_val}
                            }
                            nh = _heuristic(abs_graph, rid, goal)
                            heapq.heappush(pq, (new_g + nh, new_g, nh, next(tie_counter), rid, kcolor, open_mask))

    # 未找到路径
    return []


# —— 批量：从走廊到所有房间（返回事件序列）—— #
def plan_all_from_hall(abs_graph, hall_id=("HALL",)):
    plans = {}
    for rid in abs_graph["rooms"].keys():
        if rid == hall_id:
            continue
        plans[rid] = _plan_onekey_persist_open(abs_graph, hall_id, rid)
    return plans

# —— 打印：打印“捡钥匙 / 开门”的关键事件（含坐标）—— #
def print_plan(target_rid, events):
    print(f"\nTarget room: {target_rid}")
    if not events:
        print("No feasible plan.")
        return
    for i, evt in enumerate(events):
        if evt["type"] == "pickup":
            p = evt.get("pos", {"type": None, "value": None})
            if p["type"] == "xy":
                pstr = f"xy={p['value']}"
            else:
                pstr = "xy=None"
            print(f"{i:02d}. PICKUP   @room={evt['room']}   key={evt['key']}   pos[{pstr}]")
        elif evt["type"] == "open":
            pos = evt["pos"]
            if pos["type"] == "xy":
                pstr = f"xy={pos['value']}"
            else:
                u, v = pos["value"]
                pstr = f"rooms=({u},{v})"
            print(f"{i:02d}. OPEN     door[eid={evt['eid']}] color={evt['color']}  {evt['from']}->{evt['to']}  pos[{pstr}]")



class AGRlocked(RoomGrid):
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
        size: int = 8,
        num_rooms: int = 6,
        room_size: int = 5,
        max_hallway_keys: int = 1,
        max_keys_per_room: int = 2,
        num_rows: int = 2,
        num_cols: int = 3,

        
        base_grid: np.ndarray | None = None,
        base_rooms: np.ndarray | None = None,
        num_goals: int | None = 3,
        goals: list[tuple[int, int]] | None = None,
        goal: tuple[int, int] | None = None,
        agents_start_pos: list[tuple[int, int]] | None = None,
        agents_start_dir: list[int] | None = None,  
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
            #assert base_grid.shape[0] == base_grid.shape[1], "base_grid must be square"
            size = base_grid.shape[0]

        assert room_size >= 4
        assert num_rooms % 2 == 0

        self.num_rooms = num_rooms
        self.max_hallway_keys = max_hallway_keys
        self.max_keys_per_room = max_keys_per_room

        if max_steps is None:
            max_steps = 8 * num_rooms * room_size**2

        self.agents_start_pos = agents_start_pos
        self.agents_start_dir = agents_start_dir
        self.base_grid = base_grid
        self.base_rooms = base_rooms
        self.num_goals = num_goals
        self.initial_distance = initial_distance
        self.enable_hidden_cost = enable_hidden_cost

        if self.enable_hidden_cost:
            if hidden_cost is None:
                self.hidden_cost = np.random.random((size, size))
            else:
                self.hidden_cost = hidden_cost
        else:
            self.hidden_cost = np.ones((size, size))

        if goals is None:
            self.goals = []
        else:
            self.goals = goals
        self.goal = goal
        
        height = (room_size - 1) * num_cols + 1
        width = (room_size - 1) * num_rows + 1
        self.height = height
        self.width = width
        
        super().__init__(
            mission_space="",
            agents=2,
            agent_view_size=[5, 5],
            see_through_walls=[False, False],
            allow_agent_overlap=[True, False],
            room_size=room_size,
            num_rows= num_rows,
            num_cols= num_cols,
            max_steps=max_steps,
            **kwargs,
        )

        self.mission = self.mission_space.sample()
        self.observer = self.agents[0]
        self.target = self.agents[1]


    def reset(self, seed = None, **kwargs):
        """
        Reset the environment
        """              
        if self.base_grid is not None:
            self.load_from_base_grid(self.base_grid,self.base_rooms)
        else:
            self._gen_grid(self.width,self.height)
        for agent in self.agents:
            agent.state.terminated = False
        self.step_count = 0
        observation = self.gen_obs()
        obs = self.mod_obs(observation)
        # Add initial information of this episode
        infos = {
            'base_grid': self.base_grid,
            'base_rooms': self.base_rooms,
            'initial_distance': self.initial_distance,
            'enable_hidden_cost': self.enable_hidden_cost,
            'hidden_cost': self.hidden_cost,
            'goals': self.goals,
            'goal': self.goal,
            'agents_start_pos': self.agents_start_pos,
            'agents_start_dir': self.agents_start_dir,
        }

        if self.render_mode == 'human':
            self.render()
        return obs, infos

    def _gen_goals(self, num_goals):
        """
        Generate a list of goal positions in the grid.
        Restrict goal placement to rooms (LEFT and RIGHT columns), not the central hallway.
        """
        # Collect all non-hallway rooms: columns 0 (LEFT) and 2 (RIGHT)
        HALL = getattr(self, "hallway_col", self.num_cols // 2)
        room_list = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if c == HALL:
                    continue
                room = self.get_room(c, r)
                if room is not None:
                    room_list.append(room)

        random.shuffle(room_list)
        if self.goal is None:
            self.goals = []
            for i in range(num_goals):
                room = room_list[i % len(room_list)]
    
                obj = Goal(IDX_TO_COLOR[i])
                pos = self.place_obj(obj, top=room.top, size=room.size)  # confined to room area
                self.goals.append(pos)
    
            self.goal = self.goals[np.random.randint(0, len(self.goals))]
    
        else:
            for i in range(num_goals):
                pos = self.goals[i]
                self.grid.set(pos[0], pos[1], Goal(IDX_TO_COLOR[i]))

    def _map_rid(self, c: int, r: int, hallway_col: int | None):
        if hallway_col is not None and c == hallway_col:
            return ("HALL",)   # 统一的走廊节点
        return (c, r)

    
    def base_gen_grid(self, width, height):
        min_size = getattr(self, "min_room_size", 4)
        max_size = getattr(self, "max_room_size", 6)
    
        # 1) 随机每列宽度 & 每行高度
        col_widths = [self._rand_int(min_size, max_size + 1) for _ in range(self.num_cols)]
        row_heights = [self._rand_int(min_size, max_size + 1) for _ in range(self.num_rows)]
    
        # 2) 计算整图所需宽高（共享墙：每个房间步长为 size-1，最后+1 收尾）
        total_width  = sum(w - 1 for w in col_widths) + 1
        total_height = sum(h - 1 for h in row_heights) + 1
    
        # 若传入的 width/height 与计算出的不一致，这里直接以计算结果为准重建 Grid
        # （也可以选择断言或裁剪，这里选择重建以确保布局正确）
        self.width, self.height = total_width, total_height
        self.grid = Grid(self.width, self.height)
    
        # 3) 预计算每列、每行的左上偏移（使用 size-1 累加）
        x_offsets = [0] * self.num_cols
        y_offsets = [0] * self.num_rows
        for c in range(1, self.num_cols):
            x_offsets[c] = x_offsets[c - 1] + (col_widths[c - 1] - 1)
        for r in range(1, self.num_rows):
            y_offsets[r] = y_offsets[r - 1] + (row_heights[r - 1] - 1)
    
        # 4) 创建房间并画墙
        self.room_grid = [[None] * self.num_cols for _ in range(self.num_rows)]
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                top  = (x_offsets[c], y_offsets[r])
                size = (col_widths[c], row_heights[r])
                room = Room(top, size)
                self.room_grid[r][c] = room
                self.grid.wall_rect(*room.top, *room.size)
    
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                room = self.room_grid[r][c]
                if c < self.num_cols - 1:
                    room.neighbors[Direction.right] = self.room_grid[r][c + 1]
                if r < self.num_rows - 1:
                    room.neighbors[Direction.down]  = self.room_grid[r + 1][c]
                if c > 0:
                    room.neighbors[Direction.left]  = self.room_grid[r][c - 1]
                if r > 0:
                    room.neighbors[Direction.up]    = self.room_grid[r - 1][c]

    def _abs_init(self, hallway_col):
        """Init abstract graph with hallway column merged into a single node."""
        self.abs = {
            "rooms": {},              # rid -> {"keys": [ {"color":..., "pos":(x,y)}, ... ], "doors": [eid,...]}
            "edges": [],              # [{"u","v","color","locked","pos":(x,y)}]
            "adj": defaultdict(list)  # rid -> [edge_id,...]
        }
        # collect unique rid after hallway mapping
        unique_nodes = set()
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                rid = self._map_rid(c, r, hallway_col)
                unique_nodes.add(rid)
        for rid in unique_nodes:
            self.abs["rooms"][rid] = {"keys": [], "doors": []}
    
    def _abs_add_edge(self, u, v, color, locked: bool, pos):
        """Register a door as an abstract edge; store its grid position 'pos'."""
        eid = len(self.abs["edges"])
        self.abs["edges"].append({
            "u": u, "v": v, "color": color, "locked": bool(locked), "pos": tuple(pos)
        })
        self.abs["adj"][u].append(eid)
        self.abs["adj"][v].append(eid)
        self.abs["rooms"][u]["doors"].append(eid)
        self.abs["rooms"][v]["doors"].append(eid)
        return eid
    
    def _abs_add_key(self, rid, color, pos):
        """Register a key in room rid with its color and grid position."""
        self.abs["rooms"][rid]["keys"].append({
            "color": color,
            "pos": tuple(pos)
        })
    
    def export_grid_numpy(self) -> np.ndarray:
        """
        Export current grid to numpy array in the official format:
          array[x, y, :] = (type_idx, color_idx, state)
    
        Where:
          - type_idx: Type enum index
          - color_idx: Color enum index (0 if not applicable)
          - state: integer encoding (e.g., 0=default, 1=closed, 2=locked, etc.)
        """
        W, H = self.width, self.height
        arr = np.zeros((W, H, WorldObj.dim), dtype=np.int8)
    
        type_to_idx = {t: i for i, t in enumerate(Type)}
        color_to_idx = {c: i for i, c in enumerate(Color)}  # 假设你有 Color enum
    
        for x in range(W):
            for y in range(H):
                obj = self.grid.get(x, y)
                if obj is None:
                    arr[x, y, WorldObj.TYPE] = type_to_idx[Type.empty]
                    arr[x, y, WorldObj.COLOR] = 0
                    arr[x, y, WorldObj.STATE] = 0
                    continue
    
                # type
                t = getattr(obj, "type", None)
                if t is None:
                    t = Type.unseen
                arr[x, y, WorldObj.TYPE] = type_to_idx[t]
    
                # color
                if hasattr(obj, "color") and obj.color is not None:
                    arr[x, y, WorldObj.COLOR] = color_to_idx[obj.color]
                else:
                    arr[x, y, WorldObj.COLOR] = 0
    
                # state
                s = 0
                if hasattr(obj, "state"):
                    if isinstance(obj.state, str):
                        if obj.state == "open":
                            s = 0
                        elif obj.state == "closed":
                            s = 1
                        elif obj.state == "locked":
                            s = 2
                    else:
                        s = int(obj.state)
                arr[x, y, WorldObj.STATE] = s
    
        return arr
    
    # --- helpers ---------------------------------------------------------------

    
    def _is_door(self, obj) -> bool:
        if obj is None: return False
        t = getattr(obj, "type", None)
        if t is not None:
            return (t == Type.door) or (getattr(t, "value", None) == "door")
        return "door" in obj.__class__.__name__.lower()
    
    def _wall_cells(self, c, r, dir):
        """返回房间 (c,r) 指定边上整条墙的格子坐标（含角点）"""
        room = self.get_room(c, r)
        x0, y0 = room.top
        w,  h  = room.size
        if dir == Direction.right:
            x = x0 + w - 1
            return [(x, y) for y in range(y0, y0 + h)]
        if dir == Direction.left:
            x = x0
            return [(x, y) for y in range(y0, y0 + h)]
        if dir == Direction.down:
            y = y0 + h - 1
            return [(x, y) for x in range(x0, x0 + w)]
        if dir == Direction.up:
            y = y0
            return [(x, y) for x in range(x0, x0 + w)]
        return []
    
    def _canon_color(self, c):
        return c.name if hasattr(c, "name") else (None if c is None else str(c))
    
    def _door_xy(self, c, r, dir, expect_color=None):
        """
        放完门再调用：沿 (c,r) 的该边扫描，找真正的 Door；若没找到，再到相邻房间的对侧边扫。
        expect_color 可选；传入可避免误命中其它门。
        """
        want = self._canon_color(expect_color)
    
        # 1) 扫当前房间这侧的墙
        for x, y in self._wall_cells(c, r, dir):
            obj = self.grid.get(x, y)
            if self._is_door(obj):
                if want is None or self._canon_color(getattr(obj, "color", None)) == want:
                    return (x, y)
    
        # 2) 扫相邻房间的对侧墙（双层墙的情况）
        dc = (dir == Direction.right) - (dir == Direction.left)
        dr = (dir == Direction.down)  - (dir == Direction.up)
        nc, nr = c + dc, r + dr
        opp = {Direction.right: Direction.left,
               Direction.left:  Direction.right,
               Direction.down:  Direction.up,
               Direction.up:    Direction.down}[dir]
        if 0 <= nc < self.num_cols and 0 <= nr < self.num_rows:
            for x, y in self._wall_cells(nc, nr, opp):
                obj = self.grid.get(x, y)
                if self._is_door(obj):
                    if want is None or self._canon_color(getattr(obj, "color", None)) == want:
                        return (x, y)
    
        # 3) 仍没找到：抛错或给出 fallback（建议先抛错以便调试）
        raise RuntimeError(f"Door not found on wall: room=({c},{r}) dir={dir} color={want}")
        
    # === Helper: 用坐标反查它属于哪个 room，再映射到合并后的 rid ===
    def _rid_from_xy(self, x, y, hallway_col):
        # 找到 (x,y) 落在哪个原始 room (c,r)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                room = self.get_room(c, r)
                x0, y0 = room.top
                w,  h  = room.size
                if x0 <= x < x0 + w and y0 <= y < y0 + h:
                    return self._map_rid(c, r, hallway_col)  # 走廊列合并成 ('HALL',)
        # 万一没命中，保守返回 None（调用处做兜底）
        return None
    # --- main grid generation --------------------------------------------------
    
    def _gen_grid(self, width, height):
        # 0) base grid & hallway column
        print(" 0) base grid & hallway column")
        self.base_gen_grid(width, height)
        HALLWAY_COL = getattr(self, "hallway_col", self.num_cols // 2)
        self.hallway_col = HALLWAY_COL
        self._abs_init(HALLWAY_COL)

        # keep hallway vertical openings
        if HALLWAY_COL is not None:
            for r in range(self.num_rows - 1):
                self.remove_wall(HALLWAY_COL, r, Direction.down)
    
        # 1) Prepare door colors
        print("# 1) Prepare door colors")
        approx_doors = (self.num_rows * (self.num_cols - 1)) + ((self.num_rows - 1) * self.num_cols)
        color_seq = list(Color)
        color_seq = (color_seq * ((approx_doors // len(color_seq)) + 1))[:approx_doors]
        color_seq = self._rand_perm(color_seq)
    
        # 2) Place doors & register edges (with positions) on abstract graph
        print("# 2) Place doors & register edges (with positions) on abstract graph")
        used_colors = []
        ci = 0
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                # right door: (c,r) <-> (c+1,r)
                if c < self.num_cols - 1:
                    colr = color_seq[ci]; ci += 1
                    # compute door position BEFORE/AFTER placing (deterministic anyway)
                    self.add_door(c, r, dir=Direction.right, color=colr, locked=True, rand_pos=False)
                    dpos = self._door_xy(c, r, Direction.right)
                    used_colors.append(colr)
                    u = self._map_rid(c, r, HALLWAY_COL)
                    v = self._map_rid(c + 1, r, HALLWAY_COL)
                    self._abs_add_edge(u, v, colr, locked=True, pos=dpos)
    
                # down door: (c,r) <-> (c,r+1) but skip hallway column
                if r < self.num_rows - 1 and c != HALLWAY_COL:
                    colr = color_seq[ci]; ci += 1
                    self.add_door(c, r, dir=Direction.down, color=colr, locked=True, rand_pos=False)
                    dpos = self._door_xy(c, r, Direction.down)
                    used_colors.append(colr)
                    u = self._map_rid(c, r, HALLWAY_COL)
                    v = self._map_rid(c, r + 1, HALLWAY_COL)
                    self._abs_add_edge(u, v, colr, locked=True, pos=dpos)
                    
        used_colors = list(set(used_colors)) #+ list(set(used_colors))
        # 3) Place keys (hallway first, then rooms) & register on abstract graph with positions
        print("# 3) Place keys (hallway first, then rooms) & register on abstract graph with positions")
        rooms_flat_ids = [(c, r) for r in range(self.num_rows) for c in range(self.num_cols)]
        random.shuffle(rooms_flat_ids)
        if HALLWAY_COL is not None and used_colors:
            max_hall_keys = getattr(self, "max_hallway_keys", 3)
            num_hall_keys = min(self._rand_int(1, max_hall_keys + 1), 3)
            for k in range(num_hall_keys):
                hall_row  = self._rand_int(0, self.num_rows)
                hall_room = self.get_room(HALLWAY_COL, hall_row)
                key_color = used_colors[k]
                key_obj   = Key(color=key_color)
                self.place_obj(key_obj, top=hall_room.top, size=hall_room.size)
                kpos = getattr(key_obj, "init_pos", getattr(key_obj, "cur_pos", None))
                if kpos is None:
                    continue  # 极端情况下跳过或记录 warning
                rid_xy = self._rid_from_xy(kpos[0], kpos[1], HALLWAY_COL)
                self._abs_add_key(rid_xy, key_color, kpos)
        
        # 3) remaining keys —— 同样用坐标反查 rid
        for key_color in used_colors[num_hall_keys:]:
            c, r   = self._rand_elem(rooms_flat_ids)
            room   = self.get_room(c, r)
            key_obj = Key(color=key_color)
            self.place_obj(key_obj, top=room.top, size=room.size)
            kpos = getattr(key_obj, "init_pos", getattr(key_obj, "cur_pos", None))
            if kpos is None:
                continue
            rid_xy = self._rid_from_xy(kpos[0], kpos[1], HALLWAY_COL)
            self._abs_add_key(rid_xy, key_color, kpos)

        # 4) Snapshot base grid
        print("# 4) Snapshot base grid")
        self.base_grid = self.export_grid_numpy()
        # 2) 保存房间信息
        self.base_rooms = [
            [(room.top, room.size) for room in row]
            for row in self.room_grid]
        # 3) 保存 hallway 列
        self.base_hallway_col = self.hallway_col

        print("# 5) Place two agents: one random room; the other in hallway (if exists) else another random room")
        
        # 5) Place two agents: one random room; the other in hallway (if exists) else another random room
        rooms_flat = [self.get_room(c, r) for r in range(self.num_rows) for c in range(self.num_cols)]
        room_for_agent = self._rand_elem(rooms_flat)
        # MultiGridEnv.place_agent(self, self.agents[0], top=room_for_agent.top, size=room_for_agent.size)
    
        if HALLWAY_COL is not None:
            hall_top = self.get_room(HALLWAY_COL, 0).top
            hall_size = (self.get_room(HALLWAY_COL, 0).size[0], self.height)
            MultiGridEnv.place_agent(self, self.agents[1], top=hall_top, size=hall_size)
            MultiGridEnv.place_agent(self, self.agents[0], top=hall_top, size=hall_size)

        self.agents_start_pos = [self.agents[0].state.pos,self.agents[1].state.pos]
        self.agents_start_dir = [int(self.agents[0].state.dir),int(self.agents[1].state.dir)]
            
        #self.place_agent_near(self.agents[0],init_step = self.agents[1].state.pos,max_dist = self.initial_distance)
        # else:
        #     another_room = self._rand_elem(rooms_flat)
        #     MultiGridEnv.place_agent(self, self.agents[1], top=another_room.top, size=another_room.size)
    
        # 6) Goals
        self._gen_goals(self.num_goals)
        room = self._rid_from_xy(self.goal[0], self.goal[1], HALLWAY_COL)
        self.goal_room = room



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

    def place_agent_near(
        self,
        agent: Agent,
        init_step: tuple[int, int],
        max_dist: int = 3,
        rand_dir: bool = True,
        max_tries: int = 100,
    ) -> tuple[int, int]:
        agent.state.pos = (-1, -1)
        h, w = self.grid.height, self.grid.width
        for _ in range(max_tries):
            r = self._rand_int(0, h)
            c = self._rand_int(0, w)
    
            if abs(c - init_step[0]) + abs(r - init_step[1]) <= max_dist:
                if self.grid.is_valid_pos(c, r) and self.grid.get(c, r) is None:
                    pos = (c, r)
                    agent.state.pos = pos
                    if rand_dir:
                        agent.state.dir = self._rand_int(0, 4)
                    return pos
    
        raise RuntimeError(f"Could not place agent near {init_step} within {max_dist} after {max_tries} tries")

    
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
        if agent == self.target and agent.state.pos == self.goal:
            agent.state.terminated = True # terminate this agent only
            terminations[agent.index] = True

    def load_from_base_grid(self, base_grid, base_rooms, *, place_agents=True, place_goals=True):
        """
        从 base_grid（#4 时刻的快照）重建地图与抽象图：
          1) 重建 self.grid（先尝试 Grid.decode，失败则手动逐格 decode）
          2) 重建抽象图：rooms/edges/keys（含 door.locked 状态）
          3) 可选：重新放置 agents 与 goals
        """
        # 1) Decode 成 grid
        W, H = base_grid.shape[:2]
        self.width = W
        self.height = H
        self.grid, vis_mask = Grid.decode(base_grid)
        print("[Loader] Decoded by Grid.decode()")
    
        # 2) 重新初始化房间 grid 和抽象图
        HALLWAY_COL = getattr(self, "hallway_col", self.num_cols // 2)
        self.hallway_col = HALLWAY_COL
        self.restore_room_grid_from_tuples(base_rooms)
        
        # ---------- 3) 重建抽象图 ----------
        # 说明：你之前在 _gen_grid() 里调用了 self._abs_init(HALLWAY_COL)
        # 此处从快照恢复，需要重新 init + 扫描门与钥匙把 edges/keys 注回去
        self._abs_init(HALLWAY_COL)
    
        # 3.1 清空各房间的 key 列表
        for room in self.abs["rooms"]:
            self.abs["rooms"][room]["keys"] = []
        self.abs["edges"] = []
        # 3.2 全图扫描：登记钥匙位置
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj and getattr(obj, "type", None) == "key":
                    rid_xy = self._rid_from_xy(x, y, HALLWAY_COL)
                    self._abs_add_key(rid_xy, obj.color, (x, y))

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                # right door: (c,r) <-> (c+1,r)
                if c < self.num_cols - 1:
                    u = self._map_rid(c, r, HALLWAY_COL)
                    dpos = self._door_xy(c, r, Direction.right)
                    obj = self.grid.get(*dpos)
                    v = self._map_rid(c + 1, r, HALLWAY_COL)
                    self._abs_add_edge(u, v, obj.color, obj.is_locked, pos=dpos)
                # down door: (c,r) <-> (c,r+1) but skip hallway column
                if r < self.num_rows - 1 and c != HALLWAY_COL:
                    dpos = self._door_xy(c, r, Direction.down)
                    obj = self.grid.get(*dpos)
                    u = self._map_rid(c, r, HALLWAY_COL)
                    v = self._map_rid(c, r + 1, HALLWAY_COL)
                    self._abs_add_edge(u, v, obj.color, obj.is_locked, pos=dpos)

        self.agents[1].state.pos = self.agents_start_pos[1]
        self.agents[0].state.pos = self.agents_start_pos[0]

        self.agents[1].state.dir = self.agents_start_dir[1]
        self.agents[0].state.dir = self.agents_start_dir[0]
        
        # ---------- 5) 可选：重新生成/对齐 goals ----------
        print("[Loader] Re-generating or aligning goals")
        room = self._rid_from_xy(self.goal[0], self.goal[1], HALLWAY_COL)
        self._gen_goals(len(self.goals))
        self.goal_room = room
        print("[Loader] Done.")

    def restore_room_grid_from_tuples(self, base_rooms, *, set_neighbors=True):
        """
        用保存的 base_rooms = [[(top,size), ...], [...]] 恢复 room_grid。
        注意：不绘制墙，墙/门/物体已经由 Grid.decode(base_grid) 恢复到 self.grid 里了。
        """
        print("[Loader] Restoring room_grid from tuples")
    
        # 1) 维度
        num_rows = len(base_rooms)
        assert num_rows > 0, "base_rooms is empty"
        num_cols = len(base_rooms[0])
        for row in base_rooms:
            assert len(row) == num_cols, "base_rooms rows have inconsistent lengths"
    
        # 如果你的环境里需要显式维护 num_rows/num_cols，这里一并对齐
        self.num_rows = num_rows
        self.num_cols = num_cols
    
        # 2) 实例化 Room
        self.room_grid = [[None] * num_cols for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                (top, size) = base_rooms[r][c]
                # top/size 可能是 list，转成 tuple 更稳
                top  = tuple(top)
                size = tuple(size)
                room = Room(top, size)
                self.room_grid[r][c] = room
    
        # 3) （可选）补邻接
        if set_neighbors:
            for r in range(num_rows):
                for c in range(num_cols):
                    room = self.room_grid[r][c]
                    if c + 1 < num_cols:
                        room.neighbors[Direction.right] = self.room_grid[r][c + 1]
                    if r + 1 < num_rows:
                        room.neighbors[Direction.down]  = self.room_grid[r + 1][c]
                    if c - 1 >= 0:
                        room.neighbors[Direction.left]  = self.room_grid[r][c - 1]
                    if r - 1 >= 0:
                        room.neighbors[Direction.up]    = self.room_grid[r - 1][c]
    
        print(f"[Loader] room_grid restored: {num_rows}x{num_cols}")


import math
import heapq
from collections import namedtuple
import numpy as np
from multigrid.core.actions import Action
from multigrid.core.constants import DIR_TO_VEC


Tile = namedtuple("Tile", ["i", "j"])
Label = namedtuple('Label', ['pos_state', 'len', 'g', 'f', 'action', 'parent'])

def heuristic_euclidean(start, target):
    h = (start.i - target.i)**2 + (start.j - target.j)**2 + (start.k - target.k)**2
    return h

def heuristic_manhattan(start, target):
    if not isinstance(start, Tile):
        start = Tile(*start)
    if not isinstance(target, Tile):
        target = Tile(*target)

    h = abs(start.i - target.i) + abs(start.j - target.j)
    return h

def create_root_label(pos_state, target, heuristic):
    """
    pos_state: ((x,y), dir, has_key) 或 ((x,y), dir, has_any_key, has_target_key)
    target   : (tx,ty) 或 Tile
    """
    pos, d, *flags = pos_state  # flags 为 [has_key] 或 [has_any_key, has_target_key]

    # 规范化 Tile
    cur = pos if hasattr(pos, "i") else Tile(*pos)
    tgt = target if hasattr(target, "i") else Tile(*target)

    # g/h/f
    g = 0
    h = heuristic(cur, tgt)
    f = g + h

    # 布尔化标志位，支持 numpy.bool_
    flags = [bool(x) for x in flags]

    # 组回用于搜索的状态元组（3元或4元，自动匹配）
    state = ((cur.i, cur.j), d, *flags)

    # Label: (state, depth, g, f, parent, action)
    return Label(state, 0, g, f, None, None)

def extend(current, pos_state, target, cost, heuristic, action):
    """
    兼容：
      pos_state = ((x,y), dir, has_key)
      或 pos_state = ((x,y), dir, has_any_key, has_target_key)
    """
    pos, d, *flags = pos_state  # flags 为 [has_key] 或 [has_any_key, has_target_key]

    # 规范化 Tile
    cur_tile = pos if hasattr(pos, "i") else Tile(*pos)
    tgt_tile = target if hasattr(target, "i") else Tile(*target)

    # 累积代价与启发
    next_g   = current.g + (float(cost) if hasattr(cost, "__float__") else cost)
    next_h   = heuristic(cur_tile, tgt_tile)
    next_f   = next_g + next_h
    next_len = current.len + 1

    # 标志位统一转为 bool，避免 numpy.bool_ 等类型带来的哈希/比较问题
    flags = [bool(x) for x in flags]

    # 统一把位置存成 (i, j) 元组；其余标志按原样展开（3 元或 4 元自动匹配）
    next_state = ((cur_tile.i, cur_tile.j), d, *flags)

    # Label(state, len, g, f, action, parent)
    return Label(next_state, next_len, next_g, next_f, action, current)

def get_path(label):
    path = []
    while label is not None:
        path.append((label.action, label.pos_state))
        label = label.parent
    path.reverse()
    return path

# --- 把你的 astar 改成“开门版” ---
def astar_open(pos_state, target, env, cost=None, heuristic=heuristic_manhattan, max_iter=None,version = None):
    """
    目标：打开位于 target=(tx,ty) 的门。默认我们已持有对应钥匙。
    成功条件：产生一次 Action.toggle（即 current.pos_state[2] == True）。
    """
    # target: (tx,ty) 或 Tile，统一成 Tile
    target = Tile(*target) if not hasattr(target, "i") else target
    door_pos = (target.i, target.j)

    # 若门已是开的，直接返回空路径
    obj0 = env.grid.get(*door_pos)
    if obj0 is not None and hasattr(obj0, "is_open") and obj0.is_open:
        return []  # already open

    # 三元根状态：opened=False
    (pos, d) = pos_state
    root_state = ((pos[0], pos[1]), d, False)
    node0 = create_root_label(root_state, target, heuristic)

    pq = []
    heapq.heappush(pq, (node0.f, node0))
    visited = {}
    it = 0
    found = None

    while pq:
        _, current = heapq.heappop(pq)

        # 目标：已经对目标门执行了 toggle（opened=True）
        if current.pos_state[2] is True:
            found = current
            break

        if max_iter and it > max_iter:
            return None

        key = (tuple(current.pos_state[0]), int(current.pos_state[1]), bool(current.pos_state[2]))
        if key in visited and visited[key] <= current.g:
            it += 1
            continue
        visited[key] = current.g

        # 使用“开门版” successor；传入目标门坐标
        successors = get_successor_open(env, current.pos_state, target_door_pos=door_pos,version=version)
        for action, succ in successors:
            next_pos, next_dir, next_opened = succ
            next_tile = Tile(*next_pos)
            edge_cost = cost[next_tile.i, next_tile.j] if isinstance(cost, np.ndarray) else 1
            child = extend(
                current,
                ((next_tile.i, next_tile.j), next_dir, next_opened),
                target,
                edge_cost,
                heuristic,
                action
            )
            heapq.heappush(pq, (child.f, child))
        it += 1

    if found:
        return get_path(found)
    return None

def astar_key(pos_state, target, env, cost=None,
              heuristic=heuristic_manhattan, max_iter=None,
              agent_idx=1, ActionEnum=Action, version=None):
    # target: (tx,ty) or Tile -> Tile
    target = Tile(*target) if not hasattr(target, "i") else target
    (pos, d) = pos_state

    # ==== 读取手上物体与目标物体，并计算两种布尔 ====
    carried = getattr(getattr(env.agents[agent_idx], "state", None), "carrying", None)
    has_any_key = _is_key(carried)

    tgt_obj = env.grid.get(target.i, target.j)
    tgt_color = getattr(tgt_obj, "color", None) if _is_key(tgt_obj) else None
    carried_color = getattr(carried, "color", None) if _is_key(carried) else None

    has_target_key = (has_any_key and tgt_color is not None and carried_color == tgt_color)

    # 四元状态: ((x,y), dir, has_any_key, has_target_key)
    root_state = ((pos[0], pos[1]), d, bool(has_any_key), bool(has_target_key))
    node0 = create_root_label(root_state, target, heuristic)

    pq = []
    heapq.heappush(pq, (node0.f, node0))
    visited = {}
    it = 0
    found = None

    while pq:
        _, current = heapq.heappop(pq)
        

        # 终止：已经拿到目标钥匙
        if current.pos_state[3] is True:
            found = current
            break

        if max_iter and it > max_iter:
            return [(None, 0), (ActionEnum.stay, None)]

        key = (
            tuple(current.pos_state[0]),
            int(current.pos_state[1]),
            bool(current.pos_state[2]),
            bool(current.pos_state[3]),
        )
        if key in visited and visited[key] <= current.g:
            it += 1
            continue
        visited[key] = current.g

        # 由 get_successor 生成后继（含 drop/pickup，但需 execute_action 判合法）
        successors = get_successor(
            env, current.pos_state, target_key_pos=(target.i, target.j), version=version
        )

        for action, succ in successors:
            next_pos, next_dir, next_has_any, next_has_target = succ
            next_tile = Tile(*next_pos)
            edge_cost = cost[next_tile.i, next_tile.j] if isinstance(cost, np.ndarray) else 1

            child = extend(
                current,
                ((next_tile.i, next_tile.j), next_dir, next_has_any, next_has_target),
                target,
                edge_cost,
                heuristic,
                action,
            )
            heapq.heappush(pq, (child.f, child))
        it += 1

    if found:
        path = get_path(found)  # [(action, ...), ...]
        if len(path) < 2:
            path = [(None, 0)] + path
        return path

    # 不可达或失败：返回两步占位
    return [(None, 0), (ActionEnum.stay, None)]
    
def _is_wall(obj):
    return obj is not None and getattr(obj, "type", None) == "wall"


def _is_closed_door(obj):
    return (obj is not None) and hasattr(obj, "is_open") and (not obj.is_open)
    
def execute_action(pos_state, action, env,version = None):
    """
    Execute an action in the environment.
    
    Parameters:
    state (tuple): The pos_state of the agent (pos, dir).
    action (int): The action to be executed.
    env (object): The environment object.
    
    Returns:
    status: The status of the action execution (True: success, False: failure).
    successor state (tuple): The successor state if the action is executed successfully, else None.
    """
    pos, dir = pos_state

    if action == Action.left:
        dir = (dir - 1) % 4
        return True, (pos, dir)
    if action == Action.right:
        dir = (dir + 1) % 4
        return True, (pos, dir)
    if action == Action.forward:
        dx, dy = DIR_TO_VEC[dir]
        new_pos = (pos[0] + dx, pos[1] + dy)
        # bounds
        if not (0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height):
            return False, (pos, dir)
        # everything except walls and CLOSED doors is walkable
        obj = env.grid.get(*new_pos)
        # walls block movement
        if version == None:
            if _is_wall(obj):
                return False, (pos, dir)
        if _is_closed_door(obj):
            return False, (pos, dir)
        # open door / empty / key / box / etc. -> pass
        return True, (new_pos, dir)
    
    if action == Action.drop and (env.agents[1].state._carried_obj is not None):
        dx, dy = DIR_TO_VEC[dir]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height and (env.grid.get(*new_pos) is None):
            return True, (pos, dir)
    if action == Action.stay:
        return True, (pos, dir)
    return False, (pos, dir) # If the agent hits a wall, return the current state


def execute_reverse_action(pos_state, action, env):
    """
    Execute an action in the environment.
    
    Parameters:
    state (tuple): The pos_state of the agent (pos, dir).
    action (int): The action to be executed.
    env (object): The environment object.
    
    Returns:
    status: The status of the action execution (True: success, False: failure).
    successor state (tuple): The successor state if the action is executed successfully, else None.
    """
    pos, dir = pos_state

    if action == Action.left:
        dir = (dir - 1) % 4
        return True, (pos, dir)

    if action == Action.right:
        dir = (dir + 1) % 4
        return True, (pos, dir)
        
    if action == Action.forward:
        dx, dy = DIR_TO_VEC[dir]
        new_pos = (pos[0] - dx, pos[1] - dy)
        # bounds
        if not (0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height):
            return False, (pos, dir)
        # walls block movement
        if env.base_grid[new_pos[0], new_pos[1]] == 2: # wall = 2
            return False, (pos, dir)
        # everything except walls and CLOSED doors is walkable
        obj = env.grid.get(*new_pos)
        if _is_closed_door(obj):
            return False, (pos, dir)
        # open door / empty / key / box / etc. -> pass
        return True, (new_pos, dir)
    
    # if action == Action.drop and (env.agents[1].state._carried_obj is not None):
    #     dx, dy = DIR_TO_VEC[dir]
    #     new_pos = (pos[0] - dx, pos[1] + dy)
    #     if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height and (env.grid.get(*new_pos) is not None):
    #         return True, (pos, dir)


    if action == Action.stay:
        return True, (pos, dir)
        
    return False, (pos, dir) # If the agent hits a wall, return the current state

def execute_obs_action(pos_state, action, env):
    """
    Execute an action in the environment.
    
    Parameters:
    state (tuple): The pos_state of the agent (pos, dir).
    action (int): The action to be executed.
    env (object): The environment object.
    
    Returns:
    status: The status of the action execution (True: success, False: failure).
    successor state (tuple): The successor state if the action is executed successfully, else None.
    """
    pos, dir = pos_state

    if action == Action.left:
        dir = (dir - 1) % 4
        return True, (pos, dir)

    if action == Action.right:
        dir = (dir + 1) % 4
        return True, (pos, dir)

    if action == Action.forward:
        dx, dy = DIR_TO_VEC[dir]
        new_pos = (pos[0] + dx, pos[1] + dy)

        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
            return True, (new_pos, dir)
        
        return False, (pos, dir) # If the agent hits a wall, return the current state
    
    if action == Action.stay:
        return True, (pos, dir)

def _is_key(obj):
    if obj is None:
        return False
    if getattr(obj, "type", None) == "key":
        return True
    return obj.__class__.__name__.lower() == "key"

def _front(pos, d):
    dx, dy = DIR_TO_VEC[d]
    return (pos[0] + dx, pos[1] + dy)


def _is_door(obj):
    return (obj is not None) and hasattr(obj, "is_open")  # duck typing

def get_successor_open(env, pos_state, target_door_pos,version = None):
    """
    生成后继（用于“开门”子目标的 A*）。
    pos_state = ((x,y), dir, opened)
    target_door_pos = (tx, ty) 目标门所在格
    - 基本动作：left/right/forward/stay -> 用 execute_action 扩展（不改 env）
    - 合成动作：若前方格==目标门格且确为门对象，则加入 Action.toggle，
                状态不移动/不转向，仅把 opened 置 True；同样不改 env。
    """
    (pos, d, opened) = pos_state
    successors = []

    # 基本动作（不改 env）
    for action in (Action.left, Action.right, Action.forward, Action.stay, Action.drop):
        ok, succ = execute_action((pos, d), action, env,version)
        if ok:
            successors.append((action, (succ[0], succ[1], opened)))

    # 合成 toggle（默认已有钥匙，不在搜索里校验颜色/锁）
    if not opened and target_door_pos is not None:
        fx, fy = _front(pos, d)
        if (fx, fy) == tuple(target_door_pos):
            obj = env.grid.get(fx, fy)
            if _is_door(obj):
                successors.append((Action.toggle, (pos, d, True)))

    return successors

def get_successor(env, pos_state, target_key_pos, version=None):
    """
    四元状态拓展：pos_state = ((x,y), dir, has_any_key, has_target_key)
    - 通过 execute_action 生成后继（不改 env）
    - 若前方格 == target_key_pos 且确有钥匙对象，且当前手上没有钥匙(not has_any_key)，
      则合成一个 Action.pickup 后继：置 has_any_key=True, has_target_key=True
    返回: List[(action, ((x,y), dir, has_any_key', has_target_key'))]
    """
    (pos, d, has_any, has_target) = pos_state
    successors = []

    # 基本动作（left/right/forward/stay/drop）
    for action in (Action.left, Action.right, Action.forward, Action.stay, Action.drop):
        ok, succ = execute_action((pos, d), action, env, version)
        if not ok:
            continue

        np, nd = succ[0], succ[1]
        if action == Action.drop:
            # 丢弃后两布尔均为 False
            successors.append((action, (np, nd, False, False)))
        else:
            # 普通移动不改变拿取状态
            successors.append((action, (np, nd, has_any, has_target)))

    # 合成 pickup：必须前方就是目标钥匙格，且当前没拿任何钥匙
    if not has_any and target_key_pos is not None:
        fx, fy = _front(pos, d)
        if (fx, fy) == tuple(target_key_pos):
            obj = env.grid.get(fx, fy)
            if _is_key(obj):
                # 针对“目标那把钥匙”的 A*，拿起即同时满足两布尔
                successors.append((Action.pickup, (pos, d, True, True)))

    return successors

def get_reverse_successor(env, pos_state):
    """
    Generate the next position and direction given the current position and direction.
    
    Parameters:
    env (object): The environment object containing the grid and other relevant information.
    pos (tuple): The current position (x, y).
    dir (int): The current direction (0: east, 1: south, 2: west, 3: north).
    
    Returns:
    list: A list of tuples representing the action and next position and direction.
    """
    pos, dir = pos_state
    successors = []

    # Define the possible actions
    actions = [Action.left, Action.right, Action.forward, Action.stay,Action.toggle,Action.drop,Action.pickup]

    for action in actions:
        status, successor = execute_reverse_action((pos, dir), action, env)
        if status:
            successors.append((action, successor))

    return successors

def get_obs_successor(env, pos_state):
    """
    Generate the next position and direction given the current position and direction.
    
    Parameters:
    env (object): The environment object containing the grid and other relevant information.
    pos (tuple): The current position (x, y).
    dir (int): The current direction (0: east, 1: south, 2: west, 3: north).
    
    Returns:
    list: A list of tuples representing the action and next position and direction.
    """
    pos, dir = pos_state
    successors = []

    # Define the possible actions
    actions = [Action.left, Action.right, Action.forward, Action.stay]

    for action in actions:
        status, successor = execute_obs_action((pos, dir), action, env)
        if status:
            successors.append((action, successor))

    return successors





# import heapq
# from collections import namedtuple

# # ========== 你现有的枚举/结构（按你的项目替换） ==========
# # 假设方向编码：0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# DIR2DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# class ActionEnum:
#     left    = "turn_left"
#     right   = "turn_right"
#     forward = "forward"
#     pickup  = "pickup"
#     toggle  = "toggle"
#     stay    = "stay"

# # 你已有的 Tile/Color 等也可直接用；这里 Tile 仅作帮助（可替换为元组）
# Tile = namedtuple("Tile", ["i","j"])

# # ========== 环境判定工具，按你的 env.grid 对象属性适配 ==========
# def _is_within(env, x, y):
#     return 0 <= x < env.width and 0 <= y < env.height

# def _is_wall(env, x, y):
#     obj = env.grid.get(x, y)
#     # 你提过“不要用 base_grid==2 判断墙”，因此从 obj.type 来判断
#     return (obj is not None) and (getattr(obj, "type", None) == "wall")

# def _is_door(obj):
#     return (obj is not None) and (getattr(obj, "type", None) == "door")

# def _is_key(obj):
#     return (obj is not None) and (getattr(obj, "type", None) == "key")

# def _door_is_open(obj, opened_doors_state, pos):
#     """门是否可通行：实际开着 或 在规划状态里已打开。"""
#     if not _is_door(obj):
#         return True
#     # 环境中的门开关
#     env_open = bool(getattr(obj, "is_open", False))
#     if env_open:
#         return True
#     # 规划状态中是否已打开（用位置标识门）
#     return (pos in opened_doors_state)

# def _can_toggle_door(obj, held_color):
#     """是否有能力开这扇门（解锁门 or 匹配颜色的钥匙）"""
#     if not _is_door(obj):
#         return False
#     # 若门没有上锁（有的环境 door.locked=False 表示不需要钥匙）
#     locked = bool(getattr(obj, "locked", True))
#     if not locked:
#         return True
#     # 否则需要颜色匹配的钥匙
#     door_color = getattr(obj, "color", None)
#     return (held_color is not None) and (held_color == door_color)

# def _is_walkable(env, x, y, opened_doors_state):
#     if not _is_within(env, x, y):
#         return False
#     if _is_wall(env, x, y):
#         return False
#     obj = env.grid.get(x, y)
#     if _is_door(obj):
#         return _door_is_open(obj, opened_doors_state, (x, y))
#     return True  # 普通地面、钥匙格等

# def _front_cell(pos, dir_):
#     dx, dy = DIR2DELTA[dir_ % 4]
#     return (pos[0] + dx, pos[1] + dy)

# # ========== A* 结点与工具 ==========
# class Node:
#     __slots__ = ("pos", "dir", "held", "opened", "g", "h", "f", "parent", "action")
#     def __init__(self, pos, dir_, held_color, opened_doors, g, h, parent=None, action=None):
#         self.pos    = pos               # (x,y)
#         self.dir    = dir_ % 4          # 0..3
#         self.held   = held_color        # None 或 颜色枚举/字符串
#         self.opened = opened_doors      # frozenset({(x,y), ...})
#         self.g      = g
#         self.h      = h
#         self.f      = g + h
#         self.parent = parent
#         self.action = action

#     def key(self):
#         # 用于 visited：位置、朝向、手中钥匙、开门集合
#         return (self.pos, self.dir, self.held, self.opened)

# def _heuristic_manhattan(a, b):
#     return abs(a[0]-b[0]) + abs(a[1]-b[1])

# def _reconstruct_path(node):
#     seq = []
#     cur = node
#     while cur is not None and cur.action is not None:
#         seq.append((cur.action, cur.pos))
#         cur = cur.parent
#     seq.reverse()
#     # 如果需要保证至少两步（兼容你现有 astar_key 的占位习惯），可在这里补齐：
#     if not seq:
#         seq = [(ActionEnum.stay, None)]
#     return seq

# # ========== 核心：统一 A* ==========
# def astar_unified_to_goal(pos_state, goal_xy, env,
#                           cost=None,
#                           heuristic=_heuristic_manhattan,
#                           max_iter=None,
#                           agent_idx=1,
#                           ActionEnum_=ActionEnum):
#     """
#     统一 A*：给终点（goal_xy），自动处理 拿钥匙→开门→到达。
#     参数
#     ----
#     pos_state : ((x,y), dir) 或 ((x,y), dir, held_color)
#     goal_xy   : (gx, gy)
#     env       : 你的环境，需支持 env.width/env.height 与 env.grid.get(x,y)
#     cost      : 可选代价图 (H×W 的 numpy 数组)；否则单位代价 1
#     heuristic : 启发函数
#     max_iter  : 迭代上限（防爆）
#     agent_idx : 如果你需要从 env.agents[agent_idx] 读取初始携带物；否则忽略
#     """
#     # 1) 读取初始位姿与初始钥匙
#     if len(pos_state) == 2:
#         (pos, dir_) = pos_state
#         # 看看代理手里是否已经拿着钥匙
#         carried = getattr(getattr(env.agents[agent_idx], "state", None), "carrying", None)
#         held_color = getattr(carried, "color", None) if _is_key(carried) else None
#     else:
#         (pos, dir_, held_color) = pos_state[:3]

#     pos = (int(pos[0]), int(pos[1]))
#     dir_ = int(dir_) % 4
#     goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

#     # 2) 初始化结点与容器
#     opened0 = frozenset()  # 规划过程中打开的门（位置集合）
#     h0 = heuristic(pos, goal_xy)
#     root = Node(pos, dir_, held_color, opened0, g=0, h=h0, parent=None, action=None)

#     pq = []
#     heapq.heappush(pq, (root.f, 0, root))  # (f, tie-breaker, node)
#     visited = {}  # state_key -> best_g
#     it = 0
#     tie = 1

#     # 3) 主循环
#     while pq:
#         _, _, cur = heapq.heappop(pq)

#         # 成功：到达目标格（你也可以加“面朝向”等附加条件）
#         if cur.pos == goal_xy:
#             return _reconstruct_path(cur)

#         if max_iter is not None and it >= max_iter:
#             break

#         skey = cur.key()
#         if skey in visited and visited[skey] <= cur.g:
#             it += 1
#             continue
#         visited[skey] = cur.g

#         x, y = cur.pos

#         # ---------- 1) 转向 ----------
#         # turn_left
#         n1 = Node(cur.pos, (cur.dir - 1) % 4, cur.held, cur.opened,
#                   g=cur.g + 1, h=heuristic(cur.pos, goal_xy),
#                   parent=cur, action=ActionEnum_.left)
#         heapq.heappush(pq, (n1.f, tie, n1))
#         tie += 1

#         # turn_right
#         n2 = Node(cur.pos, (cur.dir + 1) % 4, cur.held, cur.opened,
#                   g=cur.g + 1, h=heuristic(cur.pos, goal_xy),
#                   parent=cur, action=ActionEnum_.right)
#         heapq.heappush(pq, (n2.f, tie, n2))
#         tie += 1

#         # ---------- 2) 前进 ----------
#         fx, fy = _front_cell(cur.pos, cur.dir)
#         if _is_within(env, fx, fy) and _is_walkable(env, fx, fy, cur.opened):
#             step_cost = 1
#             if hasattr(cost, "__getitem__"):
#                 try:
#                     step_cost = float(cost[fx, fy])  # 注意(y,x)
#                 except Exception:
#                     step_cost = 1
#             n3 = Node((fx, fy), cur.dir, cur.held, cur.opened,
#                       g=cur.g + step_cost, h=heuristic((fx, fy), goal_xy),
#                       parent=cur, action=ActionEnum_.forward)
#             heapq.heappush(pq, (n3.f, tie, n3))
#             tie += 1

#         # ---------- 3) 拾取钥匙（前方交互） ----------
#         fx, fy = _front_cell(cur.pos, cur.dir)  # 已有
#         if _is_within(env, fx, fy):
#             obj_front = env.grid.get(fx, fy)
#             if _is_key(obj_front) and (cur.held is None):
#                 held_color_new = getattr(obj_front, "color", None)
#                 # 拾取前方格子的钥匙：状态中仅更新 held，不移动位置
#                 n4 = Node(cur.pos, cur.dir, held_color_new, cur.opened,
#                           g=cur.g + 1, h=heuristic(cur.pos, goal_xy),
#                           parent=cur, action=ActionEnum_.pickup)
#                 heapq.heappush(pq, (n4.f, tie, n4))
#                 tie += 1

#         # ---------- 4) 开门（前方交互） ----------
#         if _is_within(env, fx, fy):
#             obj_front = env.grid.get(fx, fy)
#             if _is_door(obj_front) and _can_toggle_door(obj_front, cur.held):
#                 opened_new = frozenset(set(cur.opened) | {(fx, fy)})
#                 # 开门：状态中记录该门坐标为已开，不移动位置
#                 n5 = Node(cur.pos, cur.dir, cur.held, opened_new,
#                           g=cur.g + 1, h=heuristic(cur.pos, goal_xy),
#                           parent=cur, action=ActionEnum_.toggle)
#                 heapq.heappush(pq, (n5.f, tie, n5))
#                 tie += 1

#         it += 1

#     # 不可达：返回安全占位（与现有接口风格兼容）
#     return [(ActionEnum_.stay, None)]

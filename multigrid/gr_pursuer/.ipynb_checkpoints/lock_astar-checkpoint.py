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
    pos, d, has_key = pos_state
    cur = Tile(*pos) if not hasattr(pos, "i") else pos
    tgt = target if hasattr(target, "i") else Tile(*target)
    g = 0
    h = heuristic(cur, tgt)
    f = g + h

    return Label(((cur.i, cur.j), d, has_key), 0, g, f, None, None)

def extend(current, pos_state, target, cost, heuristic, action):
    pos, d, has_key = pos_state
    cur_tile = pos if hasattr(pos, "i") else Tile(*pos)
    tgt_tile = target if hasattr(target, "i") else Tile(*target)

    next_g = current.g + cost
    next_h = heuristic(cur_tile, tgt_tile)
    next_f = next_g + next_h
    next_len = current.len + 1  # 避免覆盖内置 len

    # 统一把状态里的位置存成 (i,j) 元组
    next_state = ((cur_tile.i, cur_tile.j), d, has_key)

    return Label(next_state, next_len, next_g, next_f, action, current)

def get_path(label):
    path = []
    while label is not None:
        path.append((label.action, label.pos_state))
        label = label.parent
    path.reverse()
    return path

# --- 把你的 astar 改成“开门版” ---
def astar_open(pos_state, target, env, cost=None, heuristic=heuristic_manhattan, max_iter=None):
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
        successors = get_successor_open(env, current.pos_state, target_door_pos=door_pos)
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
              agent_idx=1, ActionEnum=Action):
    # target: (tx,ty) 或 Tile，统一成 Tile
    target = Tile(*target) if not hasattr(target, "i") else target
    (pos, d) = pos_state
    # ---- 前缀：若当前手里有钥匙，先丢掉 ----
    prefix = []
    carried = getattr(getattr(env.agents[agent_idx], "state", None), "_carried_obj", None)
    if carried is not None:
        # 你的执行层会识别这个动作；第二个元素占位，保持和你的 path 元素结构一致
        prefix.append((ActionEnum.drop, None))

    # 规划从“未持钥匙”的状态到目标钥匙坐标（拿到即到达）
    root_state = ((pos[0], pos[1]), d, False)  # 三元：(x,y), dir, has_key=False
    node0 = create_root_label(root_state, target, heuristic)

    pq = []
    heapq.heappush(pq, (node0.f, node0))
    visited = {}
    it = 0
    found = None

    while pq:
        _, current = heapq.heappop(pq)
        # 目标：已经拿到这把钥匙（get_successor 在到达 target 坐标后应把 has_key 置 True）
        if current.pos_state[2] is True:
            found = current
            break

        if max_iter and it > max_iter:
            # 只返回前缀（至少把手里的钥匙丢了）
            return prefix if prefix else None

        key = (tuple(current.pos_state[0]), int(current.pos_state[1]), bool(current.pos_state[2]))
        if key in visited and visited[key] <= current.g:
            it += 1
            continue
        visited[key] = current.g

        # 使用三元版 successor；传入目标钥匙坐标
        successors = get_successor(env, current.pos_state, target_key_pos=(target.i, target.j))

        for action, succ in successors:
            next_pos, next_dir, next_has_key = succ
            next_tile = Tile(*next_pos)

            edge_cost = cost[next_tile.i, next_tile.j] if isinstance(cost, np.ndarray) else 1

            child = extend(
                current,
                ((next_tile.i, next_tile.j), next_dir, next_has_key),  # 三元状态
                target,
                edge_cost,
                heuristic,
                action
            )
            heapq.heappush(pq, (child.f, child))
        it += 1
    if found:
        path = get_path(found)  # 你的 get_path 返回 [(action, ...), ...]
        return ([path[0]] + prefix + path[1:]) if prefix else path
    # 没找到路：至少把手里的钥匙丢掉
    return prefix if prefix else None
    
def _is_wall(env, p):
    return getattr(env, "base_grid", None) is not None and env.base_grid[p[0], p[1]] == 2


def _is_closed_door(obj):
    return (obj is not None) and hasattr(obj, "is_open") and (not obj.is_open)
    
def execute_action(pos_state, action, env):
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
        # walls block movement
        if env.base_grid[new_pos[0], new_pos[1]] == 2: # wall = 2
            return False, (pos, dir)
        # everything except walls and CLOSED doors is walkable
        obj = env.grid.get(*new_pos)
        if _is_closed_door(obj):
            return False, (pos, dir)
        # open door / empty / key / box / etc. -> pass
        return True, (new_pos, dir)
    
    if action == Action.drop and (env.agents[1].state._carried_obj is not None):
        dx, dy = DIR_TO_VEC[dir]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height and (env.grid.get(*new_pos) is not None):
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

def get_successor_open(env, pos_state, target_door_pos):
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
        ok, succ = execute_action((pos, d), action, env)
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

def get_successor(env, pos_state, target_key_pos):
    """
    三元状态拓展：pos_state = ((x,y), dir, has_key)
    - 只用 left/right/forward/stay 通过 execute_action 生成后继（不改 env）
    - 若前方格 == target_key_pos 且确有钥匙对象，则合成一个 Action.pickup 后继，
      不移动位置/朝向，只把 has_key 置 True（同样不改 env）
    返回: List[(action, ((x,y), dir, has_key'))]
    """
    (pos, d, has_key) = pos_state
    successors = []

    # 仅使用不会改变 env 的基本动作
    for action in (Action.left, Action.right, Action.forward, Action.stay, Action.drop):
        ok, succ = execute_action((pos, d), action, env)
        if ok:
            successors.append((action, (succ[0], succ[1], has_key)))

    # 合成 pickup：前方必须正是目标钥匙格，且当前还没拿到
    if not has_key and target_key_pos is not None:
        fx, fy = _front(pos, d)
        if (fx, fy) == tuple(target_key_pos):
            obj = env.grid.get(fx, fy)
            if _is_key(obj):
                successors.append((Action.pickup, (pos, d, True)))

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
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

# def heuristic_euclidean_plane_flyaltitude(start, target, fly_altitude=None):
#     h = (start.i - target.i)**2 + (start.j - target.j)**2
#     if fly_altitude is not None:
#         h += (start.k - fly_altitude)**2
#     return math.sqrt(h)

def create_root_label(pos_state, target, heuristic):
    next_g = 0
    next_h = heuristic(pos_state[0], target)
    next_f = next_g + next_h
    next = Label(pos_state, 0, next_g, next_f, None, None)
    return next

def extend(current, pos_state, target, cost, heuristic, action):
    next_g = current.g + cost
    next_h = heuristic(pos_state[0], target)
    next_f = next_g + next_h
    len = current.len+1
    next = Label(pos_state, len, next_g, next_f, action, current)
    return next

def get_path(label):
    path = []
    while label is not None:
        
        path.append((label.action, label.pos_state))
        label = label.parent

    path.reverse()
    return path


def astar(pos_state, target, env, cost = None, heuristic=heuristic_manhattan, max_iter=None):

    print(pos_state,target)
    target = Tile(*target)
    pos, dir = pos_state
    next = create_root_label(((pos[0], pos[1]), dir), target, heuristic)

    pq = []
    heapq.heappush(pq, (next.f, next))
    found = None
    pos, dir = pos_state
    visited = {}
    iter = 0
    while pq:
        _, current = heapq.heappop(pq)
        #print(current.pos_state[0])
        if current.pos_state[0][0] == target.i and current.pos_state[0][1] == target.j:
            found = current
            print("Goal Found!")
            break

        
        # Max number of nodes explored reached
        if max_iter and iter>max_iter:
            return None

        info_tuple = (tuple(current.pos_state[0]), int(current.pos_state[1]))

        if info_tuple in visited and visited[info_tuple] < current.g:
            continue
        else:
            visited[info_tuple] = current.g
        
        successors = get_successor(env, current.pos_state)

        for action, succ in successors:

            next_pos, next_dir = succ
            next_pos = Tile(*next_pos)

            if isinstance(cost, np.ndarray):
                edge_cost = cost[next_pos.i, next_pos.j]
            else:
                edge_cost = 1

            next = extend(current, ((next_pos.i, next_pos.j), next_dir), target, edge_cost, heuristic, action)

            heapq.heappush(pq, (next.f, next))

        iter += 1
    
    if found:
        path = get_path(found)
        return path
    
    return None


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
        obj = env.grid.get(new_pos[0], new_pos[1])
        if obj is not None and obj.type == "door":
            if not obj.is_open:
                return False, (pos, dir)
        # open door / empty / key / box / etc. -> pass
        return True, (new_pos, dir)
    
    if action == Action.stay:
        return True, (pos, dir)
    

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
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
            return True, (new_pos, dir)
        
        return False, (pos, dir) # If the agent hits a wall, return the current state
    
    if action == Action.stay:
        return True, (pos, dir)

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

def get_successor(env, pos_state):
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
        status, successor = execute_action((pos, dir), action, env)
        if status:
            successors.append((action, successor))

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
    actions = [Action.left, Action.right, Action.forward, Action.stay]

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
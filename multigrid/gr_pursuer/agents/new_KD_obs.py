from .base import BaseAgent
from ..new_astar import astar, get_successor, execute_action, get_obs_successor, get_reverse_successor
#from ..lock_astar import astar_key,astar_open,get_successor

import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction,OBJECT_TO_IDX,COLOR_TO_IDX
from multigrid.core.actions import Action
from multigrid.utils.obs import gen_obs_grid_encoding
from multigrid.core.constants import Type
import random
import os
from collections import deque
from copy import deepcopy

# MODES
TRACK = 0
MOVE2GOAL = 1
BETA = 1


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


# 四个朝向的前向/右向单位向量（east, south, west, north）
FWD = {
    0: np.array([ 1,  0]),  # east
    1: np.array([ 0,  1]),  # south
    2: np.array([-1,  0]),  # west
    3: np.array([ 0, -1]),  # north
}
def right_vec_from_fwd(fwd):
    # 将前向向量旋转 +90° 得到“右向”向量
    return np.array([-fwd[1], fwd[0]])

def decode_fov_to_world(fov, observer_pos, observer_dir, grid_width, grid_height):
    """
    fov: (V, V, 3) 的整型数组
    observer_pos: np.array([x, y])
    observer_dir: 0/1/2/3 (东/南/西/北)
    grid_width, grid_height: 世界网格宽高
    返回: 列表，每项为 dict: {"x","y","type","color","state"}
    """
    V = fov.shape[0]
    f_vec = FWD[observer_dir]
    r_vec = right_vec_from_fwd(f_vec)

    # 本地FoV坐标系到世界的“左上角”（以观察者朝向定义的左上）
    top_left = (observer_pos
                + f_vec * (V - 1)
                - r_vec * (V // 2))

    out = []
    for j in range(V):          # 前后（面朝的方向为 -f_vec）
        for i in range(V):      # 左右（右手方向为 +r_vec）
            t, c, s = fov[i, j]   # 注意：MiniGrid 的可视图常用 (i, j) 这种顺序
            if t == 0 and c == 0 and s == 0:
                continue  # unseen 跳过（如需保留可去掉这行）

            abs_xy = top_left - f_vec * j + r_vec * i
            x, y = int(abs_xy[0]), int(abs_xy[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                out.append({"x": x, "y": y, "type": int(t), "color": int(c), "state": int(s)})
    return out



class BeliefUpdateObserver(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        
        super().__init__(env.observer)

        self.env = env
        self.agent.name = "BeliefUpdateObserver"
        self.agent.can_overlap = True
        self.goals = [g for g in env.goals]
        HALLWAY_COL = env.num_cols // 2
        self.all_goals = [g for g in env.goals] # all subgoals
        self.final_goals = [g for g in env.goals]

        # === [新增] 任务链结构 + 当前 subgoal 指针 ===
        self.goal_chains = {}                 # 每个goal的任务链
        self.current_subgoal_idx = {}         # 当前subgoal的索引
        for goal in self.goals:
            chain = []
            goal_room = _rid_from_xy(self.env, goal[0], goal[1], HALLWAY_COL)
            plan_events = list(self.env.plans.get(goal_room, []))

            for ev in plan_events:
                pos = tuple(ev['pos']['value'])
                stype = ev['type']
                color = COLOR_TO_IDX[ev.get('key',ev.get('color', None))]
                chain.append((stype, pos))   # 用tuple记录subgoal
                self.all_goals.append(pos) # save all subgoals
            
            chain.append(('reach', tuple(goal), None))  # 终点
            self.goal_chains[goal] = chain
            self.current_subgoal_idx[goal] = 0

        # print("self.goal_chains : ")
        # for goal in self.goal_chains:
        #     print(self.goal_chains[goal][0][1])

        self.goals = [self.goal_chains[goal][0][1]  for goal in self.goal_chains] # 更新goals 的名字
        
        print(" ")
        print("self.current_subgoal_idx: ",self.current_subgoal_idx )
        
        self.step = -1
        self.pos = env.observer.pos
        self.dir = env.observer.dir

        if init_goal_belief:
            self.goal_belief = init_goal_belief
        else:
            self.goal_belief = { g:1/len(self.goals) for g in self.goals}


        if init_actor_belief:
            self.actor_belief = init_actor_belief
        else:
            self.actor_belief = {g: set_uniform_prob(env.base_grid, self.goal_belief[g]) for g in self.goals}

        self.dist_matrix = self.compute_pairwise_distances()

            # === [新增] 初始化 subgoal→goal 的映射 ===
        self._rebuild_subgoal_to_goals()
        
    def _rebuild_subgoal_to_goals(self):
        self.subgoal_to_goals = {}
        new_goals = []  # 用来收集新的 subgoal 位置
    
        for g in self.final_goals:
            idx = self.current_subgoal_idx[g]
            chain = self.goal_chains[g]
            if idx >= len(chain):
                continue
            s = chain[idx]  # 当前 subgoal: ('pickup'|'open'|'reach', pos, color)
            pos = s[1]
            self.subgoal_to_goals.setdefault(s, set()).add(g)
    
            # === 更新 goal_belief ===
            old_pos = chain[idx - 1][1] if idx > 0 else None
            if old_pos in self.goal_belief:
                self.goal_belief[pos] = self.goal_belief.pop(old_pos)
            else:
                # 如果没有旧的，就初始化
                self.goal_belief[pos] = self.goal_belief.get(pos, 0.0)
    
            # === 更新 actor_belief ===
            if old_pos in self.actor_belief:
                self.actor_belief[pos] = self.actor_belief.pop(old_pos)
            else:
                self.actor_belief[pos] = self.actor_belief.get(
                    pos, set_uniform_prob(self.env.base_grid, 1.0))
    
            # 收集新的目标位置
            new_goals.append(pos)
    
        # === 更新 self.goals ===
        self.goals = new_goals
            


            
    def update_goal_belief_from_subgoal(self, subgoal_posterior: dict, renorm=True):
        new_goal_belief = {g: 0.0 for g in self.final_goals}

        for s, p in subgoal_posterior.items():
            goals_needing_s = self.subgoal_to_goals.get(s, set())
            if not goals_needing_s:
                continue
            share = p / len(goals_needing_s)
            for g in goals_needing_s:
                new_goal_belief[g] += share

        if renorm:
            total = sum(new_goal_belief.values())
            if total > 0:
                for g in new_goal_belief:
                    new_goal_belief[g] /= total
            else:
                new_goal_belief = dict(self.goal_belief)
        print("final goal belief:" , new_goal_belief)
        #self.goal_belief = new_goal_belief


    def compute_action(self, obs):
        self.step += 1
        self.pos = obs["observer_pos"]
        self.dir = obs["observer_dir"]
        fov = obs['image']  

        self.update_belief(obs) 

        self.update_goal_belief() 

        subgoal_posterior = {}

        grid_w, grid_h = self.env.width, self.env.height
        visible = decode_fov_to_world(fov, self.pos, self.dir, grid_w, grid_h)
        completed = self.check_complete(visible)  # 收集完成的subgoal事件: ('pickup'|'open', (x,y), color)


        print(completed)
        # === [NEW] 将事件转换为 subgoal 完成，推进任务链 + 概率继承 ===
        for s in completed:
            self.on_subgoal_completed(s)

        for g in self.final_goals:
            idx = self.current_subgoal_idx[g]           # 需要你已按前文新增的结构维护
            s = self.goal_chains[g][idx]                # s 是 ('pickup'|'open'|'reach', pos, color)
            subgoal_posterior[s] = subgoal_posterior.get(s, 0.0) + self.goal_belief[s[1]]
    
        # 把 subgoal 概率平均分配回共享它的多个 goal，并归一化
        self.update_goal_belief_from_subgoal(subgoal_posterior, renorm=True)
        
        self.actor_belief = update_actor_belief(self.actor_belief, self.goals, self.env, self.dist_matrix) 
        
        #self.render_and_save(f'belief_update_test/actor_belief_step_{self.step}.png', obs)

        print("current goal_belief:", self.goal_belief)

        return self.mcts()

    def check_complete(self,visible): # need add check actor state
        completed = []  # 收集完成的subgoal事件: ('pickup'|'open', (x,y), color)
        for v in visible:
            x, y = v['x'], v['y']
            cur_type = int(v['type'])
            base_type = int(self.env.base_grid[x, y])
            # 钥匙：base是key，但当前不是key => 被拿走
            if base_type == OBJECT_TO_IDX['key'] and cur_type != OBJECT_TO_IDX['key']:
                completed.append(('pickup', (x, y)))
            # 门：如果当前位置是门，且可从可见信息判断为open => 门被打开
            # 注：有些实现里 base_grid 只有type没有state；只要FOV里能读到 'state'=='open' 就触发
            if base_type == OBJECT_TO_IDX['door']:
                v_state = self.env.grid.get(x,y)
                if v_state.state == 'open':
                    completed.append(('open', (x, y)))
        return completed

    # === [新增] 事件触发：subgoal 完成后推进任务链 === # need add jump the subgoal
    def on_subgoal_completed(self, completed_subgoal):
        #print(completed_subgoal,self.subgoal_to_goals)
        goals_hit = self.subgoal_to_goals.get(completed_subgoal, set())
        #print(goals_hit)
        if not goals_hit:
            return
        for g in goals_hit:
            self.current_subgoal_idx[g] = min(
                self.current_subgoal_idx[g] + 1,
                len(self.goal_chains[g]) - 1
            )
        self._rebuild_subgoal_to_goals()
        print("[Info] Subgoal completed, advanced:", completed_subgoal)

    def compute_pairwise_distances(self):
        """
        Compute all pairwise distances from each state (position and direction) to the goal locations using BFS.
        """
        #free_cells = np.argwhere(self.env.base_grid == 0)
        free_cells = np.argwhere(self.env.base_grid != 2)
        
        num_cells = len(free_cells)
        num_directions = 4  # Number of possible directions (east, south, west, north)
        num_states = num_cells * num_directions

        cell_to_index = {tuple(cell): idx for idx, cell in enumerate(free_cells)}
        # Initialize distance matrix for distances to goal locations
        dist_matrix = {goal: np.full(num_states, np.inf) for goal in self.all_goals}
        for goal in self.all_goals:
            goal_idx = cell_to_index[tuple(goal)]
            queue = deque([(goal_idx, dir, 0) for dir in range(num_directions)])  # (cell_index, direction, distance)
            visited = set()

            while queue:
                current_idx, current_dir, current_dist = queue.popleft()
                state = (current_idx, current_dir)
                if state in visited:
                    continue
                visited.add(state)

                dist_matrix[goal][current_idx * num_directions + current_dir] = current_dist

                pos_state = (free_cells[current_idx], current_dir)
                for action, next_pos_state in get_reverse_successor(self.env, pos_state):
                    next_pos, next_dir = next_pos_state
                    if tuple(next_pos) in cell_to_index:
                        next_idx = cell_to_index[tuple(next_pos)]
                        queue.append((next_idx, next_dir, current_dist + 1))
        adjusted_dist_matrix = dict()
        for i in range(num_states):
            for goal in self.all_goals:
                cell = free_cells[i // num_directions]
                pos_state = (tuple(cell), i % num_directions)
                adjusted_dist_matrix[(pos_state, tuple(goal))] = dist_matrix[goal][i]

        return adjusted_dist_matrix
        
    def mcts(self, iterations = 100, exploration_weight = 1):
        start_pos_state = (self.pos, self.dir)
        start_actor_belief = deepcopy(self.actor_belief)
        start_goal_belief = deepcopy(self.goal_belief)
        root = MCTSNode(self.agent, start_pos_state, start_actor_belief, start_goal_belief, self.env, self.dist_matrix)
        
        for _ in range(iterations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(exploration_weight)
            
            if not node.is_terminal():
                node = node.expand()
            result = node.rollout()
            node.backpropagate(result)

        return root.best_child(0).action

        
    def update_belief(self, obs):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        if "target_pos" in obs or self.pos == self.env.target.pos:
            print(self.step)
            #print(self.env.target.carrying)
            print("in view")
            target_pos = self.env.target.pos
            target_dir = self.env.target.dir # 0-3 denote east south west north respectively
            
            
            for goal in self.goals:
                new_actor_belief = np.zeros_like(self.actor_belief[goal])
                new_actor_belief[tuple(target_pos)][target_dir] = self.actor_belief[goal][tuple(target_pos)][target_dir]
                self.actor_belief[goal] = new_actor_belief
                
        else:
            print(self.step)
            print("not in view")
            obs_shape = self.agent.observation_space['image'].shape[:-1]
            vis_mask = np.zeros_like(obs_shape, dtype=bool)
            vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer
            highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)

            # of the agent's view area
            f_vec = self.agent.state.dir.to_vec()
            r_vec = np.array((-f_vec[1], f_vec[0]))
            top_left = (
                self.agent.state.pos
                + f_vec * (self.agent.view_size - 1)
                - r_vec * (self.agent.view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent.view_size):
                for vis_i in range(0, self.agent.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.env.width:
                        continue
                    if abs_j < 0 or abs_j >= self.env.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True
            # highlight_mask = obs['fov']
            for goal in self.goals:
                for cell in np.argwhere(highlight_mask == 1):
                    self.actor_belief[goal][tuple(cell)] = 0
                #print(self.actor_belief[goal])
               
    def update_goal_belief(self):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        for goal in self.goals:
            self.goal_belief[goal] = np.sum(self.actor_belief[goal])

        total = sum(self.goal_belief.values())
        if total == 0:
            print("should not happen,1")
            print(self.goal_belief)
            for goal in self.goals:
                print(np.where(self.actor_belief[goal]>0))
            input()
        for goal in self.goal_belief:
            self.goal_belief[goal] /= total


    # def render_and_save(self, filename, obs):
    #     """
    #     Render the environment and save the visualization.
        
    #     Parameters:
    #     filename (str): The name of the file to save the visualization.
    #     obs (dict): The observation dictionary containing the observer and goal positions.
    #     """
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)

    #     total_belief = np.zeros_like(next(iter(self.actor_belief.values())))
    #     for goal, belief in self.actor_belief.items():
    #         total_belief += belief

    #     belief_sum = np.sum(total_belief, axis=2)
    #     log_belief_sum = np.log(belief_sum + 1e-10)
    #     vmin = np.min(log_belief_sum)
    #     vmax = np.max(log_belief_sum)
        
    #     plt.imshow(log_belief_sum, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     plt.colorbar()
    #     goal_colors = ['yellow', 'green', 'cyan', 'magenta', 'orange']
    #     goal_probs = [self.goal_belief[goal] for goal in self.goals]
    #     goal_text = '\n'.join([f'Goal {i+1} ({goal_colors[i % len(goal_colors)]}): {prob:.2f}' for i, prob in enumerate(goal_probs)])
    #     plt.title(goal_text)

    #     # Overlay obstacles
    #     obstacles = np.where(self.env.base_grid == 2)
    #     plt.scatter(obstacles[1], obstacles[0], c='black', marker='s', label='Obstacle')

    #     # Overlay observer position
    #     observer_pos = obs["observer_pos"]
    #     plt.scatter(observer_pos[1], observer_pos[0], c='blue', marker='o', label='Observer')

    #     # Overlay goal positions
        
    #     for i, goal in enumerate(self.goals):
    #         plt.scatter(goal[1], goal[0], c=goal_colors[i % len(goal_colors)], marker='*', label='Goal')

    #     # Overlay target position if observed
    #     if "target_pos" in obs:
    #         target_pos = obs["target_pos"]
    #         plt.scatter(target_pos[1], target_pos[0], c='red', marker='x', label='Target')


    #     plt.savefig(filename)
    #     plt.close()


def update_actor_belief(actor_belief, goals, env, dist_matrix, beta = BETA):
    goals = [goal for goal in actor_belief]
    new_actor_belief = {goal: np.zeros_like(actor_belief[goal]) for goal in actor_belief}

    for goal in goals:
        current_actor_belief = actor_belief[goal]
        for cell in np.argwhere(current_actor_belief > 0): # select no zero prob
            pos, dir = cell[:2], cell[2]
            pos_state = (pos, dir)
            prob = current_actor_belief[tuple(cell)]
            successors = get_successor(env, pos_state)

            tran_probs = {}

            if pos[0] == goal[0] and pos[1] == goal[1]:
                successors = list(filter(lambda x: x[0] == Action.stay, successors))

            for action, succ in successors:

                next_pos, next_dir = succ

                succ = ((next_pos[0], next_pos[1]), next_dir)

                if (succ, goal) in dist_matrix:
                    tran_probs[succ] = math.exp(- beta * (1 + dist_matrix[(succ, goal)]))

                else:
                    print("should not happen,2")
                    input()
                    tran_probs[succ] = 0

            total_prob = sum(tran_probs.values())
            if total_prob > 0:
                for succ in tran_probs:
                    tran_probs[succ] /= total_prob

            for action, succ in successors:
                new_actor_belief[goal][succ[0][0],succ[0][1],succ[1]] += prob*tran_probs[((succ[0][0],succ[0][1]),succ[1])]

    return new_actor_belief


def set_uniform_prob(grid, total_prob = 1):
    """
    Set a uniform probability for all free cells in the grid.
    
    Parameters:
    grid (np.array): The grid to be analyzed.
    
    Returns:
    np.array: A grid with uniform probabilities for all free cells.
    """
    dir = 4
    free_cells = np.argwhere(grid != 2)
    num_free_cells = len(free_cells)
    uniform_prob = total_prob / (num_free_cells * dir) if num_free_cells > 0 else 0

    prob_grid = np.zeros((*grid.shape, dir), dtype=float)
    for cell in free_cells:
        prob_grid[tuple(cell)] = uniform_prob

    return prob_grid



class MCTSNode:
    def __init__(self, agent, pos_state, actor_belief, goal_belief, env, dist_matrix, action = None, parent=None):
        self.agent = agent
        self.dist_matrix = dist_matrix
        self.pos_state = pos_state  # The current game state
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.actor_belief = {goal: actor_belief[goal] for goal in actor_belief}
        self.goal_belief = {goal: goal_belief[goal] for goal in goal_belief}
        self.env = env
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.value = 0  # Total value of the node
        self.goals = [goal for goal in actor_belief]

    def is_fully_expanded(self):
        return len(self.children) == len(get_obs_successor(self.env, self.pos_state))

    def best_child(self, exploration_weight=1.0):
        """Selects the best child using UCT for decision nodes and expectation for chance nodes."""

        return max(
            self.children, 
            key=lambda child: (child.value / (child.visits + 1e-6)) + 
                              exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
        )

    def expand(self):
        """Expands the node by adding a new child node."""
        tried_moves = {child.action for child in self.children}
        possible_succs = get_obs_successor(self.env, self.pos_state)

        for action, next_pos_state in possible_succs:
            if action not in tried_moves:
                g = self.sample_goal()
 
                actor_pos_state = self.sample_from_3d_belief(self.actor_belief[g])

                new_actor_belief = self.update_actor_belief_from_obs(actor_pos_state, next_pos_state)

                new_goal_belief = self.update_goal_belief(new_actor_belief)
      
                # goal directed update of the actor belief
                new_actor_belief = update_actor_belief(new_actor_belief, self.goals, self.env, self.dist_matrix)


                new_node = MCTSNode(self.agent, next_pos_state, new_actor_belief, new_goal_belief, self.env, self.dist_matrix, action = action, parent=self)
                self.children.append(new_node)
                return new_node

    def sample_goal(self):
        """Samples a goal based on the probability distribution in self.goal_belief."""
        goals = list(self.goal_belief.keys())  # Extract possible goals
        probabilities = np.array(list(self.goal_belief.values()))  # Extract probabilities

        if probabilities.sum() == 0:
            print("should not happen,4")
            print(self.goal_belief)
            input()
        # Normalize probabilities to ensure they sum to 1
        probabilities /= probabilities.sum()

        # Sample a goal based on the normalized probability distribution
        sampled_goal = np.random.choice(len(goals), p=probabilities)
        return goals[sampled_goal]
            
    def update_goal_belief(self, actor_belief):
        new_goal_belief = {}
        for goal in self.goal_belief:
            new_goal_belief[goal] = np.sum(actor_belief[goal])

        total = sum(new_goal_belief.values())
        for goal in self.goal_belief:
            new_goal_belief[goal] /= total

        return new_goal_belief
            
    def update_actor_belief_from_obs(self, actor_pos_state, observer_pos_state):
        
        new_actor_belief = {goal: np.zeros_like(self.actor_belief[goal]) for goal in self.actor_belief}

        actor_pos = actor_pos_state[0], actor_pos_state[1]
        actor_dir = actor_pos_state[2]

        observer_pos = observer_pos_state[0]
        observer_dir = Direction(observer_pos_state[1])
        obs_shape = self.agent.observation_space['image'].shape[:-1]
        vis_mask = np.zeros_like(obs_shape, dtype=bool)
        vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer


        highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)


        # of the agent's view area
        f_vec = observer_dir.to_vec()
        r_vec = np.array((-f_vec[1], f_vec[0]))
        top_left = (
            observer_pos
            + f_vec * (self.agent.view_size - 1)
            - r_vec * (self.agent.view_size // 2)
        )

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent.view_size):
            for vis_i in range(0, self.agent.view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True # means FoV


        if highlight_mask[actor_pos]: # if the actor is in the observer's view
            for g in self.goal_belief:
                new_actor_belief[g][actor_pos][actor_dir] = self.actor_belief[g][actor_pos][actor_dir] 
        else:
            for g in self.goal_belief: # not in observer's view: for each grid in FoV = 0, otherwise use past actor_belief
                for cell in np.argwhere(highlight_mask == True):
   
                    new_actor_belief[g][tuple(cell)] = 0

                for cell in np.argwhere(highlight_mask == False):
                    new_actor_belief[g][tuple(cell)] = self.actor_belief[g][tuple(cell)]
                

        return new_actor_belief

        
    def is_terminal(self):
        return self.env.is_done()

    def rollout(self):
        """Simulates the game to the end from the current state and returns the result."""
        return -compute_entropy(self.goal_belief)


    def backpropagate(self, result, action_penalty=0):
        """Updates the tree nodes based on the result of the rollout."""
        self.visits += 1
        self.value += result - action_penalty
        if self.parent:
            self.parent.backpropagate(result)

    

    def sample_from_3d_belief(self, actor_belief):
        """Samples a location from the 3D belief map using probability distribution."""
        depth, height, width = actor_belief.shape  # Get dimensions
        # Flatten the 3D belief map into a 1D array
        flattened_belief = np.copy(actor_belief).ravel()

        if flattened_belief.sum() == 0:
            print("should not happen,3")
            print(actor_belief)
            input()
        # Normalize probabilities to ensure they sum to 1
        flattened_belief /= flattened_belief.sum()

        # Sample an index based on the belief distribution
        sampled_index = np.random.choice(len(flattened_belief), p=flattened_belief)

        # Convert the 1D index back to 3D coordinates
        sampled_depth, rem = divmod(sampled_index, height * width)
        sampled_row, sampled_col = divmod(rem, width)

        return sampled_depth, sampled_row, sampled_col  # Return sampled (z, y, x) coordinates

def compute_entropy(goal_belief):
    """Computes the Shannon entropy of the goal belief distribution."""
    probabilities = np.array(list(goal_belief.values()))
    # probabilities = goal_belief
    
    # Ensure the probabilities sum to 1
    probabilities /= probabilities.sum()
    
    # Compute entropy, avoiding log(0) by filtering out zero probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Small offset to avoid log(0)
    
    return entropy

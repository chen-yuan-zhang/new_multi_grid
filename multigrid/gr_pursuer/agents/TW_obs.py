from .base import BaseAgent
from ..new_astar import astar, get_successor, execute_action, get_obs_successor, get_reverse_successor

import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction,Color
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

COLOR_ORDER = ["red","green","blue","purple","yellow","grey"]
COLOR2IDX = {name: i+1 for i, name in enumerate(COLOR_ORDER)}  # 0 预留给空手

def carrying_to_idx(carrying) -> int:
    # None / 0 / False 都当作空手
    if carrying in (None, 0, False):
        return 0
    # 已经是整数索引就直接用
    if isinstance(carrying, (int, np.integer)):
        return int(carrying)
    # 可能是 Key 对象或枚举，取颜色名
    color = getattr(carrying, "color", carrying)
    name = getattr(color, "name", str(color)).lower()
    return COLOR2IDX.get(name, 0)  # 未知颜色也回退到 0

# def renormalize_belief(belief_4d, target_total=1.0, eps=1e-12):
#     """
#     Renormalize a (W,H,4,num_keys) belief tensor to a target total mass.
#     """
#     s = belief_4d.sum()
#     if s < eps:
#         return belief_4d  # nothing to scale; caller may re-seed elsewhere
#     return belief_4d * (target_total / s)


# class BeliefUpdateObserver(BaseAgent):
#     def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        
#         super().__init__(env.observer)

#         self.env = env
#         self.agent.name = "BeliefUpdateObserver"
#         self.agent.can_overlap = True

#         self.goals = env.goals
#         self.step = -1
#         self.pos = env.observer.pos
#         self.dir = env.observer.dir

#         if init_goal_belief:
#             self.goal_belief = init_goal_belief
#         else:
#             self.goal_belief = { g:1/len(self.goals) for g in self.goals}

#         if init_actor_belief:
#             self.actor_belief = init_actor_belief
#         else:
#             self.actor_belief = {g: set_uniform_prob(env.base_grid, len(Color) + 1, self.goal_belief[g]) for g in self.goals}

#         self.dist_matrix = self.compute_pairwise_distances()


#     def compute_pairwise_distances(self):
#         """
#         Compute all pairwise distances from each state (position and direction) to the goal locations using BFS.
#         """
#         #free_cells = np.argwhere(self.env.base_grid == 0)
#         free_cells = np.argwhere(self.env.base_grid != 2)
        
#         num_cells = len(free_cells)
#         num_directions = 4  # Number of possible directions (east, south, west, north)
#         num_states = num_cells * num_directions

#         cell_to_index = {tuple(cell): idx for idx, cell in enumerate(free_cells)}
#         # Initialize distance matrix for distances to goal locations
#         dist_matrix = {goal: np.full(num_states, np.inf) for goal in self.goals}
#         for goal in self.goals:
#             goal_idx = cell_to_index[tuple(goal)]
#             queue = deque([(goal_idx, dir, 0) for dir in range(num_directions)])  # (cell_index, direction, distance)
#             visited = set()

#             while queue:
#                 current_idx, current_dir, current_dist = queue.popleft()
#                 state = (current_idx, current_dir)
#                 if state in visited:
#                     continue
#                 visited.add(state)

#                 dist_matrix[goal][current_idx * num_directions + current_dir] = current_dist

#                 pos_state = (free_cells[current_idx], current_dir)
#                 for action, next_pos_state in get_reverse_successor(self.env, pos_state):
#                     next_pos, next_dir = next_pos_state
#                     if tuple(next_pos) in cell_to_index:
#                         next_idx = cell_to_index[tuple(next_pos)]
#                         queue.append((next_idx, next_dir, current_dist + 1))

#         adjusted_dist_matrix = dict()

#         for i in range(num_states):
#             for goal in self.goals:
#                 cell = free_cells[i // num_directions]
#                 pos_state = (tuple(cell), i % num_directions)
#                 adjusted_dist_matrix[(pos_state, tuple(goal))] = dist_matrix[goal][i]

#         return adjusted_dist_matrix
    

        
#     def compute_action(self, obs):
        
#         self.step += 1
#         self.pos = obs["observer_pos"]
#         self.dir = obs["observer_dir"]
#         self.update_belief(obs) 
#         self.update_goal_belief() 
#         # # update the belief based on current observation, each entry is the joint prob P(state, goal, obs history)

#         # self.update_goal_belief() 
#         # # update the goal belief based on the belief of the observer, each entry is the conditional prob P(goal|obs history)
#         # # assume goal directed behavior, predict next step belief based on current belief
#         self.actor_belief = update_actor_belief_4d(self.actor_belief, self.goals, self.env, self.dist_matrix) 
#         # # update the actor belief based on the goal belief, each entry is the joint prob P(state, goal, obs history)
#         #self.render_and_save(f'belief_update_test/actor_belief_step_{self.step}.png', obs)

#         # #return self.greedy()

#         print(self.goal_belief)

#         return self.mcts()

        
#     def mcts(self, iterations = 100, exploration_weight = 1):
#         start_pos_state = (self.pos, self.dir)
#         start_actor_belief = deepcopy(self.actor_belief)
#         start_goal_belief = deepcopy(self.goal_belief)
#         root = MCTSNode(self.agent, start_pos_state, start_actor_belief, start_goal_belief, self.env, self.dist_matrix)
        
#         for _ in range(iterations):
#             node = root
#             while not node.is_terminal() and node.is_fully_expanded():
#                 node = node.best_child(exploration_weight)
            
#             if not node.is_terminal():
#                 node = node.expand()
#             result = node.rollout()
#             node.backpropagate(result)

#         return root.best_child(0).action

        

#     def render_and_save(self, filename, obs):
#         """
#         Render the environment and save the visualization.
        
#         Parameters:
#         filename (str): The name of the file to save the visualization.
#         obs (dict): The observation dictionary containing the observer and goal positions.
#         """
#         os.makedirs(os.path.dirname(filename), exist_ok=True)

#         total_belief = np.zeros_like(next(iter(self.actor_belief.values())))
#         for goal, belief in self.actor_belief.items():
#             total_belief += belief

#         belief_sum = np.sum(total_belief, axis=2)
#         log_belief_sum = np.log(belief_sum + 1e-10)
#         vmin = np.min(log_belief_sum)
#         vmax = np.max(log_belief_sum)
        
#         plt.imshow(log_belief_sum, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
#         plt.colorbar()
#         goal_colors = ['yellow', 'green', 'cyan', 'magenta', 'orange']
#         goal_probs = [self.goal_belief[goal] for goal in self.goals]
#         goal_text = '\n'.join([f'Goal {i+1} ({goal_colors[i % len(goal_colors)]}): {prob:.2f}' for i, prob in enumerate(goal_probs)])
#         plt.title(goal_text)

#         # Overlay obstacles
#         obstacles = np.where(self.env.base_grid == 2)
#         plt.scatter(obstacles[1], obstacles[0], c='black', marker='s', label='Obstacle')

#         # Overlay observer position
#         observer_pos = obs["observer_pos"]
#         plt.scatter(observer_pos[1], observer_pos[0], c='blue', marker='o', label='Observer')

#         # Overlay goal positions
        
#         for i, goal in enumerate(self.goals):
#             plt.scatter(goal[1], goal[0], c=goal_colors[i % len(goal_colors)], marker='*', label='Goal')

#         # Overlay target position if observed
#         if "target_pos" in obs:
#             target_pos = obs["target_pos"]
#             plt.scatter(target_pos[1], target_pos[0], c='red', marker='x', label='Target')


#         plt.savefig(filename)
#         plt.close()
    
    
#     def update_belief(self, obs):
#         """
#         Update the 4D belief (W,H,dir,num_keys) for each goal.
    
#         - If target is in view: collapse to the observed (x,y,dir,tcarry) by concentrating
#           ALL mass on that single state.
#         - If target is NOT in view: zero out all states falling within current FoV cells.
#         """
#         # --- convenience: unseen index for layer 0 ---
#         unseen_idx = Type.unseen.to_index()
    
#         # --- check visibility ---
#         target_in_view = ("target_pos" in obs) or (self.pos == self.env.target.pos)
    
#         if target_in_view:
#             print(self.step); print(self.env.target.carrying); print("in view")
#             tx, ty = self.env.target.pos
#             tdir   = self.env.target.dir  # 0..3 (E,S,W,N)
#             tcarry = carrying_to_idx(self.env.target.carrying)  # -> 0..num_keys-1
    
#             for goal in self.goals:
#                 prior_total = float(self.actor_belief[goal].sum())
#                 new_actor_belief = np.zeros_like(self.actor_belief[goal])
#                 new_actor_belief[tx, ty, tdir, tcarry] = prior_total
#                 self.actor_belief[goal] = new_actor_belief
    
#         else:
#             print(self.step)
#             print("not in view")
#             local_img = obs['image'] 
#             unseen_idx = Type.unseen.to_index()
        
#             # 2) 可见性掩码（FoV 内可见 = True）
#             vis_local = (local_img[..., 0] != unseen_idx)   # (view, view) 的 bool
        
#             # 3) 构造世界坐标高亮掩码
#             highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)
        
#             f_vec = self.agent.state.dir.to_vec()
#             r_vec = np.array((-f_vec[1], f_vec[0]))
#             top_left = (
#                 self.agent.state.pos
#                 + f_vec * (self.agent.view_size - 1)
#                 - r_vec * (self.agent.view_size // 2)
#             )
        
#             for vis_j in range(self.agent.view_size):
#                 for vis_i in range(self.agent.view_size):
#                     if not vis_local[vis_i, vis_j]:
#                         continue
#                     abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
#                     if 0 <= abs_i < self.env.width and 0 <= abs_j < self.env.height:
#                         highlight_mask[abs_i, abs_j] = True
        
#             # 4) 用广播一次性清零 (x,y,*,*)；之后做质量守恒
#             mask4 = (~highlight_mask)[..., None, None]  # (W,H,1,1)
#             for goal in self.goals:
#                 before_total = float(self.actor_belief[goal].sum())
#                 self.actor_belief[goal] *= mask4
#                 after_total = float(self.actor_belief[goal].sum())
#                 if after_total > 0 and before_total > 0 and abs(after_total - before_total) > 1e-12:
#                     self.actor_belief[goal] = renormalize_belief(self.actor_belief[goal], target_total=before_total)
                    
    # def update_belief(self, obs):
    #     #print(self.actor_belief)
    #     """
    #     Update the belief of the observer based on the observed FoV.
        
    #     Parameters:
    #     FoV (np.array): The field of view of the observer.
    #     pos (tuple): The position of the actor or None.
    #     """
    #     # Update the belief of the observer based on the observed FoV
    #     if "target_pos" in obs or self.pos == self.env.target.pos:
    #         print(self.step)
    #         print(self.env.target.carrying)
    #         print("in view")
    #         target_pos = self.env.target.pos
    #         target_dir = self.env.target.dir # 0-3 denote east south west north respectively
            
            
    #         for goal in self.goals:
    #             new_actor_belief = np.zeros_like(self.actor_belief[goal])
    #             new_actor_belief[tuple(target_pos)][target_dir] = self.actor_belief[goal][tuple(target_pos)][target_dir]
    #             self.actor_belief[goal] = new_actor_belief
                

    #     else:
    #         print(self.step)
    #         print("not in view")
   
    #         obs_shape = self.agent.observation_space['image'].shape[:-1]
    #         vis_mask = np.zeros_like(obs_shape, dtype=bool)
    #         vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer
  

    #         highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)


    #         # of the agent's view area
    #         f_vec = self.agent.state.dir.to_vec()
    #         r_vec = np.array((-f_vec[1], f_vec[0]))
    #         top_left = (
    #             self.agent.state.pos
    #             + f_vec * (self.agent.view_size - 1)
    #             - r_vec * (self.agent.view_size // 2)
    #         )

    #         # For each cell in the visibility mask
    #         for vis_j in range(0, self.agent.view_size):
    #             for vis_i in range(0, self.agent.view_size):
    #                 # If this cell is not visible, don't highlight it
    #                 if not vis_mask[vis_i, vis_j]:
    #                     continue

    #                 # Compute the world coordinates of this cell
    #                 abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

    #                 if abs_i < 0 or abs_i >= self.env.width:
    #                     continue
    #                 if abs_j < 0 or abs_j >= self.env.height:
    #                     continue

    #                 # Mark this cell to be highlighted
    #                 highlight_mask[abs_i, abs_j] = True
    #         # highlight_mask = obs['fov']
    #         for goal in self.goals:
    #             for cell in np.argwhere(highlight_mask == 1):
    #                 self.actor_belief[goal][tuple(cell)] = 0
    #             #print(self.actor_belief[goal])
               
               
#     def update_goal_belief(self):
#         """
#         Update the belief of the observer based on the observed FoV.
        
#         Parameters:
#         FoV (np.array): The field of view of the observer.
#         pos (tuple): The position of the actor or None.
#         """
#         # Update the belief of the observer based on the observed FoV
#         for goal in self.goals:
#             self.goal_belief[goal] = np.sum(self.actor_belief[goal])
#             print(self.goal_belief[goal])

#         total = sum(self.goal_belief.values())
#         if total == 0:
#             print("should not happen,1")
#             print(self.goal_belief)
#             for goal in self.goals:
#                 print(np.where(self.actor_belief[goal]>0))
#             input()
#         for goal in self.goals:
#             self.goal_belief[goal] /= total
#             #print(goal,self.goal_belief[goal])



# def update_actor_belief_4d(actor_belief, goals, env, dist_matrix, beta=BETA):
#     """
#     actor_belief[goal] shape: (W, H, D, C)
#       - W,H: 空间
#       - D: 朝向数 (例如 4)
#       - C: 携带状态数 (例如: 0=空手, 1=红钥匙, 2=蓝钥匙, ...)
#     dist_matrix[(succ_key, goal)]:
#       - 若你原先用的是 (pos,dir) 作为 key，这里仍可继续用 ( (i,j), d )；携带维通常不参与距离。
#     """
#     new_actor_belief = {goal: np.zeros_like(actor_belief[goal]) for goal in goals}

#     # ---- 小工具：将 (pos,dir) 后继抬升为 4D 状态 (pos,dir,carry) ----
#     def lift_successor_to_4d(env, pos, dir_idx, carry_idx, succ_pos, succ_dir):
#         """
#         依据当前 carry_idx 与地面物品，猜测 next_carry。
#         这里给出最小可用规则：
#           - 若空手且 succ_pos 有钥匙 -> 拿起对应颜色的钥匙
#           - 其他情况：携带不变
#         你可以在此加入 drop/use/open-door 等更复杂的状态转移。
#         """
#         next_carry = carry_idx

#         # 示例占位：检查格子里的物体（按你项目的 API 改）
#         cell_obj = getattr(env, "grid", None)
#         item = None
#         if cell_obj is not None and hasattr(env, "get_item_at"):
#             item = env.get_item_at(tuple(succ_pos))  # 你自己的查询接口
#         # 若无接口，就根据你的 env 表示来改写

#         # 简化规则：空手遇钥匙 -> 拿起
#         if carry_idx == 0 and item is not None and getattr(item, "is_key", False):
#             next_carry = key_color_to_idx(getattr(item, "color", None))
#             # 若你允许踩过去就自动 pickup，保留；否则删掉

#         return succ_pos, succ_dir, next_carry

#     # ---- 主循环：对每个 goal，遍历所有非零 (i,j,d,c) 质量 ----
#     for goal in goals:
#         belief_4d = actor_belief[goal]  # (W,H,D,C)
#         nz = np.argwhere(belief_4d > 0)

#         for cell in nz:
#             i, j, d, c = cell
#             pos = (i, j)
#             dir_idx = d
#             carry_idx = c
#             prob = float(belief_4d[i, j, d, c])
#             if prob <= 0:
#                 continue

#             # --- 生成 4D 后继 ---
#             successors_4d = []

#             # 若已在 goal 位置：只允许 stay（保持一致性）
#             if pos[0] == goal[0] and pos[1] == goal[1]:
#                 # 你自己的 stay 实现：这里假设 get_successor 里有 Action.stay
#                 stay_list = [(Action.stay, (pos, dir_idx))]
#                 for action, (npos, ndir) in stay_list:
#                     npos, ndir, ncarry = lift_successor_to_4d(env, pos, dir_idx, carry_idx, npos, ndir)
#                     successors_4d.append((action, (npos, ndir, ncarry)))
#             else:
#                 # 你现有的 2D/3D 后继
#                 base_succ = get_successor(env, (pos, dir_idx))  # -> list[(action, (npos, ndir))]
#                 for action, (npos, ndir) in base_succ:
#                     npos, ndir, ncarry = lift_successor_to_4d(env, pos, dir_idx, carry_idx, npos, ndir)
#                     successors_4d.append((action, (npos, ndir, ncarry)))

#             # --- 计算转移权重（用距离+1 的软最优/softmin，与你原版一致） ---
#             tran_probs = {}
#             for _action, (npos, ndir, ncarry) in successors_4d:
#                 # 距离矩阵的 key：保持与你已有的一致性
#                 # 常见做法：只用 (npos, ndir) 参与距离
#                 key = (((npos[0], npos[1]), ndir), goal)

#                 if key in dist_matrix:
#                     cost = 1 + dist_matrix[key]
#                     w = math.exp(-beta * cost)
#                 else:
#                     # 若你的 dist_matrix 只用 (npos) 而非 (npos,ndir)，可以改成：
#                     # key2 = ((npos[0], npos[1]), goal)
#                     # w = math.exp(-beta * (1 + dist_matrix.get(key2, LARGE)))
#                     print("warn: dist key not found:", key)
#                     w = 0.0

#                 tran_probs[(npos[0], npos[1], ndir, ncarry)] = w

#             total_w = sum(tran_probs.values())
#             if total_w > 0:
#                 # 归一化
#                 for k in tran_probs:
#                     tran_probs[k] /= total_w

#                 # 累加到新 belief
#                 for _action, (npos, ndir, ncarry) in successors_4d:
#                     idx = (npos[0], npos[1], ndir, ncarry)
#                     w = tran_probs.get(idx, 0.0)
#                     if w > 0:
#                         new_actor_belief[goal][idx] += prob * w
#             else:
#                 # 没有有效后继（或全部被阻断），把质量原位保留为“自环”以防质量流失
#                 new_actor_belief[goal][i, j, d, c] += prob

#     return new_actor_belief

# def set_uniform_prob(grid, num_keys=7, total_prob=1.0):
#     """
#     Set a uniform probability for all free cells in the grid with carrying dimension.

#     Parameters:
#     grid (np.array): The grid to be analyzed (2D array).
#     num_keys (int): Number of possible carrying states (e.g., 1 = only 'empty',
#                     2 = empty + red key, 3 = empty + red + blue ...).
#     total_prob (float): Total probability mass to distribute.

#     Returns:
#     np.array: A (W,H,dir,num_keys) array with uniform probabilities.
#     """
#     dir = 4
#     free_cells = np.argwhere(grid != 2)   # 假设 2 代表墙
#     num_free_cells = len(free_cells)
#     total_states = num_free_cells * dir * num_keys

#     uniform_prob = total_prob / total_states if total_states > 0 else 0

#     prob_grid = np.zeros((*grid.shape, dir, num_keys), dtype=float)
#     for cell in free_cells:
#         x, y = cell
#         prob_grid[x, y, :, :] = uniform_prob

#     return prob_grid


# # def set_uniform_prob(grid, total_prob = 1):
# #     """
# #     Set a uniform probability for all free cells in the grid.
    
# #     Parameters:
# #     grid (np.array): The grid to be analyzed.
    
# #     Returns:
# #     np.array: A grid with uniform probabilities for all free cells.
# #     """
# #     dir = 4
# #     free_cells = np.argwhere(grid != 2)
# #     num_free_cells = len(free_cells)
# #     uniform_prob = total_prob / (num_free_cells * dir) if num_free_cells > 0 else 0

# #     prob_grid = np.zeros((*grid.shape, dir), dtype=float)
# #     for cell in free_cells:
# #         prob_grid[tuple(cell)] = uniform_prob

# #     return prob_grid



# class MCTSNode:
#     def __init__(self, agent, pos_state, actor_belief, goal_belief, env, dist_matrix, action = None, parent=None):
#         self.agent = agent
#         self.dist_matrix = dist_matrix
#         self.pos_state = pos_state  # The current game state
#         self.parent = parent  # Parent node
#         self.action = action  # Action that led to this node
#         self.actor_belief = {goal: actor_belief[goal] for goal in actor_belief}
#         self.goal_belief = {goal: goal_belief[goal] for goal in goal_belief}
#         self.env = env
#         self.children = []  # List of child nodes
#         self.visits = 0  # Number of times node has been visited
#         self.value = 0  # Total value of the node

#     def is_fully_expanded(self):
#         return len(self.children) == len(get_obs_successor(self.env, self.pos_state))

#     def best_child(self, exploration_weight=1.0):
#         """Selects the best child using UCT for decision nodes and expectation for chance nodes."""

#         return max(
#             self.children, 
#             key=lambda child: (child.value / (child.visits + 1e-6)) + 
#                               exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
#         )

#     def expand(self):
#         """Expands the node by adding a new child node."""
#         tried_moves = {child.action for child in self.children}
#         possible_succs = get_obs_successor(self.env, self.pos_state)

#         for action, next_pos_state in possible_succs:
#             if action not in tried_moves:
#                 g = self.sample_goal()
 
#                 actor_pos_state = self.sample_from_4d_belief(self.actor_belief[g])

#                 new_actor_belief = self.update_actor_belief_from_obs(actor_pos_state, next_pos_state)

#                 new_goal_belief = self.update_goal_belief(new_actor_belief)
      
#                 # goal directed update of the actor belief
#                 new_actor_belief = update_actor_belief_4d(new_actor_belief, self.env.goals, self.env, self.dist_matrix)


#                 new_node = MCTSNode(self.agent, next_pos_state, new_actor_belief, new_goal_belief, self.env, self.dist_matrix, action = action, parent=self)
#                 self.children.append(new_node)
#                 return new_node

#     def sample_goal(self):
#         """Samples a goal based on the probability distribution in self.goal_belief."""
#         goals = list(self.goal_belief.keys())  # Extract possible goals
#         probabilities = np.array(list(self.goal_belief.values()))  # Extract probabilities

#         if probabilities.sum() == 0:
#             print("should not happen,4")
#             print(self.goal_belief)
#             input()
#         # Normalize probabilities to ensure they sum to 1
#         probabilities /= probabilities.sum()

#         # Sample a goal based on the normalized probability distribution
#         sampled_goal = np.random.choice(len(goals), p=probabilities)
#         return goals[sampled_goal]
            
#     def update_goal_belief(self, actor_belief):
#         new_goal_belief = {}
#         for goal in self.goal_belief:
#             new_goal_belief[goal] = np.sum(actor_belief[goal])

#         total = sum(new_goal_belief.values())
#         for goal in self.goal_belief:
#             new_goal_belief[goal] /= total

#         return new_goal_belief
            
#     def update_actor_belief_from_obs(self, actor_pos_state, observer_pos_state, obs=None):
#         """
#         actor_pos_state: (ax, ay, a_dir[, a_carry?])  —— 本函数只用到 (ax, ay, a_dir)
#         observer_pos_state: (ox, oy, o_dir_index)
#         obs: 可选，字典或 ndarray；若提供字典则用 obs['image'] 作为本轮观测
#         """
#         # 新容器（与旧 belief 同形）
#         new_actor_belief = {g: np.zeros_like(self.actor_belief[g]) for g in self.actor_belief}
    
#         # 解析参与者状态（只用位置与朝向）
#         ax, ay, a_dir = actor_pos_state[:3]
#         ox, oy = observer_pos_state[0]
#         o_dir_idx = observer_pos_state[1]
#         observer_dir = Direction(o_dir_idx)
    
#         # --- 取本轮可见性图（FoV 内可见=True） ---
#         if obs is None:
#             local_img = self.env.gen_obs()[0]['image']   # 注意：MCTS里退回到 gen_obs()
#         else:
#             local_img = obs['image'] if isinstance(obs, dict) else obs
    
#         unseen_idx = Type.unseen.to_index()
#         vis_local = (local_img[..., 0] != unseen_idx)    # (view, view) 的布尔
    
#         # --- 把本地 FoV 投影到世界坐标，得到 highlight_mask (W,H) ---
#         W, H = self.env.width, self.env.height
#         highlight_mask = np.zeros((W, H), dtype=bool)
    
#         f_vec = observer_dir.to_vec()
#         r_vec = np.array((-f_vec[1], f_vec[0]))
#         top_left = (
#             np.array([ox, oy])
#             + f_vec * (self.agent.view_size - 1)
#             - r_vec * (self.agent.view_size // 2)
#         )
    
#         for vis_j in range(self.agent.view_size):
#             for vis_i in range(self.agent.view_size):
#                 if not vis_local[vis_i, vis_j]:
#                     continue
#                 abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
#                 if 0 <= abs_i < W and 0 <= abs_j < H:
#                     highlight_mask[abs_i, abs_j] = True   # True 表示当前步“可见区”
    
#         # --- 根据是否“看见 actor”选择更新策略 ---
#         actor_in_fov = highlight_mask[ax, ay]
    
#         for g in self.goal_belief:
#             old = self.actor_belief[g]          # shape: (W,H,D) 或 (W,H,D,C)
#             before_total = float(old.sum())
#             if before_total <= 0:
#                 continue
    
#             # 动态构造广播掩码形状：(W,H,1[,1])
#             tail_dims = old.ndim - 2
#             keep_mask = (~highlight_mask)[..., *([None] * tail_dims)]  # True=保留，False=清零
    
#             if actor_in_fov:
#                 # “正证据坍缩”：仅保留 (ax, ay, a_dir, :) 的质量，其余清零
#                 tmp = np.zeros_like(old)
#                 if old.ndim == 3:
#                     # (W,H,D)
#                     tmp[ax, ay, a_dir] = old[ax, ay, a_dir]
#                 else:
#                     # (W,H,D,C) —— 保留该格、该朝向的所有携带态
#                     tmp[ax, ay, a_dir, :] = old[ax, ay, a_dir, :]
#                 new = tmp
#             else:
#                 # “负证据排除”：把当前可见区 (x,y) 的所有 (dir,carry) 组合清零，其余保持
#                 new = old * keep_mask
    
#             # 质量守恒：把总量拉回更新前
#             after_total = float(new.sum())
#             if after_total > 0 and abs(after_total - before_total) > 1e-12:
#                 new *= (before_total / after_total)
    
#             new_actor_belief[g] = new
    
#         return new_actor_belief

        
#     def is_terminal(self):
#         return self.env.is_done()

#     def rollout(self):
#         """Simulates the game to the end from the current state and returns the result."""
#         return -compute_entropy(self.goal_belief)


#     def backpropagate(self, result, action_penalty=0):
#         """Updates the tree nodes based on the result of the rollout."""
#         self.visits += 1
#         self.value += result - action_penalty
#         if self.parent:
#             self.parent.backpropagate(result)

    

#     def sample_from_4d_belief(self, actor_belief):
#         """
#         Samples a state (x, y, dir, carry) from the 4D belief map using probability distribution.
    
#         actor_belief: np.ndarray of shape (W, H, D, C)
#         """
#         W, H, D, C = actor_belief.shape
    
#         # Flatten成1D
#         flat = np.copy(actor_belief).ravel()
#         total = flat.sum()
#         if total == 0:
#             print("Warning: total belief mass is 0")
#             print(actor_belief)
#             input()
#         flat /= total  # 归一化成概率分布
    
#         # 按分布采样索引
#         sampled_index = np.random.choice(len(flat), p=flat)
    
#         # 把1D索引还原成4D坐标
#         i, rem = divmod(sampled_index, H * D * C)
#         j, rem = divmod(rem, D * C)
#         d, c = divmod(rem, C)
    
#         return (i, j, d, c)  # 返回 (x, y, dir, carry)

# def compute_entropy(goal_belief):
#     """Computes the Shannon entropy of the goal belief distribution."""
#     probabilities = np.array(list(goal_belief.values()))
#     # probabilities = goal_belief
    
#     # Ensure the probabilities sum to 1
#     probabilities /= probabilities.sum()
    
#     # Compute entropy, avoiding log(0) by filtering out zero probabilities
#     entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Small offset to avoid log(0)
    
    return entropy

from .base import BaseAgent
from ..new_astar import astar, execute_action, get_obs_successor, get_reverse_successor
from ..lock_astar import astar_key,astar_open

import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction,OBJECT_TO_IDX,COLOR_TO_IDX,Type
from multigrid.core.actions import Action
import random
import os
from collections import deque,defaultdict
from copy import deepcopy,copy
from typing import Any, Dict, List, Optional, Tuple

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])




class GreedyObserver(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        super().__init__(env.observer)
        self.env = env # width x height y get(x,y)
        self.hallway_col = env.num_cols // 2
        self.abs = env.abs
        self.goals = env.goals
        self.goal_rooms = [_rid_from_xy(self.env,x,y,self.hallway_col) for x,y in self.goals]
        #print(self.goal_rooms)
        self.high_level_length = {}
        for goal_room in self.goal_rooms:
            high_level_plan = _plan_onekey_persist_open(self.abs, ("HALL",), goal_room, held = None)
            self.high_level_length[goal_room] = (len(high_level_plan),high_level_plan)
        ###——————————————————————————————————————————————————————————————————————————————————————

        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)


    def compute_action(self, obs):

        obs_held = self.env.observer.carrying
        obs_pos = self.env.observer.pos
        obs_dir = self.env.observer.dir

        door_list = self.abs["edges"]
        subgoal_expected_payoff = {}

        if obs_held:
            for eid in range(len(door_list)):
                door = door_list[eid]
                if door["locked"] == True and obs_held.color == door["color"]:
                    nearest_dist = manhattan_distance(obs_pos, door["pos"])
                    subgoal_expected_payoff[eid] = nearest_dist
                    
            if subgoal_expected_payoff != {}:
                best_eid, best_info = min(subgoal_expected_payoff.items(),key=lambda kv: kv[1])
                door = door_list[best_eid]
                path = astar_open((obs_pos,obs_dir),door["pos"],self.env,self.hidden_cost, version = True)
                return path[1][0]
                
        subgoal_expected_payoff = {}
        for eid in range(len(door_list)):
            door = door_list[eid]
            if door["locked"] == True:
                all_keys = find_all_keys(self.abs["rooms"],door["color"])
                nearest = min(all_keys, key=lambda k: manhattan_distance(k['pos'], obs_pos) 
                                + manhattan_distance(k['pos'],door["pos"]))
                
                nearest_dist = (manhattan_distance(nearest['pos'], obs_pos)
                    + manhattan_distance(nearest['pos'], door["pos"]))
                
                subgoal_expected_payoff[eid] = {"nearest_dist":nearest_dist,"nearest":nearest}
                
        best_eid, best_info = min(subgoal_expected_payoff.items(),key=lambda kv: kv[1]["nearest_dist"])
        nearest = best_info["nearest"]
        path = astar_key((obs_pos,obs_dir),nearest['pos'],self.env,self.hidden_cost,agent_idx = 0, version = True)
       
        return path[1][0]




class Observer(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        super().__init__(env.observer)
        self.env = env # width x height y get(x,y)
        self.hallway_col = env.num_cols // 2
        self.abs = env.abs
        self.goals = env.goals
        self.goal_rooms = [_rid_from_xy(self.env,x,y,self.hallway_col) for x,y in self.goals]
        #print(self.goal_rooms)
        self.high_level_length = {}
        for goal_room in self.goal_rooms:
            high_level_plan = _plan_onekey_persist_open(self.abs, ("HALL",), goal_room, held = None)
            self.high_level_length[goal_room] = (len(high_level_plan),high_level_plan)
        self.finished_plan = -1

        self.past_plans = []
        self.dist_matrix = self.compute_pairwise_distances()

        self.goal_length = {}
        for goal in self.goals:
            goal_room = _rid_from_xy(self.env,*goal,self.hallway_col) 
            self.goal_length[goal_room] = 0
            pos = self.env.target.pos
            dir = self.env.target.dir
            for plan in self.high_level_length[goal_room][1]:
                self.goal_length[goal_room] += self.dist_matrix[(pos,dir),plan['pos']['value']]
                pos = plan['pos']['value']
            #self.goal_length[goal_room] += self.dist_matrix[(pos,dir),goal]

        ###——————————————————————————————————————————————————————————————————————————————————————

        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)

        for goal_room in self.goal_rooms:
            print(goal_room, self.goal_length[goal_room])
        
    def length_compute(self,pos,dir,plan_list):
        total_length = 0
        for plan in plan_list:
            total_length += self.dist_matrix[(pos,dir),plan['pos']['value']]
            pos = plan['pos']['value']
        return total_length

    def select_best_action(self,scores):
        obs_held = self.env.observer.carrying
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for k, v in items:
            act, pos = k
            if obs_held is not None and act == "pickup":
                continue
            if act == "open":
                if obs_held is None:
                    continue
                print(obs_held.color , self.env.grid.get(*pos).color)
                if obs_held.color != self.env.grid.get(*pos).color:
                    continue
            return k

        for k, v in items:
            act, pos = k
            return k

    def compute_action(self, obs):
        pos = self.env.target.pos
        dir = self.env.target.dir
        held = self.env.target.carrying
        obs_held = self.env.observer.carrying
        obs_pos = self.env.observer.pos
        obs_dir = self.env.observer.dir
        print(self.goal_rooms)
        goal_recognition = self.goal_recognition()
        print(goal_recognition)

        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        door_list = self.abs["edges"]
        
        subgoal_expected_payoff = {}
        for eid in range(len(door_list)):
            subgoal_expected_payoff[eid] = 0
            door = door_list[eid]
            if door["locked"] == True:
                for goal_room_idx in range(len(self.goal_rooms)):
                    goal_room = self.goal_rooms[goal_room_idx]
                    mask = _initial_open_mask(self.abs)
                    mask |= (1 << eid)
                    plan_list = _plan_onekey_persist_open(self.abs,start_rid, goal_room, held = held,open_mask = mask)
                    payoff = (self.goal_length[goal_room] - self.length_compute(pos,dir,plan_list))/(self.goal_length[goal_room]+0.01)
                    excepted_payoff = goal_recognition[goal_room_idx] * payoff
                    subgoal_expected_payoff[eid] += excepted_payoff

        print(subgoal_expected_payoff)
        max_door = max(subgoal_expected_payoff, key=subgoal_expected_payoff.get)

        
        door = self.abs["edges"][max_door]
        print(door["color"],door["pos"])
        
        path = []
        if obs_held:
            #print(obs_held.color,door["color"])
            if obs_held.color == door["color"]:
                path = astar_open((obs_pos,obs_dir),door["pos"],self.env,self.hidden_cost, version = True)
                return path[1][0]

        all_keys = find_all_keys(self.abs["rooms"],door["color"])
        nearest = min(all_keys, key=lambda k: manhattan_distance(k['pos'], obs_pos) + manhattan_distance(k['pos'],door["pos"]))
        path = astar_key((obs_pos,obs_dir),nearest['pos'],self.env,self.hidden_cost,agent_idx =0, version = True)
        return path[1][0]

    def goal_recognition(self):
        pos = self.env.target.pos
        dir = self.env.target.dir
        held = self.env.target.carrying
        self.current_palns = []
        # update door state
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
        #compute the high level plan
        for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
                                                             held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
            self.current_palns.append(plan)

        # check the plan has changed and inital the subgoal
        if self.past_plans != self.current_palns:
            self.finished_plan += 1
            self.steps = 0
            self.subgoal_dist = {}
            for plan in self.current_palns:
                self.subgoal_dist[plan['pos']['value']] = self.dist_matrix[(pos,dir),plan['pos']['value']]
        else:
            self.steps += 1
        
        self.past_plans = self.current_palns
        #compute the high level plan
        subgoal_pro = []
        subgoal_final_pro = []
        for plan in self.current_palns:
            start_rid = _rid_from_xy(self.env,*plan['pos']['value'],self.hallway_col)
            
            subgoal_dist = self.steps + self.dist_matrix[(pos,dir),plan['pos']['value']]
            subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
            diff = subgoal_dist - subgoal_optimal
            subgoal_pro.append(diff)
            softmax = []
            
            if plan['type'] == 'pickup':
                for goal_room in self.goal_rooms:
                    current_high_length  = self.finished_plan + len(_plan_onekey_persist_open(self.abs, start_rid,
                                                                                              goal_room, held = plan['key'])) + 1
                    var = current_high_length - self.high_level_length[goal_room][0]
                    softmax.append(var)
            elif plan['type'] == 'open':
                for goal_room in self.goal_rooms:
                    mask = _initial_open_mask(self.abs)
                    mask |= (1 << plan['eid'])
                    current_high_length = self.finished_plan + len(_plan_onekey_persist_open(self.abs,
                                                                                             start_rid, goal_room, held = held,
                                                                                             open_mask = mask)) + 1
                    var = current_high_length - self.high_level_length[goal_room][0]
                    softmax.append(var)
            else:
                for goal_room in self.goal_rooms:
                    if goal_room == plan["room"]:
                        softmax.append(0.0)
                    else:
                        softmax.append(20.0)
            subgoal_final_pro.append(softmax_prob(softmax))
        
        
        weights = softmax_prob(subgoal_pro)
        return weighted_goal_probabilities(weights, subgoal_final_pro)

    
    def compute_pairwise_distances(self):
        """
        Compute shortest distances from every state (cell + direction) to every free cell.
        Return: dict with keys ((pos, dir), target_cell) -> distance
        """
        # 1) 可走格子与索引
        free_cells = np.argwhere(self.env.base_grid != 2)
        num_cells = len(free_cells)
        num_directions = 4
        num_states = num_cells * num_directions
        cell_to_index = {tuple(cell): idx for idx, cell in enumerate(free_cells)}
    
        # 2) 距离矩阵：行 = 目标格子索引 (target_cell_idx)，列 = 状态索引 (cell_idx * 4 + dir)
        #    初始为 inf
        dist_matrix = np.full((num_cells, num_states), np.inf)
    
        # 3) 对每一个 free_cell 作为“目标”，做一次反向 BFS
        for tgt_idx, tgt_cell in enumerate(free_cells):
            queue = deque([(cell_to_index[tuple(tgt_cell)], d, 0) for d in range(num_directions)])
            visited = set()
    
            while queue:
                current_idx, current_dir, current_dist = queue.popleft()
                state = (current_idx, current_dir)
                if state in visited:
                    continue
                visited.add(state)
    
                # 写入该目标格子对应的距离
                dist_matrix[tgt_idx, current_idx * num_directions + current_dir] = current_dist
                # 从当前状态反推上一个状态（谁能走到我）
                pos_state = (free_cells[current_idx], current_dir)
                for _action, next_pos_state in get_reverse_successor(self.env, pos_state):
                    next_pos, next_dir = next_pos_state
                    next_pos_t = tuple(next_pos)
                    if next_pos_t in cell_to_index:
                        next_idx = cell_to_index[next_pos_t]
                        queue.append((next_idx, next_dir, current_dist + 1))
    
        # 4) 打平为 ((pos, dir), target_cell) -> distance 的查表
        adjusted_dist_matrix = dict()
        for s in range(num_states):
            cell = free_cells[s // num_directions]
            dir_ = s % num_directions
            pos_state = (tuple(cell), dir_)
            for tgt_idx, tgt_cell in enumerate(free_cells):
                adjusted_dist_matrix[(pos_state, tuple(tgt_cell))] = dist_matrix[tgt_idx, s]
    
        return adjusted_dist_matrix

    # def compute_action(self, obs):
    #     pos = self.env.target.pos
    #     dir = self.env.target.dir
    #     held = self.env.target.carrying
    #     obs_held = self.env.observer.carrying
    #     obs_pos = self.env.observer.pos
    #     obs_dir = self.env.observer.dir
    #     self.current_palns = []
    #     print(self.goal_rooms)
        
    #     # update key and door state
    #     for room in self.abs["rooms"]:
    #         self.abs["rooms"][room]["keys"] = []
    #     for x in range(self.env.width):
    #         for y in range(self.env.height):
    #             obj = self.env.grid.get(x,y)
    #             if obj and obj.type == "key":
    #                 room = _rid_from_xy(self.env,x,y,self.hallway_col)
    #                 self.abs["rooms"][room]["keys"].append({'color':obj.color,'pos':(x,y)})  
    #     for door in self.abs["edges"]:
    #         obj = self.env.grid.get(*door["pos"])
    #         if obj:
    #             door["locked"] = (self.env.grid.get(*door["pos"]).state == "locked")
    #         else:
    #             door["locked"] = False
    #     # update plan state
    #     for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
    #                                                          held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
    #         self.current_palns.append(plan)
    #     # check the plan has changed and inital the subgoal
    #     if self.past_plans != self.current_palns:
    #         self.finished_plan += 1
    #         self.steps = 0
    #         self.subgoal_dist = {}
    #     else:
    #         self.steps += 1
    #     self.past_plans = self.current_palns
    #     start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
    #     door_list = self.abs["edges"]
    #     subgoal_expected_payoff = {}
    #     for eid in range(len(door_list)):
    #         subgoal_expected_payoff[eid] = 0
    #         door = door_list[eid]
    #         if door["locked"] == True:
    #             mask = _initial_open_mask(self.abs)
    #             mask |= (1 << eid)
    #             goal_recognition = self.goal_recognition(open_mask = mask)
    #             print(self.goal_rooms,goal_recognition)
    #             for goal_room_idx in range(len(self.goal_rooms)):
    #                 goal_room = self.goal_rooms[goal_room_idx]
    #                 plan_list = _plan_onekey_persist_open(self.abs,start_rid, goal_room, held = held,open_mask = mask)
    #                 payoff = (self.goal_length[goal_room] - self.length_compute(pos,dir,plan_list))/self.goal_length[goal_room]
    #                 excepted_payoff = goal_recognition[goal_room_idx] * payoff
    #                 subgoal_expected_payoff[eid] += excepted_payoff

    #     print(subgoal_expected_payoff)
    #     max_door = max(subgoal_expected_payoff, key=subgoal_expected_payoff.get)
    #     door = self.abs["edges"][max_door]
    #     print(door["color"],door["pos"])
    #     path = []
    #     if obs_held:
    #         #print(obs_held.color,door["color"])
    #         if obs_held.color == door["color"]:
    #             path = astar_open((obs_pos,obs_dir),door["pos"],self.env,self.hidden_cost, version = True)
    #             return path[1][0]

    #     all_keys = find_all_keys(self.abs["rooms"],door["color"])
    #     nearest = min(all_keys, key=lambda k: manhattan_distance(k['pos'], obs_pos) + manhattan_distance(k['pos'],door["pos"]))
    #     path = astar_key((obs_pos,obs_dir),nearest['pos'],self.env,self.hidden_cost,agent_idx =0, version = True)
    #     return path[1][0]
        
    # def goal_recognition(self,open_mask = None):
    #     pos = self.env.target.pos
    #     dir = self.env.target.dir
    #     held = self.env.target.carrying
    #     self.current_palns = []
    #     subgoal_pro = []
    #     subgoal_final_pro = []
    #     current_palns = []
    #     start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)

        
    #     for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
    #                                                          open_mask = open_mask,
    #                                                          held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
    #         self.subgoal_dist[plan['pos']['value']] = self.dist_matrix[(pos,dir),plan['pos']['value']]
    #         start_rid = _rid_from_xy(self.env,*plan['pos']['value'],self.hallway_col)
    #         subgoal_dist = self.steps + self.dist_matrix[(pos,dir),plan['pos']['value']]
    #         subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
    #         diff = subgoal_dist - subgoal_optimal
    #         subgoal_pro.append(diff)
    #         softmax = []
    #         if plan['type'] == 'pickup':
    #             for goal_room in self.goal_rooms:
    #                 current_high_length  = self.finished_plan + len(_plan_onekey_persist_open(self.abs, start_rid,
    #                                                                                           goal_room, held = plan['key'])) + 1
    #                 var = current_high_length - self.high_level_length[goal_room][0]
    #                 softmax.append(var)
    #         elif plan['type'] == 'open':
    #             for goal_room in self.goal_rooms:
    #                 if open_mask:
    #                     mask = open_mask
    #                 else:
    #                     mask = _initial_open_mask(self.abs)
    #                 mask |= (1 << plan['eid'])
    #                 current_high_length = self.finished_plan + len(_plan_onekey_persist_open(self.abs,
    #                                                                                          start_rid, goal_room, held = held,
    #                                                                                          open_mask = mask)) + 1
    #                 var = current_high_length - self.high_level_length[goal_room][0]
    #                 softmax.append(var)
    #         else:
    #             for goal_room in self.goal_rooms:
    #                 if goal_room == plan["room"]:
    #                     softmax.append(0.0)
    #                 else:
    #                     softmax.append(20.0)
    #         subgoal_final_pro.append(softmax_prob(softmax))
    #     weights = softmax_prob(subgoal_pro)
    #     return weighted_goal_probabilities(weights, subgoal_final_pro)

def weighted_goal_probabilities(weights, matrix):
    """
    Compute weighted probabilities over goals.

    Parameters
    ----------
    weights : list[float] or np.ndarray
        Length n, weight for each row.
    matrix : list[list[float]] or np.ndarray
        Shape (n, m), each row is a probability distribution over m goals.

    Returns
    -------
    np.ndarray
        Shape (m,), weighted probability distribution.
    """
    weights = np.array(weights, dtype=float)
    matrix = np.array(matrix, dtype=float)

    # 检查维度是否匹配
    assert matrix.shape[0] == len(weights), "weights 长度必须与 matrix 行数相等"

    # 做加权平均
    weighted_sum = np.dot(weights, matrix)

    # 归一化（避免数值误差）
    weighted_prob = weighted_sum / weighted_sum.sum()

    return weighted_prob

def softmax_prob(xs):
    exps = [math.exp(-x) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def current_task_options_onekey_persist_open(
    abs_graph: Dict[str, Any],
    rid: Any,                            # 当前所在房间 id（BFS 起点）
    held: Optional[str] = None,          # 当前手里拿的钥匙（可能是对象或颜色字符串）
    open_mask: Optional[int] = None,     # 已开启“门/边”的位掩码；None 则用初始掩码
    *,
    end_rid: Optional[Any] = None,       # 目标房间 rid（goal 所在房间）
    goal_pos = None,
    include_paths: bool = False,         # 是否附带到任务发生房间/门口的路径
) -> List[Dict[str, Any]]:
    """
    “单钥匙 + 持久开门”规则下，基于【当前就能通行】的连通域，列出可执行任务：
      - 在已可达区域的任意房间里的 'pickup'
      - 在已可达区域边界、且当前钥匙能开的未开之门的 'open'
      - 若 end_rid 在可达集合，则加入 'goal' 任务（最简，仅标注 type 与 room）

    注：不做开新门后的递归展开；仅考虑当前 open_mask 与 held。
    """
    rooms = abs_graph["rooms"]
    adj   = _build_neighbors(abs_graph)   # 邻接表：每条边 (nb, color, eid)

    # 统一 held：有些工程里 held 是对象（带 .color），有些是字符串
    if held is not None and hasattr(held, "color"):
        held = held.color
    held = _canon(held)

    # open_mask 默认值
    if open_mask is None:
        open_mask = _initial_open_mask(abs_graph)

    # ---------- 第一步：在“当前即可通行”的边上做 BFS，得到整块可达房间 ----------
    def edge_traversable(col, eid) -> bool:
        opened = ((open_mask >> eid) & 1)
        return bool(opened)

    prev_room: Dict[Any, Optional[Any]] = {rid: None}     # 房间 -> 上一个房间
    prev_edge: Dict[Any, Optional[int]] = {rid: None}     # 房间 -> 通过的 eid
    q = deque([rid])
    while q:
        r = q.popleft()
        for nb, col, eid in adj[r]:
            if edge_traversable(col, eid) and nb not in prev_room:
                prev_room[nb] = r
                prev_edge[nb] = eid
                q.append(nb)

    reachable_rooms = set(prev_room.keys())  # 含起点 rid

    # 辅助：回溯房间路径（如需要）
    def room_path_to(target_r):
        path = []
        cur = target_r
        while cur is not None:
            path.append(cur)
            cur = prev_room[cur]
        path.reverse()
        return path

    tasks: List[Dict[str, Any]] = []

    # ---------- 第二步：在“可达房间”里枚举 pickup ----------
    for r in reachable_rooms:
        rk = list(_iter_room_keys(rooms[r]))  # [(kcolor, kraw), ...]
        if not rk:
            continue

        seen_color = set()
        for kcolor, kraw in rk:
            kc = _canon(kcolor)
            seen_color.add(kc)

            # 空手可以捡任意；有钥匙仅能换不同颜色
            if (held is None) or (held != kc):
                kpos_type, kpos_val = _key_position(kraw)
                item = {
                    "type": "pickup",
                    "room": r,                         # 发生房间
                    "key":  kc,
                    "pos":  {"type": kpos_type, "value": kpos_val},
                }
                if include_paths:
                    p = room_path_to(r)
                    item["via_rooms"] = p             # 到达该房间的房间序列
                    item["via_steps"] = max(0, len(p) - 1)
                tasks.append(item)

    # ---------- 第三步：在“可达边界”上枚举 open（当前钥匙能开的、尚未开过的门） ----------
    seen_eid = set()  # 避免同一扇门从两侧重复加入
    for r in reachable_rooms:
        for nb, col, eid in adj[r]:
            if eid in seen_eid:
                continue
            ccol   = _canon(col)
            opened = ((open_mask >> eid) & 1)

            # 仅列出“未开 + 有颜色 + 颜色匹配当前钥匙”的门
            if (not opened) and (ccol is not None) and (ccol == held):
                # 注意：open 动作发生在 r 侧门口即可
                pos_info = _edge_position(abs_graph, eid)
                item = {
                    "type":  "open",
                    "eid":   eid,
                    "color": ccol,
                    "from":  r,
                    "to":    nb,
                    "pos":   pos_info
                }
                if include_paths:
                    p = room_path_to(r)
                    item["via_rooms"] = p             # 到达门口所在房间的路径
                    item["via_steps"] = max(0, len(p) - 1)
                tasks.append(item)
                seen_eid.add(eid)

    # ---------- 第四步：若 end_rid 在可达集合，则加入 goal 任务（最简形态） ----------
    if end_rid is not None:
        for goal_idx  in range(len(end_rid)) :
            if end_rid[goal_idx] in reachable_rooms:
                goal_item = {"type": "goal", "room": end_rid[goal_idx],"pos":{"type": "xy", "value": goal_pos[goal_idx]}}
                if include_paths:
                    p = room_path_to(end_rid)
                    goal_item["via_rooms"] = p
                    goal_item["via_steps"] = max(0, len(p) - 1)
                tasks.append(goal_item)
    return tasks


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

    
def find_all_keys(data,color):
    all_keys = []
    for room, info in data.items():
        for key in info.get("keys", []):
            if color ==  key["color"]:
                all_keys.append({
                    "room": room,
                    "color": key["color"],
                    "pos": key["pos"]
                })
    return all_keys


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


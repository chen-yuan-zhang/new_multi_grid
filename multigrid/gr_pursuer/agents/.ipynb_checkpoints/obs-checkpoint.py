from .base import BaseAgent
from ..new_astar import astar, execute_action, get_obs_successor, get_reverse_successor
from ..lock_astar import astar_key,astar_open
from ..new_astar import astar

import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC, Direction,OBJECT_TO_IDX,COLOR_TO_IDX,Type
from multigrid.core.actions import Action
from multigrid.envs.new_locked import _plan_onekey_persist_open,_build_neighbors,_canon,_initial_open_mask
from multigrid.envs.new_locked import _iter_room_keys,_key_position,_edge_position

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
        ###—————————————————————————————————————————————————————————————————————————————————————
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)

    def compute_action(self, obs):
        # 读取观测者状态
        obs_held = self.env.observer.carrying
        obs_pos  = self.env.observer.pos
        obs_dir  = self.env.observer.dir
    
        # 同步门/钥匙状态（注意：该函数不要修改全局结构外的东西）
        self.update_door_state()
        door_list = self.abs["edges"]
    
        # 候选计划：[(total_dist, eid, plan_type, target_pos, nearest_key_dict)]
        # plan_type: "open" 或 "pickup_then_open"
        candidates = []
    
        for eid, door in enumerate(door_list):
            if not door.get("locked", False):
                continue  # 只考虑上锁的门
    
            door_color = door["color"]
            door_pos   = door["pos"]
    
            # 情况 A：手里就有对应颜色的钥匙 → 直接去开门
            if obs_held is not None and getattr(obs_held, "color", None) == door_color:
                total_dist = manhattan_distance(obs_pos, door_pos)
                candidates.append((total_dist, eid, "open", door_pos, None))
                continue
    
            # 情况 B：手里没有对应钥匙 → 去找最近的该色钥匙，再去该门
            all_keys = find_all_keys(self.abs["rooms"], door_color)  # [{'color':..., 'pos':(x,y)}, ...]
            if not all_keys:
                # 没有该颜色钥匙可拿，跳过这个门
                continue
    
            # 选择 obs→key + key→door 最短的那把钥匙
            def key_chain_dist(k):
                kp = k['pos']
                return manhattan_distance(obs_pos, kp) + manhattan_distance(kp, door_pos)
    
            nearest_key = min(all_keys, key=key_chain_dist)
            total_dist  = key_chain_dist(nearest_key)
    
            candidates.append((total_dist, eid, "pickup_then_open", nearest_key['pos'], nearest_key))
    
        # 没有可行动的目标（例如全都无钥匙可达）
        if not candidates:
            # 这里按你工程里的“等待/随机/维持方向”策略返回一个安全动作
            # 例如：保持不动或向前（请替换为你项目里的 no-op）
            return 0
    
        # 选择总距离最小的门；若距离相同，用 eid 稳定打破平手
        candidates.sort(key=lambda t: (t[0], t[1]))
        best_dist, best_eid, plan_type, target_pos, nearest_key = candidates[0]
        best_door = door_list[best_eid]
    
        # 具体执行路径规划
        if plan_type == "open":
            path = astar_open((obs_pos, obs_dir), best_door["pos"], self.env, self.hidden_cost, version=True)
        else:
            path = astar_key((obs_pos, obs_dir), target_pos, self.env, self.hidden_cost, agent_idx=0, version=True)
    
        # 保护：确保有下一步
        if not path or len(path) < 2:
            return 0  
    
        return path[1][0]

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




class Observer(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        super().__init__(env.observer)
        self.env = env # width x height y get(x,y)
        self.hallway_col = env.num_cols // 2
        self.abs = env.abs
        self.goals = env.goals
        self.goal = env.goal
        self.belif_goal = None
        self.unsolvable_goal = []
        self.finished_plan = -1
        
        self.goal_rooms = [_rid_from_xy(self.env,x,y,self.hallway_col) for x,y in self.goals]
        #print(self.goal_rooms)
        self.high_level_length = {}
        self.dist_matrix = self.compute_pairwise_distances()
        self.past_plans = []
        self.goal_length = {}        
        self.prior = [1.0/len(self.goal_rooms) for g in self.goal_rooms]
        self.current_prior = [1.0/len(self.goal_rooms) for g in self.goal_rooms]
        self.past_pos = None
        self.past_dir = None

        ###——————————————————————————————————————————————————————————————————————————————————————

        # 在 __init__ 里先加两个缓存
        #self._locked_mask_prev = None
        #self._keys_sig_prev = None
        
        self.update_high_level()
        self.subgoal_dist = {}
        self.current_palns = []
        pos = self.env.target.pos
        dir = self.env.target.dir
        held = self.env.target.carrying
        for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
                                                     held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
            self.current_palns.append(plan)
        for plan in self.current_palns:
            self.subgoal_dist[plan['pos']['value']] = self.dist_matrix[(pos,dir),plan['pos']['value']]

        ###——————————————————————————————————————————————————————————————————————————————————————

        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.width, env.height), dtype=np.float32)
    
    def _is_wall(self, pos):
        obj = self.env.grid.get(*pos)
        return (obj is not None) and (getattr(obj, "type", None) == "wall")
    

    def update_high_level(self):
        pos = self.env.target.pos
        held = self.env.target.carrying
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        self.unsolvable_goal = []
        self.past_plans = []
        self.goal_length = {}
        
        for goal_room_idx in range(len(self.goal_rooms)):
            goal = self.goals[goal_room_idx]
            goal_room = self.goal_rooms[goal_room_idx]
            high_level_plan = _plan_onekey_persist_open(self.abs, start_rid, goal_room, held = held,goal = goal)
            self.high_level_length[goal_room] = [len(high_level_plan),high_level_plan]
            if self.high_level_length[goal_room][0] == 0:
                self.unsolvable_goal.append(goal_room)
            self.goal_length[goal_room] = high_level_plan
        self.new_env = False
        

    def length_compute(self,pos,dir,plan_list):
        if plan_list == []:
            return 5000
        total_length = 0
        for plan in plan_list:
            total_length += self.dist_matrix[(pos,dir),plan['pos']['value']]
            pos = plan['pos']['value']
        return total_length

    def goal_recognition(self):
        pos = self.env.target.pos
        dir = self.env.target.dir
        held = self.env.target.carrying
        self.current_palns = []
        #compute the high level plan
        for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
                                                             held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
            self.current_palns.append(plan)

        if held:
            rid = _rid_from_xy(self.env, pos[0], pos[1], self.hallway_col)
            self.abs["rooms"][rid]["keys"].append({
                                        'color': held.color,
                                        'pos': (pos[0],pos[1])})

        
        if self.past_pos and self.unsolvable_goal != []:
            if self.past_pos == pos and self.past_dir == dir:
                for goal_room in self.unsolvable_goal:
                    self.high_level_length[goal_room][0] += 10
            else:
                for goal_room in self.unsolvable_goal:
                    self.high_level_length[goal_room][0] -= 10
            # else:
            #     obj = self.env.grid.get(*new_pos)
            #     if self.past_pos == pos and self.past_dir == dir:
            #         if obj and obj.type == "door":
            #             for goal_room, (length_val, path_list) in self.high_level_length.items():
            #                 for event in path_list:
            #                     if 'pos' in event and event['pos']['value'] == new_pos:
            #                         self.high_level_length[goal_room][0] -= 10
            #                         break
            #         else:
            #             for goal_room in self.unsolvable_goal:
            #                 self.high_level_length[goal_room][0] += 10
            #     else:
            #         for goal_room in self.unsolvable_goal:
            #             self.high_level_length[goal_room][0] -= 10

        if self.new_env :
            #print("new env!")
            self.update_high_level()
            self.finished_plan = -1
            self.steps = 0
            self.prior = self.current_prior

        # check the plan has changed and inital the subgoal
        if self.past_plans != self.current_palns:
            self.weights = [1/len(self.current_palns) for g in self.current_palns]
            self.finished_plan += 1
            self.steps = 0
            self.subgoal_dist = {}
            start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
            for goal_room_idx in range(len(self.goal_rooms)):
                goal = self.goals[goal_room_idx]
                goal_room = self.goal_rooms[goal_room_idx]
                high_level_plan = _plan_onekey_persist_open(self.abs, start_rid, goal_room, held = held,goal = goal)
                self.goal_length[goal_room] = high_level_plan
                
            for plan in self.current_palns:
                if plan['type'] == 'pickup':
                    self.subgoal_dist[plan['pos']['value']] = len(astar_key(
                                    (pos,dir),plan['pos']['value'],self.env,self.hidden_cost,agent_idx =1))
                elif plan['type'] == 'open':
                    self.subgoal_dist[plan['pos']['value']] = len(astar_open(
                                    (pos,dir),plan['pos']['value'],self.env,self.hidden_cost))

                else:
                    self.subgoal_dist[plan['pos']['value']] = len(astar((pos, dir), plan['pos']['value'], self.env, self.hidden_cost))
                    
        else:
            self.steps += 1


        # if self.belif_goal:
        #     return [1 if g == self.belif_goal else 0 for g in self.goals]
        # return [1 for i in range(len(self.goals))] #uniform
        # return [1 if g == self.goal else 0 for g in self.goals]
        if self.current_palns == []:
            return [1 for i in range(len(self.goals))]

        
        #compute the high level plan
        subgoal_pro = []
        subgoal_final_pro = []

        for plan in self.current_palns:
            start_rid = _rid_from_xy(self.env,*plan['pos']['value'],self.hallway_col)
            softmax = []
            if plan['type'] == 'pickup':
                subgoal_dist = self.steps + len(astar_key(
                                    (pos,dir),plan['pos']['value'],self.env,self.hidden_cost,agent_idx =1))
                subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
                diff = subgoal_dist - subgoal_optimal
                subgoal_pro.append(diff)
                
                for goal_room_idx in range(len(self.goal_rooms)):
                    goal = self.goals[goal_room_idx]
                    goal_room = self.goal_rooms[goal_room_idx]
                    optimal_high_length = self.high_level_length[goal_room][0]
                    if self.high_level_length[goal_room][0] == 1:
                        optimal_high_length = -100
                    plan_len = len(_plan_onekey_persist_open(self.abs, start_rid, goal_room, held = plan['key'],goal = goal))
                    current_high_length  = self.finished_plan + plan_len + 1
                    var = current_high_length - optimal_high_length
                    softmax.append(var)
            elif plan['type'] == 'open':
                subgoal_dist = self.steps + len(astar_open(
                                    (pos,dir),plan['pos']['value'],self.env,self.hidden_cost))
                subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
                diff = subgoal_dist - subgoal_optimal
                subgoal_pro.append(diff)
                
                for goal_room_idx in range(len(self.goal_rooms)):
                    goal = self.goals[goal_room_idx]
                    goal_room = self.goal_rooms[goal_room_idx]
                    optimal_high_length = self.high_level_length[goal_room][0]
                    if self.high_level_length[goal_room][0] == 1:
                        optimal_high_length = -100
                    mask = _initial_open_mask(self.abs)
                    mask |= (1 << plan['eid'])
                    plan_len = len(_plan_onekey_persist_open(self.abs, start_rid, goal_room, held = held,open_mask = mask,goal = goal))
                    current_high_length = self.finished_plan + plan_len + 1
                    var = current_high_length - optimal_high_length
                    softmax.append(var)
            else:
                subgoal_dist = self.steps + len(astar((pos, dir), plan['pos']['value'], self.env, self.hidden_cost))
                subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
                diff = subgoal_dist - subgoal_optimal
                subgoal_pro.append(diff)
                
                for goal_room in self.goal_rooms:
                    if goal_room == plan["room"]:
                        softmax.append(0.0)
                    else:
                        softmax.append(5.0)
            subgoal_final_pro.append(softmax_prob(softmax))


        self.past_pos = pos
        self.past_dir = dir
        
        # print([(plan['pos']['value'],plan['type']) for plan in self.current_palns])
        print("subgoal_pro", [f"{x}" for x in subgoal_pro])
        for i in subgoal_final_pro:
            print([f"{x:.2f}" for x in i])
        #print(self.prior)
        if self.past_plans != self.current_palns:
            w = np.linalg.lstsq(np.array(subgoal_final_pro).T, np.array(self.current_prior), rcond=None)[0]
            weights = softmax_prob(subgoal_pro)
            self.weights  = np.exp(w)
            self.weights   = self.weights   / np.sum(self.weights) 
            weights = [self.weights[g]*weights[g] for g in range(len(weights))]
            weights = [w / sum(weights) for w in weights]
            likelihood = weighted_goal_probabilities(weights, subgoal_final_pro)
        else:
            weights = softmax_prob(subgoal_pro)
            weights = [self.weights[g]*weights[g] for g in range(len(weights))]
            weights = [w / sum(weights) for w in weights]
            likelihood  = weighted_goal_probabilities(weights, subgoal_final_pro)


        print("prio weights",[f"{x:.2f}" for x in self.weights])
        print("weights",[f"{x:.2f}" for x in weights])
                
        post = {g: 0.0 for g in self.goal_rooms}
        for gi, g in enumerate(self.goal_rooms):
            post[g] += likelihood[gi]
        for gi, g in enumerate(self.goal_rooms):
            post[g] *= self.prior[gi]
        Z = sum(post.values()) + 1e-12
        for g in post:
            post[g] /= Z
        prob = list(post.values())
        self.current_prior = prob
        self.past_plans = self.current_palns

        
        # #paper
        # if self.env.step_count > 5 and  self.belif_goal is None:

        # #if max(prob)> (1/len(self.goals)+0.1) and  self.belif_goal is None:
        #     best_idx = int(np.argmax(prob))
        #     self.belif_goal = self.goals[best_idx]

        # if self.belif_goal:
        #     return [1 if g == self.belif_goal else 0 for g in self.goals]

        # return [1 for i in range(len(self.goals))] 
        
        #return [1 if g == self.goal else 0 for g in self.goals] # upperbound

    
        print("prob:",[f"{x:.2f}" for x in prob])
        print(self.goal_rooms)
        
        best_idx = int(np.argmax(prob))
        self.belif_goal = self.goals[best_idx]
        return prob   #gr
        #return 0


    def compute_action(self, obs):
        pos = self.env.target.pos
        dir = self.env.target.dir
        held = self.env.target.carrying
        obs_held = self.env.observer.carrying
        obs_pos = self.env.observer.pos
        obs_dir = self.env.observer.dir
        self.update_door_state()
        abs_graph = deepcopy(self.abs)
        goal_recognition = self.goal_recognition()

        #print(goal_recognition)
        start_rid = _rid_from_xy(self.env,*pos,self.hallway_col)
        door_list = abs_graph["edges"]
        subgoal_expected_payoff = {}

        for eid in range(len(door_list)):
            door = door_list[eid]
            abs_graph_new = deepcopy(abs_graph)
            all_keys = find_all_keys(abs_graph_new["rooms"],door["color"])
            if door["locked"] == True and (all_keys != [] or (obs_held is not None and obs_held.color == door["color"])):
                key_distance = 0
                subgoal_expected_payoff[eid] = 0
                if all_keys  != []:
                    if (obs_held is None or obs_held.color != door["color"]):
                        nearest = min(all_keys, key=lambda k: manhattan_distance(k['pos'], obs_pos) + manhattan_distance(k['pos'],door["pos"]))
                        key_distance = manhattan_distance(nearest["pos"],door["pos"]) + manhattan_distance(nearest["pos"],obs_pos)
                        if {"color":nearest["color"],"pos":nearest["pos"]} in abs_graph_new["rooms"][nearest["room"]]["keys"]:
                            keys = abs_graph_new["rooms"][nearest["room"]]["keys"]
                            target = {"color": nearest["color"], "pos": nearest["pos"]}
                            try:
                                keys.remove(target)
                            except ValueError:
                                pass 
                    else:
                        key_distance = manhattan_distance(obs_pos,door["pos"])
                for goal_room_idx in range(len(self.goal_rooms)):
                    goal_room = self.goal_rooms[goal_room_idx]
                    goal = self.goals[goal_room_idx]
                    mask = _initial_open_mask(abs_graph_new)
                    mask |= (1 << eid)
                    plan_list = _plan_onekey_persist_open(abs_graph_new,start_rid, goal_room, held = held,open_mask = mask, goal = goal)
                    payoff = (self.length_compute(pos,dir,self.goal_length[goal_room])
                              - self.length_compute(pos,dir,plan_list))  - 0.1*key_distance
                    excepted_payoff = goal_recognition[goal_room_idx] * payoff
                    subgoal_expected_payoff[eid] += excepted_payoff

       # print(subgoal_expected_payoff)
        if subgoal_expected_payoff == {}:
            return 0
        max_door = max(subgoal_expected_payoff, key=subgoal_expected_payoff.get)
        door = self.abs["edges"][max_door]
       # print(door["pos"],door["color"],max_door,subgoal_expected_payoff[max_door])
        
        path = []
        if obs_held:
            if obs_held.color == door["color"]:
                path = astar_open((obs_pos,obs_dir),door["pos"],self.env,self.hidden_cost, version = True)
                if path[1][0] in [Action.drop,Action.pickup,Action.toggle]:
                    self.new_env = True
                return path[1][0] 
        all_keys = find_all_keys(abs_graph["rooms"],door["color"])
        nearest = min(all_keys, key=lambda k: manhattan_distance(k['pos'], obs_pos) + manhattan_distance(k['pos'],door["pos"]))
        path = astar_key((obs_pos,obs_dir),nearest['pos'],self.env,self.hidden_cost,agent_idx =0, version = True)
        if path[1][0] in [Action.drop,Action.pickup,Action.toggle]:
            self.new_env = True

        #print( path[1][0])

        return path[1][0] 

        # if self.goal:
        #     return path[1][0]
        # else:
        #     return 0


    # def goal_recognition(self):
    #     pos = self.env.target.pos
    #     dir = self.env.target.dir
    #     held = self.env.target.carrying
    #     self.current_palns = []
    #     #compute the high level plan
    #     for plan in current_task_options_onekey_persist_open(self.abs,_rid_from_xy(self.env,*pos,self.hallway_col),
    #                                                          held = held, end_rid = self.goal_rooms, goal_pos = self.goals):
    #         self.current_palns.append(plan)
    #     # check the plan has changed and inital the subgoal
        
    #     if held:
    #         rid = _rid_from_xy(self.env, pos[0], pos[1], self.hallway_col)
    #         self.abs["rooms"][rid]["keys"].append({
    #                                     'color': held.color,
    #                                     'pos': (pos[0],pos[1])})
    #     if self.new_env:
    #         print("new env!")
    #         self.update_high_level()
    #         self.steps = 0
    #         self.subgoal_dist = {}
    #         self.prior = self.current_prior
    #         for plan in self.current_palns:
    #             self.subgoal_dist[plan['pos']['value']] = self.dist_matrix[(pos,dir),plan['pos']['value']]
    #     else:
    #         self.steps += 1
    #     self.past_plans = self.current_palns
    #     abs_graph = deepcopy(self.abs)
    #     #compute the high level plan
    #     subgoal_pro = []
    #     subgoal_final_pro = []
    #     if self.current_palns == []:
    #         return [1 for i in range(len(self.goals))]
    #     #print(self.subgoal_dist)
    #     #print(self.current_palns)
    #     for plan in self.current_palns:
    #         start_rid = _rid_from_xy(self.env,*plan['pos']['value'],self.hallway_col)
    #         subgoal_dist = self.steps + self.dist_matrix[(pos,dir),plan['pos']['value']]
    #         subgoal_optimal = self.subgoal_dist[plan['pos']['value']]
    #         diff = subgoal_dist - subgoal_optimal
    #         subgoal_pro.append(diff)
    #         softmax = []         
    #         if plan['type'] == 'pickup':
    #             for goal_room_idx in range(len(self.goal_rooms)):
    #                 goal_room = self.goal_rooms[goal_room_idx]
    #                 goal = self.goals[goal_room_idx]
    #                 current_high_length  = len(_plan_onekey_persist_open(abs_graph, start_rid,goal_room, held = plan['key'],goal = goal)) + 1
    #                 var = current_high_length - self.high_level_length[goal_room][0]
    #                 print(current_high_length,self.high_level_length[goal_room][0],goal_room,plan['type'],plan['pos']['value'])
    #                 print(_plan_onekey_persist_open(abs_graph, start_rid,goal_room, held = plan['key'],goal = goal),"\n")
    #                 print(self.high_level_length[goal_room][1])
    #                 softmax.append(var)
    #         elif plan['type'] == 'open':
    #             for goal_room_idx in range(len(self.goal_rooms)):
    #                 goal_room = self.goal_rooms[goal_room_idx]
    #                 goal = self.goals[goal_room_idx]
    #                 mask = _initial_open_mask(abs_graph)
    #                 mask |= (1 << plan['eid'])
    #                 current_high_length = len(_plan_onekey_persist_open(abs_graph,start_rid, goal_room, held = held,open_mask = mask,goal=goal)) + 1
    #                 var = current_high_length - self.high_level_length[goal_room][0]
    #                 print(current_high_length,self.high_level_length[goal_room][0],goal_room,plan['type'],plan['pos']['value'])
    #                 # if goal_room ==  (0, 1):
    #                 print(_plan_onekey_persist_open(abs_graph,start_rid, goal_room, held = held,open_mask = mask,goal=goal),"\n")
    #                 print(self.high_level_length[goal_room][1])
    #                 softmax.append(var)
    #         else:
    #             for goal_room in self.goal_rooms:
    #                 if goal_room == plan["room"]:
    #                     softmax.append(0.0)
    #                 else:
    #                     softmax.append(5.0)
    #         subgoal_final_pro.append(softmax_prob(softmax))
        
    #     weights = softmax_prob(subgoal_pro)
    #     print("weights",weights)
    #     print("subgoal_final_pro")
    #     for i in subgoal_final_pro:
    #         print(i)
    #     likelihood  = weighted_goal_probabilities(weights, subgoal_final_pro)

    #     post = {g: 0.0 for g in self.goal_rooms}
    #     for gi, g in enumerate(self.goal_rooms):
    #         post[g] += likelihood[gi]
    #     # 乘上先验并归一化
    #     #if not new_env == False:
    #     for g in post:
    #         post[g] *= self.prior[g]
    #     Z = sum(post.values()) + 1e-12
    #     for g in post:
    #         post[g] /= Z
    #     prob = list(post.values())
    #     self.current_prior = post

        
    #     # if self.goal:
    #     #     return [1 if g == self.goal else 0 for g in self.goals]
    #     # else:
    #     #     for i in range(len(prob)):
    #     #         if prob[i] > 0.5:
    #     #             self.goal = self.goals[i]
    #     #     return prob   #gr

    #     #return [1 if g == self.goal else 0 for g in self.goals]
    #     #return [1 for i in range(len(self.goals))] #uniform
    #     print("prob:",prob)
    #     print(self.goal_rooms)
    #     return prob   #gr
    
    def update_door_state(self):
        # 1) 清空 rooms 里的 keys（保持你原流程）
        for room in self.abs["rooms"]:
            self.abs["rooms"][room]["keys"] = []
    
        # 2) 扫描 grid 的钥匙，写回 abs，并同时构造 keys_sig
        keys_acc = []
        for x in range(self.env.width):
            for y in range(self.env.height):
                obj = self.env.grid.get(x, y)
                if obj and obj.type == "key":
                    room = _rid_from_xy(self.env, x, y, self.hallway_col)
                    color = _canon_color(obj.color)
                    self.abs["rooms"][room]["keys"].append({"color": obj.color, "pos": (x, y)})
                    room = _canon_room(room)
                    keys_acc.append((room, color, x, y))
    
        # 排序后不可变，便于比较
        keys_sig = tuple(sorted(keys_acc))
    
        # 3) 更新门状态，并构造 bitmask
        locked_mask = 0
        for i, door in enumerate(self.abs["edges"]):
            obj = self.env.grid.get(*door["pos"])
            if obj and (obj.type == "door"):
                locked = (obj.state == "locked")
            else:
                locked = False
            door["locked"] = locked
            if locked:
                locked_mask |= (1 << i)
    
        # # 4) 与上一次签名比较，判断是否变化（无需 deepcopy）
        # changed = False
        # if (self._locked_mask_prev is None) or (self._keys_sig_prev is None):
        #     changed = True  # 第一次调用视为“有变化”
        # else:
        #     changed = (locked_mask != self._locked_mask_prev) or (keys_sig != self._keys_sig_prev)
    
        # # 5) 缓存当前签名 & 输出标记
        # #print("here?")
        # self._locked_mask_prev = locked_mask
        # self._keys_sig_prev = keys_sig
        # self.new_env = changed


    def compute_pairwise_distances(self):
        """
        Compute shortest distances from every state (cell + direction) to every free cell.
        Return: dict with keys ((pos, dir), target_cell) -> distance
        """
        # 1) 可走格子与索引
        free_cells = np.argwhere(self.env.base_grid[:, :, 0] != 2)
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

def compute_prior_for_uniform_posterior(likelihood):
    likelihood = np.array(likelihood, dtype=float)
    inv = 1 / likelihood
    prior = inv / np.sum(inv)
    return prior

def _canon_color(c):
    # 枚举/字符串统一成字符串
    return getattr(c, "value", getattr(c, "name", str(c))).lower()

def _canon_room(room):
    # room 可能是 (r,c) 或 ('HALL',) 或其他；统一成字符串
    return str(room)
    
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





def _project_to_simplex(w):
    """把 w 投影到 {w >= 0, sum w = 1} 的概率单纯形（O(n log n)算法）。"""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        w = w.ravel()
    n = w.size
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    rho = rho[-1] if rho.size else 0
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w_proj = np.maximum(w - theta, 0.0)
    return w_proj

def fit_weights_keep_prior(lik_rows, max_iter=50000, lr=0.5, tol=1e-9):
    """
    给定若干 subgoal 的似然行向量（每行长度=K，已归一化），
    反求组合权重 w，使 v = sum_j w_j*lik_rows[j] 尽量“各列相等”（posterior≈prior）。

    lik_rows: list[np.ndarray], 形状 (m, K)，每一行是一个 subgoal 对 K 个 goal 的似然分布
    返回: w (m,)  满足 w>=0, sum w=1
    """
    L = np.asarray(lik_rows, dtype=float)  # (m, K)
    m, K = L.shape

    # 预处理：每行归一化，避免尺度差异
    L = L / (L.sum(axis=1, keepdims=True) + 1e-12)

    # 初始化：用“信息量”启发（列方差的倒数）；也可用均匀
    w = np.ones(m, dtype=float) / m

    # 惩罚目标： minimize f(w) = ||v - mean(v)||^2，其中 v = L^T w
    for _ in range(max_iter):
        v = L.T @ w                    # (K,)
        v_bar = v.mean()
        r = v - v_bar                  # 残差（去均值）
        f = float(np.dot(r, r))        # 当前目标

        # 梯度：df/dw = 2 * L * r
        grad = 2.0 * (L @ r)           # (m,)

        # 梯度步 + 投影回单纯形
        w_new = _project_to_simplex(w - lr * grad)

        # 收敛判据
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new

    return w

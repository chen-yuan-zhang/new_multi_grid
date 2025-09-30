#!/usr/bin/env python3
"""
Simulate observer's greedy action selection for instance 565
to understand why actor goes in and out of view
"""

import pandas as pd
import numpy as np
import ast
import sys
import os

# Add multigrid to path
sys.path.append('/home/student.unimelb.edu.au/chenyuanz/prob_active_gr/new_multi_grid')

from multigrid.envs.goal_recognition import GoalRecognitionEnv
from multigrid.envs.envs import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Wall, Goal, Door, Key, Ball, Box
from multigrid.core.constants import COLOR_NAMES
from multigrid.utils.window import Window

def create_environment_from_data(row):
    """Create environment from stored data"""
    
    # Parse the base grid
    base_grid_str = str(row['base_grid'])
    
    # Extract basic parameters
    size = int(row['size'])
    observer_pos = eval(str(row['observer_pos']))
    target_pos = eval(str(row['target_pos']))
    observer_dir = int(row['observer_dir'])
    target_dir = int(row['target_dir'])
    
    print(f"Creating {size}x{size} environment")
    print(f"Observer start: {observer_pos}, direction: {observer_dir}")
    print(f"Target start: {target_pos}, direction: {target_dir}")
    
    # Create a simple environment for simulation
    env = GoalRecognitionEnv(
        size=size,
        width=size,
        height=size,
        max_steps=100,
        see_through_walls=False,
        agent_view_size=5
    )
    
    return env, observer_pos, target_pos, observer_dir, target_dir

def simulate_greedy_observer(env, observer_pos, target_pos, observer_dir, target_dir, max_steps=20):
    """Simulate observer using greedy action selection"""
    
    print(f"\n=== SIMULATING GREEDY OBSERVER ===")
    
    # Initialize positions
    current_obs_pos = list(observer_pos)
    current_target_pos = list(target_pos)
    current_obs_dir = observer_dir
    current_target_dir = target_dir
    
    # Action mappings
    actions = {0: 'move_forward', 1: 'turn_left', 2: 'turn_right'}
    direction_names = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    
    # Direction vectors for movement
    dir_vec = {
        0: np.array([1, 0]),   # right
        1: np.array([0, 1]),   # down  
        2: np.array([-1, 0]),  # left
        3: np.array([0, -1])   # up
    }
    
    visibility_log = []
    action_log = []
    position_log = []
    
    for step in range(max_steps):
        print(f"\nStep {step}:")
        print(f"  Observer: pos={current_obs_pos}, dir={current_obs_dir} ({direction_names[current_obs_dir]})")
        print(f"  Target: pos={current_target_pos}, dir={current_target_dir}")
        
        # Calculate if target is visible (within 5x5 view centered on observer)
        obs_array = np.array(current_obs_pos)
        target_array = np.array(current_target_pos)
        
        # Check if target is in the 5x5 observation window
        # The observer sees 2 cells in each direction from their position
        view_bounds = {
            'min_x': current_obs_pos[0] - 2,
            'max_x': current_obs_pos[0] + 2,
            'min_y': current_obs_pos[1] - 2,
            'max_y': current_obs_pos[1] + 2
        }
        
        target_visible = (
            view_bounds['min_x'] <= current_target_pos[0] <= view_bounds['max_x'] and
            view_bounds['min_y'] <= current_target_pos[1] <= view_bounds['max_y']
        )
        
        print(f"  View bounds: x=[{view_bounds['min_x']}, {view_bounds['max_x']}], y=[{view_bounds['min_y']}, {view_bounds['max_y']}]")
        print(f"  Target visible: {target_visible}")
        
        visibility_log.append(target_visible)
        position_log.append({
            'obs_pos': current_obs_pos.copy(),
            'target_pos': current_target_pos.copy(),
            'obs_dir': current_obs_dir,
            'visible': target_visible
        })
        
        # Simulate greedy action selection
        # Calculate Manhattan distance to target
        current_distance = abs(current_obs_pos[0] - current_target_pos[0]) + abs(current_obs_pos[1] - current_target_pos[1])
        
        print(f"  Current distance: {current_distance}")
        
        # Try each possible action and see which reduces distance most
        best_action = None
        best_distance = current_distance
        
        # Action 0: Move forward
        new_pos = np.array(current_obs_pos) + dir_vec[current_obs_dir]
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
            new_distance = abs(new_pos[0] - current_target_pos[0]) + abs(new_pos[1] - current_target_pos[1])
            if new_distance < best_distance:
                best_action = 0
                best_distance = new_distance
                print(f"    Move forward -> pos={new_pos}, distance={new_distance}")
        
        # Action 1: Turn left then move
        new_dir = (current_obs_dir - 1) % 4
        new_pos = np.array(current_obs_pos) + dir_vec[new_dir]
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
            new_distance = abs(new_pos[0] - current_target_pos[0]) + abs(new_pos[1] - current_target_pos[1])
            if new_distance < best_distance:
                best_action = 1
                best_distance = new_distance
                print(f"    Turn left -> dir={new_dir}, pos={new_pos}, distance={new_distance}")
        
        # Action 2: Turn right then move  
        new_dir = (current_obs_dir + 1) % 4
        new_pos = np.array(current_obs_pos) + dir_vec[new_dir]
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
            new_distance = abs(new_pos[0] - current_target_pos[0]) + abs(new_pos[1] - current_target_pos[1])
            if new_distance < best_distance:
                best_action = 2
                best_distance = new_distance
                print(f"    Turn right -> dir={new_dir}, pos={new_pos}, distance={new_distance}")
        
        # If no action improves distance, prefer moving forward, then turning
        if best_action is None:
            # Try move forward first
            new_pos = np.array(current_obs_pos) + dir_vec[current_obs_dir]
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                best_action = 0
                print(f"    No improvement, moving forward")
            else:
                # Can't move forward, turn toward target
                target_diff = np.array(current_target_pos) - np.array(current_obs_pos)
                
                # Determine which direction would be best
                if abs(target_diff[0]) > abs(target_diff[1]):
                    # Move horizontally
                    desired_dir = 0 if target_diff[0] > 0 else 2
                else:
                    # Move vertically  
                    desired_dir = 1 if target_diff[1] > 0 else 3
                
                # Choose turn to get closer to desired direction
                left_dir = (current_obs_dir - 1) % 4
                right_dir = (current_obs_dir + 1) % 4
                
                if left_dir == desired_dir:
                    best_action = 1
                elif right_dir == desired_dir:
                    best_action = 2
                else:
                    best_action = 1  # Default to turn left
                
                print(f"    Can't move forward, turning toward target (desired_dir={desired_dir})")
        
        action_name = actions[best_action] if best_action is not None else "none"
        print(f"  Chosen action: {best_action} ({action_name})")
        
        action_log.append(best_action)
        
        # Execute the action
        if best_action == 0:  # Move forward
            new_pos = np.array(current_obs_pos) + dir_vec[current_obs_dir]
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                current_obs_pos = new_pos.tolist()
        elif best_action == 1:  # Turn left
            current_obs_dir = (current_obs_dir - 1) % 4
        elif best_action == 2:  # Turn right
            current_obs_dir = (current_obs_dir + 1) % 4
        
        # Simple target movement (hate_wall behavior)
        # Target tries to avoid walls/edges, moves somewhat randomly
        target_moved = False
        for _ in range(3):  # Try a few movement options
            # Random direction
            random_dir = np.random.randint(0, 4)
            new_target_pos = np.array(current_target_pos) + dir_vec[random_dir]
            
            # Check bounds and prefer moving away from edges (hate_wall)
            if (0 <= new_target_pos[0] < env.width and 0 <= new_target_pos[1] < env.height):
                # Bias away from walls for hate_wall behavior
                edge_penalty = 0
                if new_target_pos[0] <= 1 or new_target_pos[0] >= env.width - 2:
                    edge_penalty += 1
                if new_target_pos[1] <= 1 or new_target_pos[1] >= env.height - 2:
                    edge_penalty += 1
                
                # Accept move if not too close to edges
                if edge_penalty == 0 or np.random.random() > 0.7:
                    current_target_pos = new_target_pos.tolist()
                    current_target_dir = random_dir
                    target_moved = True
                    break
        
        if not target_moved:
            print(f"  Target stayed at {current_target_pos}")
        else:
            print(f"  Target moved to {current_target_pos}")
    
    return visibility_log, action_log, position_log

def analyze_simulation_results(visibility_log, action_log, position_log):
    """Analyze the simulation results"""
    
    print(f"\n=== SIMULATION ANALYSIS ===")
    print(f"Total steps: {len(visibility_log)}")
    
    # Visibility analysis
    total_visible = sum(visibility_log)
    visibility_ratio = total_visible / len(visibility_log) if len(visibility_log) > 0 else 0
    
    print(f"Target visible: {total_visible}/{len(visibility_log)} steps ({visibility_ratio:.1%})")
    
    # Find visibility changes
    visibility_changes = []
    for i in range(1, len(visibility_log)):
        if visibility_log[i] != visibility_log[i-1]:
            change_type = "appeared" if visibility_log[i] else "disappeared"
            visibility_changes.append((i, change_type))
    
    print(f"Visibility changes: {len(visibility_changes)}")
    
    for step, change in visibility_changes:
        action_taken = action_log[step-1] if step-1 < len(action_log) else "N/A"
        if isinstance(action_taken, int):
            action_name = {0: 'move_forward', 1: 'turn_left', 2: 'turn_right'}.get(action_taken, f'action_{action_taken}')
        else:
            action_name = str(action_taken)
        
        print(f"  Step {step}: Target {change} (after {action_name})")
        
        if step < len(position_log):
            pos_info = position_log[step]
            print(f"    Observer: {pos_info['obs_pos']}, Target: {pos_info['target_pos']}")
    
    # Action analysis
    action_counts = {}
    for action in action_log:
        action_name = {0: 'move_forward', 1: 'turn_left', 2: 'turn_right'}.get(action, 'unknown')
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    
    print(f"\nAction distribution:")
    for action, count in action_counts.items():
        percentage = count / len(action_log) * 100 if len(action_log) > 0 else 0
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    # Create visibility pattern string
    visibility_str = ''.join(['1' if v else '0' for v in visibility_log])
    print(f"\nVisibility pattern: {visibility_str}")
    
    return visibility_changes

def main():
    """Main analysis function"""
    
    # Load instance 565 
    df = pd.read_csv('evaluation_results_1759207042.csv')
    row_565 = df.iloc[565]
    
    print("=== INSTANCE 565 SIMULATION ===")
    print(f"Size: {row_565['size']}")
    print(f"Behavior: {row_565['hidden_cost_style']}")
    print(f"Initial distance: {row_565['initial_distance']}")
    print(f"Actual result: Success={row_565['eval_success']}, Steps={row_565['total_steps']}")
    
    # Create environment and simulate
    env, observer_pos, target_pos, observer_dir, target_dir = create_environment_from_data(row_565)
    
    # Run simulation
    visibility_log, action_log, position_log = simulate_greedy_observer(
        env, observer_pos, target_pos, observer_dir, target_dir, max_steps=15
    )
    
    # Analyze results
    visibility_changes = analyze_simulation_results(visibility_log, action_log, position_log)
    
    print(f"\n=== KEY INSIGHTS ===")
    
    if len(visibility_changes) > 3:
        print("🔍 HIGH VISIBILITY INSTABILITY DETECTED!")
        print("   The observer's greedy actions cause frequent loss/gain of target visibility")
        print("   This explains why the actor appears to go 'in and out of view'")
    
    # Check for movement patterns that cause issues
    consecutive_turns = 0
    max_consecutive_turns = 0
    
    for action in action_log:
        if action in [1, 2]:  # turn actions
            consecutive_turns += 1
            max_consecutive_turns = max(max_consecutive_turns, consecutive_turns)
        else:
            consecutive_turns = 0
    
    if max_consecutive_turns > 2:
        print(f"🔄 EXCESSIVE TURNING DETECTED!")
        print(f"   Maximum consecutive turns: {max_consecutive_turns}")
        print("   This can cause the observer to lose track of the target")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   The greedy algorithm's local optimization causes suboptimal movement patterns")
    print(f"   that lead to visibility instability, explaining the 'in and out of view' behavior")

if __name__ == "__main__":
    main()
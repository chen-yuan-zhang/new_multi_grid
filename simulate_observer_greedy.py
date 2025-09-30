#!/usr/bin/env python3
"""
Simulate the BeliefUpdateObserver's greedy action selection for instance 565
to understand why actor goes in and out of view
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast

def parse_instance_data():
    """Parse instance 565 data from the CSV"""
    df = pd.read_csv('evaluation_results_1759207042.csv')
    row_565 = df.iloc[565]
    
    return {
        'size': int(row_565['size'].iloc[0] if hasattr(row_565['size'], 'iloc') else row_565['size']),
        'behavior': str(row_565['hidden_cost_style'].iloc[0] if hasattr(row_565['hidden_cost_style'], 'iloc') else row_565['hidden_cost_style']),
        'observer_start_pos': eval(str(row_565['observer_pos'].iloc[0] if hasattr(row_565['observer_pos'], 'iloc') else row_565['observer_pos'])),
        'target_start_pos': eval(str(row_565['target_pos'].iloc[0] if hasattr(row_565['target_pos'], 'iloc') else row_565['target_pos'])),
        'observer_start_dir': int(row_565['observer_dir'].iloc[0] if hasattr(row_565['observer_dir'], 'iloc') else row_565['observer_dir']),
        'target_start_dir': int(row_565['target_dir'].iloc[0] if hasattr(row_565['target_dir'], 'iloc') else row_565['target_dir']),
        'initial_distance': int(row_565['initial_distance'].iloc[0] if hasattr(row_565['initial_distance'], 'iloc') else row_565['initial_distance']),
        'total_steps': int(row_565['total_steps'].iloc[0] if hasattr(row_565['total_steps'], 'iloc') else row_565['total_steps']),
        'success': bool(row_565['eval_success'].iloc[0] if hasattr(row_565['eval_success'], 'iloc') else row_565['eval_success']),
        'actions': str(row_565['all_actions'].iloc[0] if hasattr(row_565['all_actions'], 'iloc') else row_565['all_actions'])
    }

def simulate_observer_greedy_behavior(instance_data, max_steps=20):
    """
    Simulate the observer's greedy behavior based on the actual implementation:
    1. If moving forward reduces distance to target, move forward
    2. Otherwise, turn toward the direction that points to the target
    """
    
    print("=== OBSERVER GREEDY SIMULATION ===")
    print(f"Grid size: {instance_data['size']}x{instance_data['size']}")
    print(f"Target behavior: {instance_data['behavior']}")
    print(f"Observer start: {instance_data['observer_start_pos']}, dir: {instance_data['observer_start_dir']}")
    print(f"Target start: {instance_data['target_start_pos']}, dir: {instance_data['target_start_dir']}")
    print(f"Initial distance: {instance_data['initial_distance']}")
    print()
    
    # Initialize positions and directions
    obs_pos = list(instance_data['observer_start_pos'])
    target_pos = list(instance_data['target_start_pos'])
    obs_dir = instance_data['observer_start_dir']
    target_dir = instance_data['target_start_dir']
    
    # Direction mappings (0: right, 1: down, 2: left, 3: up)
    direction_vectors = {
        0: (1, 0),   # right
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (0, -1)   # up
    }
    
    direction_names = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    action_names = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward'}
    
    # Simulation logs
    step_log = []
    visibility_log = []
    action_log = []
    
    for step in range(max_steps):
        print(f"\nStep {step}:")
        print(f"  Observer: pos={obs_pos}, dir={obs_dir} ({direction_names[obs_dir]})")
        print(f"  Target: pos={target_pos}")
        
        # Check if target is visible (within 5x5 observation window)
        # The 5x5 window is centered on observer's position
        view_radius = 2  # 5x5 means 2 cells in each direction
        target_visible = (
            abs(obs_pos[0] - target_pos[0]) <= view_radius and
            abs(obs_pos[1] - target_pos[1]) <= view_radius
        )
        
        print(f"  Target visible: {target_visible} (distance: {abs(obs_pos[0] - target_pos[0]) + abs(obs_pos[1] - target_pos[1])})")
        
        # Log data
        step_log.append({
            'step': step,
            'obs_pos': obs_pos.copy(),
            'target_pos': target_pos.copy(),
            'obs_dir': obs_dir,
            'target_visible': target_visible
        })
        
        visibility_log.append(target_visible)
        
        # === OBSERVER GREEDY ACTION SELECTION ===
        # This replicates the BeliefUpdateObserver.greedy() method
        
        current_distance = abs(obs_pos[0] - target_pos[0]) + abs(obs_pos[1] - target_pos[1])
        print(f"  Current Manhattan distance: {current_distance}")
        
        # Check if moving forward reduces distance
        forward_vec = direction_vectors[obs_dir]
        forward_pos = [obs_pos[0] + forward_vec[0], obs_pos[1] + forward_vec[1]]
        
        # Check bounds
        grid_size = instance_data['size']
        forward_valid = (0 <= forward_pos[0] < grid_size and 0 <= forward_pos[1] < grid_size)
        
        if forward_valid:
            forward_distance = abs(forward_pos[0] - target_pos[0]) + abs(forward_pos[1] - target_pos[1])
            print(f"  Forward move would: pos={forward_pos}, distance={forward_distance}")
            
            if forward_distance < current_distance:
                # Move forward - it reduces distance
                action = 2  # move_forward
                obs_pos = forward_pos
                print(f"  Action: {action_names[action]} (reduces distance)")
            else:
                # Turn toward target
                action = choose_turn_action(obs_pos, target_pos, obs_dir, direction_vectors)
                obs_dir = execute_turn(obs_dir, action)
                print(f"  Action: {action_names[action]} (turn toward target)")
        else:
            # Can't move forward (would hit boundary), turn toward target
            action = choose_turn_action(obs_pos, target_pos, obs_dir, direction_vectors)
            obs_dir = execute_turn(obs_dir, action)
            print(f"  Action: {action_names[action]} (blocked, turn toward target)")
        
        action_log.append(action)
        
        # === TARGET MOVEMENT (simplified hate_wall behavior) ===
        # Target tries to move away from walls/boundaries
        target_moved = False
        
        # Try to move in a direction that's away from boundaries
        possible_moves = []
        for dir_val, vec in direction_vectors.items():
            new_pos = [target_pos[0] + vec[0], target_pos[1] + vec[1]]
            
            if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
                # Calculate "wall penalty" - how close to edges
                edge_penalty = 0
                if new_pos[0] <= 1 or new_pos[0] >= grid_size - 2:
                    edge_penalty += 1
                if new_pos[1] <= 1 or new_pos[1] >= grid_size - 2:
                    edge_penalty += 1
                
                possible_moves.append((new_pos, edge_penalty, dir_val))
        
        if possible_moves:
            # For hate_wall behavior, prefer moves with lower edge penalty
            possible_moves.sort(key=lambda x: x[1])  # Sort by edge penalty
            
            # Add some randomness but bias toward lower penalties
            if np.random.random() < 0.7 and len(possible_moves) > 0:
                # Choose best move (lowest penalty)
                target_pos, _, target_dir = possible_moves[0]
                target_moved = True
            elif len(possible_moves) > 1:
                # Choose second best or random
                target_pos, _, target_dir = possible_moves[min(1, len(possible_moves)-1)]
                target_moved = True
        
        if not target_moved:
            print(f"  Target stayed at {target_pos}")
        else:
            print(f"  Target moved to {target_pos}")
    
    return step_log, visibility_log, action_log

def choose_turn_action(obs_pos, target_pos, obs_dir, direction_vectors):
    """
    Choose which way to turn to face toward the target.
    This replicates the logic from BeliefUpdateObserver.greedy()
    """
    
    # Calculate the direction vector from current position to target
    target_vec = (target_pos[0] - obs_pos[0], target_pos[1] - obs_pos[1])
    
    # Find the best direction to face
    best_direction = None
    max_dot_product = -float('inf')
    
    # Check all 4 directions to find the one most aligned with target vector
    for direction in range(4):
        dir_vec = direction_vectors[direction]
        
        # Calculate dot product to measure alignment (higher is better)
        dot_product = dir_vec[0] * target_vec[0] + dir_vec[1] * target_vec[1]
        
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            best_direction = direction
    
    # If already facing the right direction, this shouldn't be called
    if best_direction is None or best_direction == obs_dir:
        return 0  # turn_left as default
    
    # Calculate the shortest turn (left or right) to reach best direction
    turn_diff = (best_direction - obs_dir) % 4
    
    if turn_diff == 1 or turn_diff == -3:
        return 1  # turn_right
    elif turn_diff == 3 or turn_diff == -1:
        return 0  # turn_left
    else:
        # 180 degree turn needed, choose left
        return 0  # turn_left

def execute_turn(current_dir, action):
    """Execute a turn action and return new direction"""
    if action == 0:  # turn_left
        return (current_dir - 1) % 4
    elif action == 1:  # turn_right
        return (current_dir + 1) % 4
    else:
        return current_dir  # move_forward doesn't change direction

def analyze_visibility_pattern(step_log, visibility_log, action_log):
    """Analyze the visibility pattern and identify causes"""
    
    print("\n=== VISIBILITY ANALYSIS ===")
    
    total_steps = len(visibility_log)
    visible_steps = sum(visibility_log)
    visibility_ratio = visible_steps / total_steps if total_steps > 0 else 0
    
    print(f"Target visible: {visible_steps}/{total_steps} steps ({visibility_ratio:.1%})")
    
    # Find visibility changes
    visibility_changes = []
    for i in range(1, len(visibility_log)):
        if visibility_log[i] != visibility_log[i-1]:
            change_type = "appeared" if visibility_log[i] else "disappeared"
            visibility_changes.append((i, change_type))
    
    print(f"Visibility changes: {len(visibility_changes)}")
    
    # Analyze each visibility change
    action_names = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward'}
    
    for step, change in visibility_changes:
        if step-1 < len(action_log):
            action_taken = action_log[step-1]
            action_name = action_names.get(action_taken, f'action_{action_taken}')
            
            print(f"\n  Step {step}: Target {change} (after {action_name})")
            
            if step < len(step_log):
                step_data = step_log[step]
                prev_data = step_log[step-1] if step-1 >= 0 else None
                
                print(f"    Observer moved from {prev_data['obs_pos'] if prev_data else 'N/A'} to {step_data['obs_pos']}")
                print(f"    Target at: {step_data['target_pos']}")
                
                # Calculate distance change
                if prev_data:
                    prev_dist = abs(prev_data['obs_pos'][0] - prev_data['target_pos'][0]) + abs(prev_data['obs_pos'][1] - prev_data['target_pos'][1])
                    curr_dist = abs(step_data['obs_pos'][0] - step_data['target_pos'][0]) + abs(step_data['obs_pos'][1] - step_data['target_pos'][1])
                    print(f"    Distance changed from {prev_dist} to {curr_dist}")
    
    # Action analysis
    action_counts = {}
    for action in action_log:
        action_name = action_names.get(action, f'action_{action}')
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    
    print(f"\nAction distribution:")
    for action, count in action_counts.items():
        percentage = count / len(action_log) * 100 if len(action_log) > 0 else 0
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    return visibility_changes

def create_visualization(step_log, visibility_log, action_log, instance_data):
    """Create a visualization of the simulation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Grid visualization of movement
    ax1 = axes[0, 0]
    grid_size = instance_data['size']
    
    # Draw grid
    for i in range(grid_size + 1):
        ax1.axhline(y=i-0.5, color='lightgray', linewidth=0.5)
        ax1.axvline(x=i-0.5, color='lightgray', linewidth=0.5)
    
    # Plot observer path
    obs_x = [step['obs_pos'][0] for step in step_log]
    obs_y = [step['obs_pos'][1] for step in step_log]
    ax1.plot(obs_x, obs_y, 'b-o', label='Observer Path', markersize=4)
    
    # Plot target path  
    target_x = [step['target_pos'][0] for step in step_log]
    target_y = [step['target_pos'][1] for step in step_log]
    ax1.plot(target_x, target_y, 'r-s', label='Target Path', markersize=4)
    
    # Mark start positions
    ax1.plot(obs_x[0], obs_y[0], 'bo', markersize=8, label='Observer Start')
    ax1.plot(target_x[0], target_y[0], 'rs', markersize=8, label='Target Start')
    
    ax1.set_xlim(-0.5, grid_size-0.5)
    ax1.set_ylim(-0.5, grid_size-0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Movement Paths on Grid')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Visibility over time
    ax2 = axes[0, 1]
    steps = list(range(len(visibility_log)))
    visibility_values = [1 if v else 0 for v in visibility_log]
    
    ax2.plot(steps, visibility_values, 'ro-', markersize=6, linewidth=2)
    ax2.fill_between(steps, 0, visibility_values, alpha=0.3, color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Target Visible')
    ax2.set_title('Target Visibility Over Time')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Actions over time
    ax3 = axes[1, 0]
    action_names = {0: 'Turn Left', 1: 'Turn Right', 2: 'Move Forward'}
    action_colors = {0: 'purple', 1: 'orange', 2: 'blue'}
    
    action_steps = list(range(len(action_log)))
    for i, action in enumerate(action_log):
        color = action_colors.get(action, 'gray')
        ax3.scatter(i, action, c=color, s=50, alpha=0.7, edgecolors='black')
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Action')
    ax3.set_title('Observer Actions Over Time')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Turn Left', 'Turn Right', 'Move Forward'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance over time
    ax4 = axes[1, 1]
    distances = []
    for step_data in step_log:
        dist = abs(step_data['obs_pos'][0] - step_data['target_pos'][0]) + abs(step_data['obs_pos'][1] - step_data['target_pos'][1])
        distances.append(dist)
    
    ax4.plot(range(len(distances)), distances, 'g-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Manhattan Distance')
    ax4.set_title('Observer-Target Distance Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('instance_565_observer_simulation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'instance_565_observer_simulation.png'")
    plt.show()

def main():
    """Main simulation function"""
    
    # Parse instance data
    instance_data = parse_instance_data()
    
    print("=== INSTANCE 565 OBSERVER SIMULATION ===")
    print(f"Analyzing why actor goes 'in and out of view'")
    print(f"Original result: Success={instance_data['success']}, Total steps={instance_data['total_steps']}")
    print()
    
    # Run simulation
    step_log, visibility_log, action_log = simulate_observer_greedy_behavior(
        instance_data, max_steps=15
    )
    
    # Analyze results
    visibility_changes = analyze_visibility_pattern(step_log, visibility_log, action_log)
    
    # Create visualization
    create_visualization(step_log, visibility_log, action_log, instance_data)
    
    print(f"\n=== KEY INSIGHTS ===")
    
    if len(visibility_changes) > 3:
        print("🔍 HIGH VISIBILITY INSTABILITY DETECTED!")
        print("   The greedy observer frequently loses and regains sight of the target")
        print("   This is caused by:")
        print("   1. Local optimization - observer only considers immediate distance reduction")  
        print("   2. No lookahead - doesn't consider that target is also moving")
        print("   3. Turning behavior - when blocked, observer turns which can lose sight")
    
    turn_actions = sum(1 for action in action_log if action in [0, 1])
    if turn_actions > len(action_log) * 0.4:
        print(f"\n🔄 EXCESSIVE TURNING: {turn_actions}/{len(action_log)} actions are turns")
        print("   This suggests the observer gets 'confused' about optimal direction")
    
    visibility_ratio = sum(visibility_log) / len(visibility_log) if len(visibility_log) > 0 else 0
    if visibility_ratio < 0.6:
        print(f"\n👁️ LOW VISIBILITY: Target only visible {visibility_ratio:.1%} of the time") 
        print("   The observer's movement pattern is causing frequent loss of target")
    
    print(f"\n💡 ROOT CAUSE ANALYSIS:")
    print(f"   The 'in and out of view' phenomenon is caused by the greedy algorithm's")
    print(f"   myopic optimization that doesn't account for:")
    print(f"   - Target movement prediction")
    print(f"   - Observation window constraints (5x5 view)")
    print(f"   - Multi-step planning to maintain visibility")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible target movement
    main()
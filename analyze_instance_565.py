#!/usr/bin/env python3
"""
Deep analysis of instance 565 to understand why actor goes in and out of view
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def analyze_instance_565():
    """Analyze instance 565 in detail"""
    
    # Load the data
    df = pd.read_csv('evaluation_results_1759207042.csv')
    row_565 = df.iloc[565]
    
    print("=== INSTANCE 565 ANALYSIS ===")
    print(f"Grid Size: {row_565['size']}x{row_565['size']}")
    print(f"Behavior: {row_565['hidden_cost_style']}")
    print(f"Observer Start: {row_565['observer_pos']}")
    print(f"Target Start: {row_565['target_pos']}")
    print(f"Initial Distance: {row_565['initial_distance']}")
    print(f"Total Steps: {row_565['total_steps']}")
    print(f"Success: {row_565['eval_success']}")
    print(f"Convergence Step: {row_565['eval_convergence_step']}")
    print()
    
    # Parse the observations
    all_obs = ast.literal_eval(str(row_565['all_obs']))
    all_actions = ast.literal_eval(str(row_565['all_actions']))
    
    print(f"Number of observations: {len(all_obs)}")
    print(f"Number of actions: {len(all_actions)}")
    print()
    
    # Analyze each step
    actor_visible_steps = []
    observer_positions = []
    target_positions = []
    
    for step, obs in enumerate(all_obs):
        # Parse observation data
        image = np.array(obs['image'])
        direction = obs['direction']
        target_pos = obs.get('target_pos', None)
        
        # Check if actor (red object) is visible in the image
        # Actor is represented as [1, 0, 0] (red)
        actor_visible = np.any(np.all(image == [1, 0, 0], axis=-1))
        actor_visible_steps.append(actor_visible)
        
        # Extract positions if available
        if target_pos:
            target_positions.append((target_pos[0], target_pos[1]))
        
        print(f"Step {step}: Actor visible: {actor_visible}, Direction: {direction}")
        if target_pos:
            print(f"  Target position: {target_pos}")
        
        # Show where actor appears in the 5x5 view if visible
        if actor_visible:
            actor_positions = np.where(np.all(image == [1, 0, 0], axis=-1))
            if len(actor_positions[0]) > 0:
                print(f"  Actor in view at: {list(zip(actor_positions[0], actor_positions[1]))}")
    
    print()
    print("=== VISIBILITY PATTERN ANALYSIS ===")
    
    # Analyze visibility pattern
    visibility_changes = []
    for i in range(1, len(actor_visible_steps)):
        if actor_visible_steps[i] != actor_visible_steps[i-1]:
            change_type = "appeared" if actor_visible_steps[i] else "disappeared"
            visibility_changes.append((i, change_type))
    
    print(f"Actor visibility changes: {len(visibility_changes)}")
    for step, change in visibility_changes:
        action_taken = all_actions[step-1] if step-1 < len(all_actions) else "N/A"
        print(f"  Step {step}: Actor {change} (after action: {action_taken})")
    
    # Calculate visibility statistics
    total_visible = sum(actor_visible_steps)
    visibility_ratio = total_visible / len(actor_visible_steps)
    
    print(f"Actor visible in {total_visible}/{len(actor_visible_steps)} steps ({visibility_ratio:.2%})")
    
    print()
    print("=== ACTION ANALYSIS ===")
    
    # Analyze actions taken
    action_counts = {}
    for action in all_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("Actions taken:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} times")
    
    # Look for patterns in actions when actor disappears/appears
    print()
    print("=== ACTIONS AROUND VISIBILITY CHANGES ===")
    
    for step, change in visibility_changes:
        print(f"\nStep {step} - Actor {change}:")
        # Show actions before and after
        for offset in range(-2, 3):
            action_step = step + offset - 1  # -1 because actions are 0-indexed relative to observations
            if 0 <= action_step < len(all_actions):
                marker = " -> " if offset == 0 else "    "
                print(f"{marker}Action {action_step}: {all_actions[action_step]}")
    
    return {
        'visibility_changes': len(visibility_changes),
        'visibility_ratio': visibility_ratio,
        'total_steps': len(actor_visible_steps),
        'action_counts': action_counts
    }

def visualize_observer_movement():
    """Create a visualization of observer movement and actor visibility"""
    
    df = pd.read_csv('evaluation_results_1759207042.csv')
    row_565 = df.iloc[565]
    
    all_obs = ast.literal_eval(str(row_565['all_obs']))
    all_actions = ast.literal_eval(str(row_565['all_actions']))
    
    # Track visibility over time
    steps = list(range(len(all_obs)))
    visibility = []
    
    for obs in all_obs:
        image = np.array(obs['image'])
        actor_visible = np.any(np.all(image == [1, 0, 0], axis=-1))
        visibility.append(1 if actor_visible else 0)
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot 1: Actor visibility over time
    plt.subplot(2, 1, 1)
    plt.plot(steps, visibility, 'ro-', markersize=4, linewidth=2)
    plt.title('Actor Visibility Over Time - Instance 565')
    plt.xlabel('Step')
    plt.ylabel('Actor Visible (1=Yes, 0=No)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # Annotate visibility changes
    for i in range(1, len(visibility)):
        if visibility[i] != visibility[i-1]:
            change_type = "Appeared" if visibility[i] else "Disappeared"
            plt.annotate(f'{change_type}\n(Step {i})', 
                        xy=(i, visibility[i]), 
                        xytext=(i, visibility[i] + 0.3 if visibility[i] else visibility[i] - 0.3),
                        ha='center', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Plot 2: Actions taken
    plt.subplot(2, 1, 2)
    action_nums = []
    action_labels = []
    
    # Convert actions to numbers for plotting
    action_map = {'move_forward': 0, 'turn_left': 1, 'turn_right': 2}
    
    for action in all_actions:
        if action in action_map:
            action_nums.append(action_map[action])
        else:
            action_nums.append(-1)  # Unknown action
    
    plt.plot(range(len(action_nums)), action_nums, 'bo-', markersize=3, linewidth=1)
    plt.title('Observer Actions Over Time')
    plt.xlabel('Step')
    plt.ylabel('Action Type')
    plt.yticks([0, 1, 2], ['Move Forward', 'Turn Left', 'Turn Right'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('instance_565_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'instance_565_analysis.png'")

if __name__ == "__main__":
    # Run the analysis
    results = analyze_instance_565()
    
    print()
    print("=== SUMMARY ===")
    print(f"Visibility changes: {results['visibility_changes']}")
    print(f"Actor visible {results['visibility_ratio']:.1%} of the time")
    print(f"Most common action: {max(results['action_counts'], key=results['action_counts'].get)}")
    
    # Create visualization
    visualize_observer_movement()
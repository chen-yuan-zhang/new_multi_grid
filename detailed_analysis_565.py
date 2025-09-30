#!/usr/bin/env python3
"""
Deep analysis of instance 565 to understand why actor goes in and out of view
Using a different parsing approach for the observation data
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def extract_actor_visibility_pattern():
    """Extract and analyze actor visibility pattern from instance 565"""
    
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
    
    # Get the observations string
    obs_str = str(row_565['all_obs'])
    actions_str = str(row_565['all_actions'])
    
    print(f"Observations data length: {len(obs_str)} characters")
    print(f"Actions data: {actions_str}")
    
    # Parse actions using regex
    action_matches = re.findall(r"'(move_forward|turn_left|turn_right)'", actions_str)
    print(f"Extracted {len(action_matches)} actions: {action_matches}")
    
    # Extract image arrays to check for actor presence
    # Look for patterns that indicate actor (red = [1, 0, 0]) in the 5x5 view
    
    # Find all image array sections
    image_pattern = r"'image': array\(\[\[\[(.*?)\]\]\]\)"
    image_matches = re.findall(image_pattern, obs_str, re.DOTALL)
    
    print(f"\nFound {len(image_matches)} image observations")
    
    actor_visibility = []
    
    # For each image, check if actor ([1, 0, 0]) is present
    for i, image_data in enumerate(image_matches):
        # Count occurrences of [1, 0, 0] pattern (actor/red object)
        red_count = image_data.count('[1, 0, 0]')
        actor_visible = red_count > 0
        actor_visibility.append(actor_visible)
        
        if i < 10:  # Show first 10 for debugging
            print(f"Step {i}: Red pixels: {red_count}, Actor visible: {actor_visible}")
    
    # Analyze visibility pattern
    print(f"\n=== VISIBILITY PATTERN ===")
    print(f"Total observations: {len(actor_visibility)}")
    
    if len(actor_visibility) > 0:
        total_visible = sum(actor_visibility)
        visibility_ratio = total_visible / len(actor_visibility)
        print(f"Actor visible in {total_visible}/{len(actor_visibility)} steps ({visibility_ratio:.2%})")
        
        # Find visibility changes
        visibility_changes = []
        for i in range(1, len(actor_visibility)):
            if actor_visibility[i] != actor_visibility[i-1]:
                change_type = "appeared" if actor_visibility[i] else "disappeared"
                visibility_changes.append((i, change_type))
        
        print(f"\nVisibility changes: {len(visibility_changes)}")
        for step, change in visibility_changes:
            action_taken = action_matches[step-1] if step-1 < len(action_matches) else "N/A"
            print(f"  Step {step}: Actor {change} (after action: {action_taken})")
        
        # Show visibility sequence
        visibility_str = ''.join(['1' if v else '0' for v in actor_visibility])
        print(f"\nVisibility sequence: {visibility_str}")
        
        # Analyze action patterns around visibility changes
        if len(visibility_changes) > 0:
            print(f"\n=== ACTION PATTERNS AROUND VISIBILITY CHANGES ===")
            for step, change in visibility_changes:
                print(f"\nStep {step} - Actor {change}:")
                # Show actions around this change
                for offset in range(-2, 3):
                    action_step = step + offset - 1
                    if 0 <= action_step < len(action_matches):
                        marker = " -> " if offset == 0 else "    "
                        print(f"{marker}Action {action_step}: {action_matches[action_step]}")
    
    return actor_visibility, action_matches, visibility_changes

def analyze_movement_pattern():
    """Analyze the movement pattern that causes visibility issues"""
    
    actor_visibility, actions, visibility_changes = extract_actor_visibility_pattern()
    
    if len(actions) == 0:
        print("No actions found to analyze")
        return
    
    print(f"\n=== MOVEMENT ANALYSIS ===")
    
    # Count action types
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("Action distribution:")
    for action, count in action_counts.items():
        percentage = count / len(actions) * 100
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    # Look for turning patterns that might cause visibility issues
    consecutive_turns = 0
    max_consecutive_turns = 0
    turn_sequences = []
    current_sequence = []
    
    for i, action in enumerate(actions):
        if 'turn' in action:
            consecutive_turns += 1
            current_sequence.append((i, action))
        else:
            if consecutive_turns > 0:
                turn_sequences.append(current_sequence.copy())
                max_consecutive_turns = max(max_consecutive_turns, consecutive_turns)
            consecutive_turns = 0
            current_sequence = []
    
    # Don't forget the last sequence if it ends with turns
    if consecutive_turns > 0:
        turn_sequences.append(current_sequence.copy())
        max_consecutive_turns = max(max_consecutive_turns, consecutive_turns)
    
    print(f"\nTurning pattern analysis:")
    print(f"Maximum consecutive turns: {max_consecutive_turns}")
    print(f"Number of turn sequences: {len(turn_sequences)}")
    
    if turn_sequences:
        print("\nTurn sequences:")
        for i, seq in enumerate(turn_sequences[:5]):  # Show first 5
            seq_str = " -> ".join([f"{step}:{action}" for step, action in seq])
            print(f"  Sequence {i+1}: {seq_str}")
    
    # Check if visibility changes correlate with turning
    turn_related_changes = 0
    if len(visibility_changes) > 0:
        print(f"\n=== CORRELATION BETWEEN TURNS AND VISIBILITY ===")
        for step, change in visibility_changes:
            # Check if this change happened around turning actions
            around_turn = False
            for offset in range(-2, 3):
                action_step = step + offset - 1
                if 0 <= action_step < len(actions) and 'turn' in actions[action_step]:
                    around_turn = True
                    break
            
            if around_turn:
                turn_related_changes += 1
                print(f"  Step {step} ({change}): Related to turning")
            else:
                print(f"  Step {step} ({change}): Not related to turning")
        
        print(f"\nVisibility changes related to turning: {turn_related_changes}/{len(visibility_changes)} ({turn_related_changes/len(visibility_changes)*100:.1f}%)")
    
    return action_counts, turn_sequences

def create_detailed_visualization():
    """Create a detailed visualization of the visibility and movement pattern"""
    
    actor_visibility, actions, visibility_changes = extract_actor_visibility_pattern()
    
    if len(actor_visibility) == 0:
        print("No visibility data to visualize")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    steps = list(range(len(actor_visibility)))
    visibility_values = [1 if v else 0 for v in actor_visibility]
    
    # Plot 1: Actor visibility
    ax1.plot(steps, visibility_values, 'ro-', markersize=6, linewidth=2, markerfacecolor='red', markeredgecolor='darkred')
    ax1.fill_between(steps, 0, visibility_values, alpha=0.3, color='red')
    ax1.set_title('Instance 565: Actor Visibility Pattern', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Actor Visible')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['No', 'Yes'])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)
    
    # Annotate visibility changes
    for step, change in visibility_changes:
        color = 'green' if change == 'appeared' else 'orange'
        ax1.annotate(f'{change.title()}\n(Step {step})', 
                    xy=(step, visibility_values[step] if step < len(visibility_values) else 0), 
                    xytext=(step, 1.1 if change == 'appeared' else -0.05),
                    ha='center', fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Plot 2: Actions
    if len(actions) > 0:
        action_map = {'move_forward': 2, 'turn_left': 1, 'turn_right': 0}
        action_nums = [action_map.get(action, -1) for action in actions]
        action_steps = list(range(len(actions)))
        
        # Color code the actions
        colors = []
        for action in actions:
            if action == 'move_forward':
                colors.append('blue')
            elif action == 'turn_left':
                colors.append('purple')
            elif action == 'turn_right':
                colors.append('orange')
            else:
                colors.append('gray')
        
        ax2.scatter(action_steps, action_nums, c=colors, s=50, alpha=0.7, edgecolors='black')
        ax2.plot(action_steps, action_nums, 'k-', alpha=0.3, linewidth=1)
        
        ax2.set_title('Observer Actions Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Action Type')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Turn Right', 'Turn Left', 'Move Forward'])
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Move Forward'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Turn Left'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Turn Right')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('instance_565_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'instance_565_detailed_analysis.png'")
    plt.show()

if __name__ == "__main__":
    print("🔍 DEEP ANALYSIS OF INSTANCE 565")
    print("=" * 50)
    
    # Run the analysis
    actor_visibility, actions, visibility_changes = extract_actor_visibility_pattern()
    
    if len(actor_visibility) > 0:
        action_counts, turn_sequences = analyze_movement_pattern()
        
        print(f"\n=== SUMMARY ===")
        print(f"Total steps: {len(actor_visibility)}")
        print(f"Visibility changes: {len(visibility_changes)}")
        print(f"Actor visible: {sum(actor_visibility)}/{len(actor_visibility)} steps ({sum(actor_visibility)/len(actor_visibility)*100:.1f}%)")
        print(f"Most common action: {max(action_counts.keys(), key=lambda k: action_counts[k]) if action_counts else 'N/A'}")
        print(f"Maximum consecutive turns: {max([len(seq) for seq in turn_sequences]) if turn_sequences else 0}")
        
        # Create detailed visualization
        create_detailed_visualization()
        
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Analysis of why actor goes in and out of view
        if len(visibility_changes) > 2:
            print(f"• High visibility instability: {len(visibility_changes)} changes in {len(actor_visibility)} steps")
            print(f"• This suggests the observer is moving in a way that frequently loses/gains sight of the actor")
        
        if action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0) > len(actions) * 0.5:
            print(f"• Excessive turning: {(action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0))/len(actions)*100:.1f}% of actions are turns")
            print(f"• This likely causes the actor to move in and out of the 5x5 observation window")
        
        if not any(actor_visibility):
            print(f"• Actor never visible: Observer completely lost track of the target")
        elif all(actor_visibility):
            print(f"• Actor always visible: Good tracking but convergence failed for other reasons")
        else:
            print(f"• Intermittent visibility: Observer partially tracking the target")
    else:
        print("❌ Could not extract visibility data")
#!/usr/bin/env python3
"""
Visualize a scenario from the dataset to manually verify if the target plan makes sense.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.core.actions import Action

def direction_to_arrow(dir_int):
    """Convert direction integer to arrow symbol."""
    # 0: right, 1: down, 2: left, 3: up
    arrows = ['→', '↓', '←', '↑']
    return arrows[dir_int]

def direction_to_vector(dir_int):
    """Convert direction to (dx, dy) for visualization."""
    # 0: right, 1: down, 2: left, 3: up
    vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    return vectors[dir_int]

def visualize_scenario(scenario_idx, df):
    """Visualize a scenario with grid, goals, start positions, and planned trajectory."""
    
    row = df.iloc[scenario_idx]
    
    # Parse data
    base_grid = np.array(json.loads(row['base_grid']))
    hidden_cost = np.array(json.loads(row['hidden_cost']))
    goals = eval(row['goals'])
    goal = eval(row['goal'])
    observer_pos = eval(row['observer_pos'])
    target_pos = eval(row['target_pos'])
    observer_dir = row['observer_dir']
    target_dir = row['target_dir']
    all_actions = [Action(v) for v in json.loads(row['all_actions'])]
    
    print(f"Scenario {scenario_idx}:")
    print(f"  Grid size: {base_grid.shape}")
    print(f"  Observer: pos={observer_pos}, dir={observer_dir} {direction_to_arrow(observer_dir)}")
    print(f"  Target: pos={target_pos}, dir={target_dir} {direction_to_arrow(target_dir)}")
    print(f"  Goal: {goal}")
    print(f"  All goals: {goals}")
    print(f"  Trajectory length: {len(all_actions)}")
    print(f"  Hidden cost style: {row['hidden_cost_style']}")
    print(f"  Actions: {[a.name for a in all_actions]}")
    print()
    
    # Simulate the trajectory to see where target actually goes
    current_pos = target_pos
    current_dir = target_dir
    trajectory_positions = [current_pos]
    
    print("Simulating trajectory:")
    print(f"  Start: pos={current_pos}, dir={current_dir} {direction_to_arrow(current_dir)}")
    
    for i, action in enumerate(all_actions):
        if action == Action.left:
            current_dir = (current_dir - 1) % 4
            print(f"  Step {i}: Turn LEFT -> now facing {direction_to_arrow(current_dir)}")
        elif action == Action.right:
            current_dir = (current_dir + 1) % 4
            print(f"  Step {i}: Turn RIGHT -> now facing {direction_to_arrow(current_dir)}")
        elif action == Action.forward:
            # Calculate next position
            dx, dy = direction_to_vector(current_dir)
            next_pos = (current_pos[0] + dy, current_pos[1] + dx)  # Note: (row, col) format
            
            # Check if blocked by wall
            if (0 <= next_pos[0] < base_grid.shape[0] and 
                0 <= next_pos[1] < base_grid.shape[1] and 
                base_grid[next_pos] == 0):  # 0 = free, 1 = wall
                current_pos = next_pos
                trajectory_positions.append(current_pos)
                print(f"  Step {i}: Move FORWARD -> now at {current_pos}")
            else:
                print(f"  Step {i}: Move FORWARD BLOCKED (wall at {next_pos}) -> stay at {current_pos}")
        elif action == Action.stay:
            print(f"  Step {i}: STAY at {current_pos}")
    
    print(f"  Final position: {current_pos}")
    print(f"  Goal position: {goal}")
    print(f"  Reached goal: {current_pos == goal}")
    print(f"  Unique positions visited: {len(set(trajectory_positions))}")
    print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Grid with walls and hidden costs
    ax1.set_title(f'Scenario {scenario_idx}: Grid Layout\n(Hidden cost style: {row["hidden_cost_style"]})', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # Show grid
    for i in range(base_grid.shape[0]):
        for j in range(base_grid.shape[1]):
            if base_grid[i, j] == 1:  # Wall
                rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                ax1.add_patch(rect)
            else:  # Free space - color by hidden cost
                cost_val = hidden_cost[i, j]
                # Normalize cost for coloring (darker = higher cost)
                if cost_val < 10:
                    color = plt.cm.YlOrRd(cost_val / 10)  # 0-10 range
                else:
                    color = 'lightgray'
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5, edgecolor='gray', facecolor=color, alpha=0.6)
                ax1.add_patch(rect)
                # Show cost value
                if cost_val < 10:
                    ax1.text(j + 0.5, i + 0.5, f'{cost_val:.1f}', ha='center', va='center', fontsize=6, color='black')
    
    # Mark all goals
    for g in goals:
        if g == goal:
            # True goal - green star
            ax1.plot(g[1] + 0.5, g[0] + 0.5, marker='*', markersize=20, color='green', markeredgecolor='darkgreen', markeredgewidth=2)
            ax1.text(g[1] + 0.5, g[0] + 0.2, 'TRUE\nGOAL', ha='center', va='top', fontsize=8, fontweight='bold', color='darkgreen')
        else:
            # Other goals - yellow star
            ax1.plot(g[1] + 0.5, g[0] + 0.5, marker='*', markersize=15, color='yellow', markeredgecolor='orange', markeredgewidth=1.5)
    
    # Mark observer start
    ax1.plot(observer_pos[1] + 0.5, observer_pos[0] + 0.5, marker='o', markersize=12, color='blue', markeredgecolor='darkblue', markeredgewidth=2)
    # Draw observer direction arrow
    dx, dy = direction_to_vector(observer_dir)
    ax1.arrow(observer_pos[1] + 0.5, observer_pos[0] + 0.5, dx * 0.3, dy * 0.3, head_width=0.15, head_length=0.1, fc='darkblue', ec='darkblue', linewidth=2)
    ax1.text(observer_pos[1] + 0.5, observer_pos[0] - 0.3, 'OBS', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkblue')
    
    # Mark target start
    ax1.plot(target_pos[1] + 0.5, target_pos[0] + 0.5, marker='s', markersize=12, color='red', markeredgecolor='darkred', markeredgewidth=2)
    # Draw target direction arrow
    dx, dy = direction_to_vector(target_dir)
    ax1.arrow(target_pos[1] + 0.5, target_pos[0] + 0.5, dx * 0.3, dy * 0.3, head_width=0.15, head_length=0.1, fc='darkred', ec='darkred', linewidth=2)
    ax1.text(target_pos[1] + 0.5, target_pos[0] - 0.3, 'START', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkred')
    
    ax1.set_xlim(0, base_grid.shape[1])
    ax1.set_ylim(0, base_grid.shape[0])
    ax1.invert_yaxis()
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Planned trajectory
    ax2.set_title(f'Scenario {scenario_idx}: Planned Trajectory\n({len(all_actions)} actions, {len(set(trajectory_positions))} unique positions)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    
    # Show grid (simplified)
    for i in range(base_grid.shape[0]):
        for j in range(base_grid.shape[1]):
            if base_grid[i, j] == 1:  # Wall
                rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
                ax2.add_patch(rect)
            else:  # Free space
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5, edgecolor='lightgray', facecolor='white')
                ax2.add_patch(rect)
    
    # Mark goal
    ax2.plot(goal[1] + 0.5, goal[0] + 0.5, marker='*', markersize=20, color='green', markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Draw trajectory
    if len(trajectory_positions) > 1:
        # Draw path
        trajectory_array = np.array(trajectory_positions)
        ax2.plot(trajectory_array[:, 1] + 0.5, trajectory_array[:, 0] + 0.5, 'r-', linewidth=2, alpha=0.7, label='Path')
        
        # Mark positions with numbers
        for idx, pos in enumerate(trajectory_positions[::max(1, len(trajectory_positions)//10)]):  # Show every Nth position
            ax2.text(pos[1] + 0.5, pos[0] + 0.5, str(idx), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='circle', facecolor='red', alpha=0.7))
    else:
        # Agent didn't move - show warning
        ax2.text(target_pos[1] + 0.5, target_pos[0] + 0.5, '⚠️\nNO\nMOVE', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Mark start
    ax2.plot(target_pos[1] + 0.5, target_pos[0] + 0.5, marker='s', markersize=15, color='blue', markeredgecolor='darkblue', markeredgewidth=2, label='Start')
    
    # Mark end
    final_pos = trajectory_positions[-1]
    ax2.plot(final_pos[1] + 0.5, final_pos[0] + 0.5, marker='o', markersize=15, color='orange', markeredgecolor='darkorange', markeredgewidth=2, label='End')
    
    ax2.set_xlim(0, base_grid.shape[1])
    ax2.set_ylim(0, base_grid.shape[0])
    ax2.invert_yaxis()
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'scenario_{scenario_idx}_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    
    return current_pos == goal, len(set(trajectory_positions))


def main():
    # Load dataset
    dataset_path = "results_test_new.csv"
    print(f"📖 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Found {len(df)} scenarios\n")
    
    # Visualize first few scenarios
    scenarios_to_check = [0, 4, 10, 50, 100]  # Check a variety
    
    results = []
    for idx in scenarios_to_check:
        if idx < len(df):
            print("="*80)
            reached_goal, unique_positions = visualize_scenario(idx, df)
            results.append({
                'scenario': idx,
                'reached_goal': reached_goal,
                'unique_positions': unique_positions
            })
            print()
    
    print("="*80)
    print("\n📊 SUMMARY:")
    for r in results:
        status = "✅ Reached goal" if r['reached_goal'] else "❌ Did NOT reach goal"
        print(f"  Scenario {r['scenario']}: {status} | {r['unique_positions']} unique positions")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check how A* computes the plan for scenarios in the CSV and compare with stored actions.
"""

import sys
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.gr_pursuer.astar import astar
from multigrid.gr_pursuer.agents.target import AstarTarget

def visualize_scenario_with_astar(scenario_idx, df):
    """Visualize a scenario and compare A* plan with stored actions."""
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
    stored_actions = [Action(v) for v in json.loads(row['all_actions'])]
    behavior_style = row['hidden_cost_style']
    
    print(f"=" * 80)
    print(f"Scenario {scenario_idx}: {row['size']}x{row['size']} grid, {behavior_style}")
    print(f"=" * 80)
    print(f"Observer: pos={observer_pos}, dir={observer_dir}")
    print(f"Target: pos={target_pos}, dir={target_dir}")
    print(f"Goal: {goal}")
    print(f"Goals: {goals}")
    print(f"Stored actions ({len(stored_actions)}): {[a.name for a in stored_actions]}")
    print()
    
    # Create environment
    env = AGREnv(
        base_grid=base_grid,
        goals=goals,
        goal=goal,
        hidden_cost=hidden_cost,
        enable_hidden_cost=True,
        agents_start_pos=[observer_pos, target_pos],
        agents_start_dir=[observer_dir, target_dir],
        render_mode=None
    )
    
    obs, info = env.reset()
    
    # Method 1: Use AstarTarget agent (what generator_working.py does)
    print("🔍 Method 1: Using AstarTarget agent")
    target_agent = AstarTarget(env)
    first_action = target_agent.compute_action(obs)
    
    if target_agent.path is None:
        print("  ❌ A* failed to find a path!")
        return
    
    print(f"  ✅ A* found path with {len(target_agent.path)} steps")
    print(f"  Path format: [(action, (pos, dir)), ...]")
    print(f"  First few steps:")
    for i in range(min(5, len(target_agent.path))):
        step = target_agent.path[i]
        if step[0] is None:
            print(f"    Step {i}: action=None (initial state), pos={step[1][0]}, dir={step[1][1]}")
        else:
            print(f"    Step {i}: action={step[0].name}, pos={step[1][0]}, dir={step[1][1]}")
    
    # Extract actions from path (how generator_working.py does it)
    astar_actions = [step[0] for step in target_agent.path[1:]]  # Skip initial state
    print(f"\n  Actions from A* path ({len(astar_actions)}): {[a.name for a in astar_actions[:10]]}{'...' if len(astar_actions) > 10 else ''}")
    print()
    
    # Method 2: Direct A* call
    print("🔍 Method 2: Direct A* call")
    pos_state = (target_pos, target_dir)
    direct_path = astar(pos_state, goal, env, cost=hidden_cost)
    
    if direct_path is None:
        print("  ❌ Direct A* failed to find a path!")
    else:
        print(f"  ✅ Direct A* found path with {len(direct_path)} steps")
        direct_actions = [step[0] for step in direct_path[1:]]
        print(f"  Actions from direct path ({len(direct_actions)}): {[a.name for a in direct_actions[:10]]}{'...' if len(direct_actions) > 10 else ''}")
    print()
    
    # Compare with stored actions
    print("📊 Comparison:")
    print(f"  Stored actions length: {len(stored_actions)}")
    print(f"  A* actions length: {len(astar_actions)}")
    
    if len(stored_actions) == len(astar_actions):
        matches = sum(1 for i in range(len(stored_actions)) if stored_actions[i] == astar_actions[i])
        print(f"  ✅ Lengths match!")
        print(f"  Actions match: {matches}/{len(stored_actions)} ({100*matches/len(stored_actions):.1f}%)")
        
        if matches < len(stored_actions):
            print(f"\n  First mismatch:")
            for i in range(len(stored_actions)):
                if stored_actions[i] != astar_actions[i]:
                    print(f"    Step {i}: stored={stored_actions[i].name}, A*={astar_actions[i].name}")
                    break
    else:
        print(f"  ❌ Length mismatch!")
        print(f"     Stored has {len(stored_actions)} actions")
        print(f"     A* produces {len(astar_actions)} actions")
    print()
    
    # Simulate stored actions to see if they reach the goal
    print("🎮 Simulating stored actions:")
    env.reset()
    target_positions = [tuple(target_pos)]
    
    for i, action in enumerate(stored_actions):
        obs, rew, term, trunc, info = env.step({0: Action.stay, 1: action})
        new_pos = tuple(env.agents[1].pos)
        target_positions.append(new_pos)
        
        if i < 5:
            print(f"  Step {i}: action={action.name}, pos={new_pos}, dir={env.agents[1].dir}")
    
    final_pos = tuple(env.agents[1].pos)
    print(f"  ...")
    print(f"  Final position: {final_pos}")
    print(f"  Goal position: {goal}")
    print(f"  Reached goal: {final_pos == goal}")
    print(f"  Unique positions visited: {len(set(target_positions))}")
    print()
    
    # Simulate A* actions
    print("🎮 Simulating A* actions:")
    env.reset()
    astar_positions = [tuple(target_pos)]
    
    for i, action in enumerate(astar_actions):
        obs, rew, term, trunc, info = env.step({0: Action.stay, 1: action})
        new_pos = tuple(env.agents[1].pos)
        astar_positions.append(new_pos)
        
        if i < 5:
            print(f"  Step {i}: action={action.name}, pos={new_pos}, dir={env.agents[1].dir}")
    
    final_pos = tuple(env.agents[1].pos)
    print(f"  ...")
    print(f"  Final position: {final_pos}")
    print(f"  Goal position: {goal}")
    print(f"  Reached goal: {final_pos == goal}")
    print(f"  Unique positions visited: {len(set(astar_positions))}")
    print()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    size = base_grid.shape[0]
    
    for ax_idx, (positions, title) in enumerate([
        (target_positions, f"Stored Actions ({len(stored_actions)} steps)"),
        (astar_positions, f"A* Actions ({len(astar_actions)} steps)")
    ]):
        ax = axes[ax_idx]
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(-0.5, size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Draw grid
        for i in range(size):
            for j in range(size):
                if base_grid[i, j] == 1:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='gray', alpha=0.8)
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            linewidth=0.5, edgecolor='lightgray', 
                                            facecolor='white')
                    ax.add_patch(rect)
        
        # Draw goals
        for g in goals:
            if g == goal:
                circle = patches.Circle((g[1], g[0]), 0.3, 
                                       linewidth=2, edgecolor='darkgreen', 
                                       facecolor='green', alpha=0.7, label='True Goal')
            else:
                circle = patches.Circle((g[1], g[0]), 0.3, 
                                       linewidth=2, edgecolor='gray', 
                                       facecolor='lightgray', alpha=0.5)
            ax.add_patch(circle)
        
        # Draw trajectory
        if len(positions) > 1:
            trajectory_x = [p[1] for p in positions]
            trajectory_y = [p[0] for p in positions]
            ax.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, alpha=0.6, label='Trajectory')
            
            # Mark start and end
            ax.plot(trajectory_x[0], trajectory_y[0], 'ro', markersize=12, label='Start')
            ax.plot(trajectory_x[-1], trajectory_y[-1], 'bs', markersize=12, label='End')
            
            # Add step numbers at key points
            step_interval = max(1, len(positions) // 10)
            for i in range(0, len(positions), step_interval):
                ax.text(trajectory_x[i], trajectory_y[i], str(i), 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'scenario_{scenario_idx}_astar_comparison.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization to: scenario_{scenario_idx}_astar_comparison.png")
    plt.close()


def main():
    df = pd.read_csv('results_test_new.csv')
    print(f"📖 Loaded {len(df)} scenarios from results_test_new.csv\n")
    
    # Check first few scenarios
    for i in range(min(3, len(df))):
        visualize_scenario_with_astar(i, df)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

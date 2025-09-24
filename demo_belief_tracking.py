#!/usr/bin/env python3
"""
Simple demo of belief tracking functionality.

This script demonstrates the key features of the belief tracking system
in a minimal, easy-to-understand way.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver


def main():
    print("🎯 Belief Tracking Demo")
    print("="*40)
    
    # 1. Create environment and agents
    print("1. Setting up environment...")
    env = AGREnv(size=8, num_goals=3, max_steps=50)
    obs, info = env.reset()
    
    print(f"   Goals: {env.goals}")
    print(f"   True goal: {env.goal}")
    print(f"   Observer at: {obs[0]['observer_pos']}")
    
    # 2. Create belief observer
    print("\n2. Creating belief observer...")
    belief_observer = BeliefUpdateObserver(env)
    
    print(f"   Initial goal beliefs: {belief_observer.goal_belief}")
    print(f"   Number of goals tracked: {len(belief_observer.actor_belief)}")
    print(f"   Number of behavior types: {len(belief_observer.actor_belief[env.goals[0]])}")
    
    # 3. Run a few simulation steps
    print("\n3. Running simulation...")
    
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Observer computes action using beliefs
        observer_action = belief_observer.compute_action(obs[0])
        
        # Target takes a simple action
        target_action = [2, 2, 1, 2, 0][step]  # Some preset actions
        
        print(f"Observer action: {observer_action}, Target action: {target_action}")
        
        # Step the environment
        obs, rewards, terminations, truncations, infos = env.step([observer_action, target_action])
        
        # Augment observations with belief data
        augmented_obs = belief_observer.augment_observation(obs)
        
        # Show belief updates
        goal_beliefs = augmented_obs[0]['goal_belief']
        print("Goal beliefs:", {str(k): f"{v:.3f}" for k, v in goal_beliefs.items()})
        
        # Show raw actor belief structure
        actor_belief = augmented_obs[0]['actor_belief']
        first_goal = list(actor_belief.keys())[0]
        behavior_sums = [np.sum(grid) for grid in actor_belief[first_goal]]
        print(f"Actor belief for goal {first_goal} (by behavior): {[f'{s:.4f}' for s in behavior_sums]}")
        
        # Update obs for next step
        obs = augmented_obs
        
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished!")
            break
    
    # 4. Demonstrate different access patterns
    print(f"\n4. Demonstrating belief access patterns...")
    
    # Direct access from belief observer
    print("\nDirect access from BeliefUpdateObserver:")
    print(f"  belief_observer.goal_belief = {belief_observer.goal_belief}")
    print(f"  belief_observer.actor_belief keys = {list(belief_observer.actor_belief.keys())}")
    
    # Access from augmented observations
    print("\nAccess from augmented observations:")
    print(f"  obs['goal_belief'] = {augmented_obs[0]['goal_belief']}")
    print(f"  obs['actor_belief'] keys = {list(augmented_obs[0]['actor_belief'].keys())}")
    print(f"  obs['behavior_patterns'] keys = {list(augmented_obs[0]['behavior_patterns'].keys())}")
    
    # Access specific behavior beliefs
    print("\nAccess specific behavior beliefs:")
    raw_beliefs = augmented_obs[0]['actor_belief']
    goal = list(raw_beliefs.keys())[0]
    print(f"  Goal {goal}:")
    for i, grid in enumerate(raw_beliefs[goal]):
        print(f"    Behavior {i}: shape={grid.shape}, sum={np.sum(grid):.6f}")
    
    print(f"\n✅ Demo completed successfully!")
    print("   The belief tracking system is working and ready to use!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
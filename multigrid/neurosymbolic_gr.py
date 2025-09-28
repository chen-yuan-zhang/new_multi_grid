"""
Neurosymbolic Goal Recognition Demo

This file demonstrates how to use the belief tracking system for goal recognition
in a multi-agent environment. It shows:

1. How to create a BeliefUpdateObserver for tracking multi-behavior actor beliefs
2. How to augment observations with belief distributions
3. How to access goal beliefs and actor beliefs in different formats
4. Integration patterns for the neurosymbolic goal recognition system

Key Components:
- BeliefUpdateObserver: Tracks goal beliefs and multi-behavior actor beliefs  
- Environment augmentation: Adds belief info to observations
- Multi-behavior structure: {goal: [grid_bt0, grid_bt1, grid_bt2, grid_bt3]}

Usage:
    python3 multigrid/neurosymbolic_gr.py

Author: Multi-agent Goal Recognition Team
"""

import numpy as np
import random
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver

def belief_tracking_demo():
    """
    Demonstrate belief tracking and observation augmentation in a multi-agent environment.
    Shows how to:
    1. Create environment with BeliefUpdateObserver
    2. Access belief distributions (goal belief and multi-behavior actor belief)
    3. Augment observations with belief information
    4. Use beliefs for decision making
    """
    print("🎯 Neurosymbolic Goal Recognition Demo")
    print("=" * 50)
    
    # Initialize environment
    print("1. Setting up environment with belief tracking...")
    env = AGREnv(size=8, num_goals=3, max_steps=50)
    obs, info = env.reset()
    
    # Create belief tracker for the observer agent
    belief_observer = BeliefUpdateObserver(env)
    
    print(f"   Goals: {env.goals}")
    print(f"   Observer at: {env.observer.state.pos}")
    print(f"   Target at: {env.target.state.pos}")
    
    print("\n2. Initial belief state...")
    print(f"   Goal beliefs: {belief_observer.goal_belief}")
    print(f"   Number of behavior types tracked: {len(belief_observer.actor_belief[env.goals[0]])}")
    
    print("\n3. Running simulation with belief updates...")
    
    # Run simulation
    max_steps = 10
    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        
        # Observer makes action based on beliefs (greedy planning)
        observer_action = belief_observer.compute_action(obs[0])
        
        # Target makes some action (in practice this would be unknown to observer)
        target_actions = [2, 2, 1, 2, 0, 2, 2, 1, 2, 2]  # Example sequence
        target_action = target_actions[step % len(target_actions)]
        
        # Combine actions (as dict: {agent_id: action})
        actions = {0: observer_action, 1: target_action}
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # IMPORTANT: Augment observations with belief distributions
        augmented_obs = belief_observer.augment_observation(next_obs)
        breakpoint()
        # Display current beliefs
        goal_belief = augmented_obs[0]['goal_belief']
        actor_belief = augmented_obs[0]['actor_belief']
        behavior_patterns = augmented_obs[0]['behavior_patterns']
        
        print(f"Observer action: {observer_action}, Target action: {target_action}")
        print(f"Goal beliefs: {dict((str(k), f'{v:.3f}') for k, v in goal_belief.items())}")
        
        # Show multi-behavior actor belief for each goal
        for goal in actor_belief.keys():
            behavior_sums = [f'{np.sum(grid):.4f}' for grid in actor_belief[goal]]
            print(f"Actor belief for goal {goal} (by behavior): {behavior_sums}")
        # first_goal = list(actor_belief.keys())[0]
        # behavior_sums = [f'{np.sum(grid):.4f}' for grid in actor_belief[first_goal]]
        # print(f"Actor belief for goal {first_goal} (by behavior): {behavior_sums}")
        
        # Check if episode is done
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished")
            break
            
        obs = next_obs
    
    print("\n4. Demonstrating different ways to access beliefs...")
    
    # Method 1: Direct access from BeliefUpdateObserver
    print("\nDirect access from BeliefUpdateObserver:")
    print(f"  belief_observer.goal_belief = {belief_observer.goal_belief}")
    print(f"  belief_observer.actor_belief keys = {list(belief_observer.actor_belief.keys())}")
    
    # Method 2: Access from augmented observations (recommended for integration)
    final_augmented_obs = belief_observer.augment_observation(obs)
    print("\nAccess from augmented observations:")
    print(f"  obs['goal_belief'] = {final_augmented_obs[0]['goal_belief']}")
    print(f"  obs['actor_belief'] keys = {list(final_augmented_obs[0]['actor_belief'].keys())}")
    print(f"  obs['behavior_patterns'] keys = {list(final_augmented_obs[0]['behavior_patterns'].keys())}")
    
    # Method 3: Access specific behavior beliefs
    print("\nAccess specific behavior beliefs:")
    actor_belief = final_augmented_obs[0]['actor_belief']
    first_goal = list(actor_belief.keys())[0]
    for idx, goal in enumerate(actor_belief.keys()):
        print(f"  Goal {goal}:")
        grid_sums = [np.sum(grid) for grid in actor_belief[goal]]
        # check if each grid are identical to each other 
        is_identical = all(np.array_equal(actor_belief[goal][0], grid) for grid in actor_belief[goal])
        print(f"    Are all behavior grids identical? {'Yes' if is_identical else 'No'}")
        # normalize grid_sums 
        total = sum(grid_sums)
        if total > 0:
            grid_sums = [s / total for s in grid_sums]
        else:
            grid_sums = [0 for s in grid_sums]
            
        for i, s in enumerate(grid_sums):
            print(f"    Behavior {i}: normalized sum={s:.6f}")
       
        print("=== end of goal ===")
  
    
    print("\n✅ Demo completed!")
    print("   Key takeaways:")
    print("   - Use BeliefUpdateObserver for belief tracking")
    print("   - Call belief_observer.augment_observation(obs) to add beliefs to observations") 
    print("   - Access beliefs via obs['goal_belief'], obs['actor_belief'], obs['behavior_patterns']")
    print("   - Actor beliefs maintain multi-behavior structure: {goal: [grid_bt0, grid_bt1, ...]}")


def simple_interaction_demo():
    """
    Original simple demo - kept for backward compatibility.
    """
    # Initialize environment
    env = AGREnv()
    obs, info = env.reset()
    
    print("Environment initialized")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Observation keys for observer: {obs[0].keys()}")
    
    # Run for a few steps with random observer actions and preset actor actions
    max_steps = 20
    for step in range(max_steps):
        # Random action for observer (agent 0)
        observer_action = random.choice([0, 1, 2, 3])  # Forward, left, right, stay
        
        # Preset action for actor (agent 1) - in a real scenario, this would come from elsewhere
        # For this demo we'll use a simple predetermined sequence
        actor_actions = [2, 2, 0, 2, 2, 1, 2, 2, 0, 2]  # Forward, forward, turn left, etc.
        actor_action = actor_actions[step % len(actor_actions)]
        
        # Combine actions (as dict: {agent_id: action})
        actions = {0: observer_action, 1: actor_action}
        
        # Step the environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print basic information
        print(f"\nStep {step+1}")
        print(f"Observer action: {observer_action}, Actor action: {actor_action}")
        print(f"Observer reward: {rewards[0]}")
        
        # Check if episode is done
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished")
            break
            
        # Update observation for next step
        obs = next_obs
    
    env.close()

if __name__ == "__main__":
    # Run the belief tracking demo (recommended)
    belief_tracking_demo()
    
    # Uncomment to run the simple demo instead
    # simple_interaction_demo()
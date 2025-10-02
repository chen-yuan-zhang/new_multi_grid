"""
Example usage of AGREnv with augmented belief observations using two agents.

This example demonstrates how to use the modified AGREnv environment 
with BeliefUpdateObserver to get belief distributions in observations.
"""

from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
import numpy as np


def two_agent_example():
    """Example of using AGREnv with two agents where observer tracks target with beliefs."""
    
    # Create environment
    env = AGREnv(size=10, num_goals=3, max_steps=100)
    
    # Reset environment to get initial observations
    obs, info = env.reset()
    
    # Create belief observer (this will be agent 0 - the observer)
    belief_observer = BeliefUpdateObserver(env)
    
    print("Initial observations:")
    print("Observer obs keys:", list(obs[0].keys()))
    print("Target obs keys:", list(obs[1].keys()))
    print("Goals:", env.goals)
    print("True goal:", env.goal)
    
    # Run simulation
    for step in range(10):
        print(f"\n=== Step {step+1} ===")
        
        # Observer (agent 0) uses belief-based action selection
        observer_action = belief_observer.compute_action(obs[0])
        
        # Target (agent 1) uses simple movement (you can replace with your target policy)
        target_actions = [2, 2, 1, 2, 2, 0, 2, 2]  # Forward, forward, turn right, etc.
        target_action = target_actions[step % len(target_actions)]
        
        # Combine actions for both agents
        actions = [observer_action, target_action]
        
        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Augment observation with belief data
        augmented_obs = belief_observer.augment_observation(obs)
        
        # Print belief information
        if "goal_belief" in augmented_obs[0]:
            print("Goal belief:", {str(k): f"{v:.3f}" for k, v in augmented_obs[0]["goal_belief"].items()})
            
        if "actor_belief" in augmented_obs[0]:
            print("Actor belief available for goals:", list(augmented_obs[0]["actor_belief"].keys()))
            # Show structure of raw multi-behavior beliefs
            first_goal = list(augmented_obs[0]["actor_belief"].keys())[0]
            print(f"Number of behavior types for goal {first_goal}:", len(augmented_obs[0]["actor_belief"][first_goal]))
            
        if "behavior_patterns" in augmented_obs[0]:
            print("Behavior patterns available:", list(augmented_obs[0]["behavior_patterns"].keys()))
        
        print(f"Observer reward: {rewards[0]}, Target reward: {rewards[1]}")
        
        # Check if episode is done
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished")
            break
            
        # Update observations for next step
        obs = augmented_obs


def simple_belief_access_demo():
    """Simple demo showing direct belief access without environment stepping."""
    
    # Create environment and belief observer
    env = AGREnv(size=8, num_goals=2, max_steps=50)
    obs, info = env.reset()
    belief_observer = BeliefUpdateObserver(env)
    
    print("Direct belief access demo:")
    print("Initial goal belief:", belief_observer.goal_belief)
    print("Actor belief keys:", list(belief_observer.actor_belief.keys()))
    
    # Show how to access individual behavior beliefs
    first_goal = list(belief_observer.actor_belief.keys())[0]
    print(f"Number of behavior types for goal {first_goal}:", len(belief_observer.actor_belief[first_goal]))
    print(f"Shape of behavior 0 belief for goal {first_goal}:", belief_observer.actor_belief[first_goal][0].shape)
    
    # Augment observation
    augmented_obs = belief_observer.augment_observation(obs)
    print("Augmented observation has keys:", list(augmented_obs[0].keys()))


if __name__ == "__main__":
    print("Running two-agent example...")
    two_agent_example()
    
    print("\n" + "="*50 + "\n")
    
    print("Running simple belief access demo...")
    simple_belief_access_demo()
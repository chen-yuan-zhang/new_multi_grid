import numpy as np
import random
from multigrid.envs.goal_prediction import AGREnv

def simple_interaction_demo():
    """
    Demonstrate basic interaction with the goal prediction environment.
    This only shows the simplest interaction pattern with preset actor actions.
    """
    # Initialize environment
    env = AGREnv(render_mode="human")
    obs, info = env.reset()
    
    print("Environment initialized")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Observation keys for observer: {obs[0].keys()}")
    
    # Run for a few steps with random observer actions and preset actor actions
    max_steps = 20
    for step in range(max_steps):
        # Random action for observer (agent 0)
        observer_action = env.action_space[0].sample()
        
        # Preset action for actor (agent 1) - in a real scenario, this would come from elsewhere
        # For this demo we'll use a simple predetermined sequence
        actor_actions = [2, 2, 0, 2, 2, 1, 2, 2, 0, 2]  # Forward, forward, turn left, etc.
        actor_action = actor_actions[step % len(actor_actions)]
        
        # Combine actions
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
    simple_interaction_demo()
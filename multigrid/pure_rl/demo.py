
import numpy as np
import random
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.pure_rl.obs_to_belief_image_array import preprocess_obs_for_rl_policy
import os 
from pathlib import Path
from PIL import Image
import cv2
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
import torch
from ray.rllib.utils.numpy import convert_to_numpy, softmax

def belief_tracking_demo(ppo_checkpoint_path):
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
    
    init_belief = belief_observer.goal_belief  # dict {goal: prob}
    
    # Load RL Policy model 
    
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            ppo_checkpoint_path,
            'learner_group',
            'learner',
            'rl_module',
            DEFAULT_MODULE_ID,
        )
    )
    
    
    
    # Run simulation
    # Target makes some action (in practice this would be unknown to observer)
    target_actions = [2, 2, 1, 2, 0, 2, 2, 1, 2, 2]  # Example sequence
    max_steps = 10
    cum_return = 0.0
    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        
        # Observer makes action based on beliefs (greedy planning)
        _ = belief_observer.compute_action(obs[0], render_and_save=False, get_action=False)
        
        target_action = target_actions[step % len(target_actions)]
        
        obs_processed = preprocess_obs_for_rl_policy(
            belief_update_observer=belief_observer,
            obs=obs[0],
        )
        input_dict = {
            Columns.OBS: torch.from_numpy(obs_processed).unsqueeze(0),
        }
        
        
        rl_module_out = rl_module.forward_inference(input_dict)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        observer_action = np.random.choice(env.action_space[0].n, p=softmax(logits[0]))
        
        # Combine actions (as dict: {agent_id: action})
        actions = {0: observer_action, 1: target_action}
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        # draw the observer obs using minigrid rendering
        # when rendering, enable the mask to show the agent's field of view only

        
        # IMPORTANT: Augment observations with belief distributions
        augmented_obs = belief_observer.augment_observation(next_obs)
        # Display current beliefs
        goal_belief = augmented_obs[0]['goal_belief']
        
        print(f"Observer action: {observer_action}, Target action: {target_action}")
        print(f"Goal beliefs: {dict((str(k), f'{v:.3f}') for k, v in goal_belief.items())}")
        
       
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished")
            break
            
        obs = next_obs
    
    print("\n4. Demonstrating different ways to access beliefs...")
    

    # Method Access from augmented observations (recommended for integration)
    final_augmented_obs = belief_observer.augment_observation(obs)
    print("\nAccess from augmented observations:")
    print(f"  obs['goal_belief'] = {final_augmented_obs[0]['goal_belief']}")

  
    
    print("\n✅ Demo completed!")
    print("   Key takeaways:")
    print("   - Use BeliefUpdateObserver for belief tracking")
    print("   - Call belief_observer.augment_observation(obs) to add beliefs to observations") 
    print("   - Access beliefs via obs['goal_belief'], obs['actor_belief'], obs['behavior_patterns']")
    print("   - Actor beliefs maintain multi-behavior structure: {goal: [grid_bt0, grid_bt1, ...]}")
    breakpoint()
    env.close()


if __name__ == "__main__":
    # Run the belief tracking demo (recommended)
    ppo_checkpoint_path = '/home/sukai/Project/chenyuan_project/new_multi_grid_rl/multigrid/pure_rl/ppo_observer_checkpoints'
    
    belief_tracking_demo(ppo_checkpoint_path)
    

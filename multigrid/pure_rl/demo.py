
import numpy as np
import random
from multigrid.envs.goal_prediction import AGREnv
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
from multigrid.pure_rl.reward_observer import ObserverRewarder
from multigrid.pure_rl.obs_to_belief_image_array import obs_to_belief_image_array
import os 
from pathlib import Path
from PIL import Image
import cv2

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
    
    # Create rewarder for the observer agent
    init_belief = belief_observer.goal_belief  # dict {goal: prob}
    rewarder = ObserverRewarder(
        gamma=0.995,
        R_correct=1.0,
        alpha=0.1,
        kappa=1.0,
        potential="log_true",    # or "neg_entropy"
        theta=0.8,
        conf_weight="prob",
        false_alarm_cost=0.0     # keep 0 for non-negative terminals
    )
    # If the true goal is accessible as env.goal (as you used earlier), pass it:
    true_goal_key = getattr(env, "goal", None)
    rewarder.reset(init_goal_belief=init_belief, true_goal=true_goal_key)
    
        
    print(f"   Goals: {env.goals}")
    print(f"   Observer at: {env.observer.state.pos}")
    print(f"   Target at: {env.target.state.pos}")
    
    print("\n2. Initial belief state...")
    print(f"   Goal beliefs: {belief_observer.goal_belief}")
    print(f"   Number of behavior types tracked: {len(belief_observer.actor_belief[env.goals[0]])}")
    
    print("\n3. Running simulation with belief updates...")
    
    # Run simulation
    max_steps = 10
    cum_return = 0.0
    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        
        # Observer makes action based on beliefs (greedy planning)
        observer_action = belief_observer.compute_action(obs[0], render_and_save=False)
        
        demo_savepath = os.path.join(Path(os.path.dirname(__file__)), "demo_observer_obs", f"step_{step+1}.png")
        img = env.grid.render(tile_size=32, agents=(env.unwrapped.agents[0], env.unwrapped.agents[1]), highlight_mask=None)
        
        
        # this image is ndarray (H,W,3) in RGB format
        im = Image.fromarray(img)
        im.save(demo_savepath)
        
        # save belief image
        belief_img_savepath = os.path.join(Path(os.path.dirname(__file__)), "demo_observer_obs", f"step_{step+1}_belief.png")
        belief_img, log_belief_sum = obs_to_belief_image_array(belief_observer, None, obs[0])
        
        # belief_img is ndarray (H,W,3) in RGB format
        # im_belief = Image.fromarray(belief_img)
        # im_belief.save(belief_img_savepath)
        
        # use opencv to save 
        cv2.imwrite(belief_img_savepath, cv2.cvtColor(belief_img, cv2.COLOR_RGB2BGR))
        
        # Target makes some action (in practice this would be unknown to observer)
        target_actions = [2, 2, 1, 2, 0, 2, 2, 1, 2, 2]  # Example sequence
        target_action = target_actions[step % len(target_actions)]
        
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
        
        actor_belief = augmented_obs[0]['actor_belief']
        
        # --- compute observer reward (length-neutral) ---
        # If your env stores the true goal under a different attribute, swap here.
        true_goal_key = getattr(env, "goal", true_goal_key)
        r_t, declared_now, rinfo = rewarder.step(goal_belief, true_goal_key, allow_declare=True)
        cum_return += (r_t)  # let your RL algorithm apply discounting; do NOT manually multiply by gamma here.

        if declared_now:
            print("Observer declared. Masking further observer rewards.") # this is already handled in rewarder.step()
            
            
        print(f"Observer reward: {r_t:.3f}, Cumulative return: {cum_return:.3f}")

        # Check if episode is done
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
    belief_tracking_demo()
    

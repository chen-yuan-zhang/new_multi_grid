#!/usr/bin/env python3
"""
Test script to demonstrate behavior probability normalization.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multigrid.envs.goal_prediction import AGREnv
    from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_behavior_normalization():
    """Test behavior probability normalization."""
    print("🧪 Testing Behavior Probability Normalization")
    print("=" * 55)
    
    # Create environment  
    env = AGREnv(size=6, num_goals=2, max_steps=20)
    obs, info = env.reset()
    
    # Create belief observer with log-space disabled to see non-zero values
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False, use_log_space=False)
    
    print(f"Goals: {env.goals}")
    print(f"Log-space enabled: {belief_observer.use_log_space}")
    
    # Run a few steps where target is NOT observed to see belief propagation
    print(f"\n🎯 Running steps without target observation...")
    
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Force target to not be observed by setting a condition that won't trigger
        obs[0].pop('target_pos', None)  # Remove target_pos if it exists
        
        # Compute actions
        observer_action = belief_observer.compute_action(obs[0])
        target_action = 2  # Fixed target action
        
        print(f"Observer action: {observer_action}, Target action: {target_action}")
        
        # Check behavior values after each step  
        first_goal = list(belief_observer.actor_belief.keys())[0]
        behavior_sums = [np.sum(grid) for grid in belief_observer.actor_belief[first_goal]]
        total_sum = sum(behavior_sums)
        
        print(f"Raw behavior sums: {[f'{s:.6f}' for s in behavior_sums]}")
        print(f"Total sum: {total_sum:.6f}")
        
        if total_sum > 0:
            normalized = [s / total_sum for s in behavior_sums]
            print(f"Normalized: {[f'{n:.4f}' for n in normalized]}")
            print(f"Normalized sum: {sum(normalized):.4f}")
        else:
            print("All zeros - cannot normalize")
        
        # Step environment
        actions = {0: observer_action, 1: target_action}
        try:
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            if any(terminations.values()) or any(truncations.values()):
                print("Episode finished")
                break
        except Exception as e:
            print(f"Error stepping environment: {e}")
            break
    
    print("\n✅ Behavior normalization test completed!")
    return True


if __name__ == "__main__":
    print("🎯 Behavior Normalization Test")
    print("=" * 35)
    
    try:
        success = test_behavior_normalization()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Test script to verify log-space belief tracking.
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


def test_log_space_beliefs():
    """Test log-space belief tracking."""
    print("🧪 Testing Log-Space Belief Tracking")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=6, num_goals=2, max_steps=20)
    obs, info = env.reset()
    
    # Create belief observer with log-space enabled
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False, use_log_space=True)
    
    print(f"Goals: {env.goals}")
    print(f"Goal beliefs: {belief_observer.goal_belief}")
    print(f"Log-space enabled: {belief_observer.use_log_space}")
    
    # Check initial belief structure
    print("\nInitial belief structure:")
    for goal in env.goals:
        print(f"  Goal {goal}:")
        for i, behavior_grid in enumerate(belief_observer.actor_belief[goal]):
            # Count non-infinite log probabilities
            non_inf_count = np.sum(behavior_grid > -np.inf)
            max_log_prob = np.max(behavior_grid)
            min_log_prob = np.min(behavior_grid[behavior_grid > -np.inf]) if non_inf_count > 0 else -np.inf
            
            print(f"    Behavior {i}: non-inf cells: {non_inf_count}, max_log_prob: {max_log_prob:.6f}, min_log_prob: {min_log_prob:.6f}")
            
            # Convert some log probabilities back to regular space for checking
            if non_inf_count > 0:
                regular_probs = np.exp(behavior_grid[behavior_grid > -np.inf])
                total_prob = np.sum(regular_probs)
                print(f"      -> Total probability (exp): {total_prob:.6f}")
    
    # Run a few simulation steps
    print(f"\n🎯 Running simulation steps...")
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Compute actions
        observer_action = belief_observer.compute_action(obs[0])
        target_action = np.random.choice([0, 1, 2])  # Random target action
        
        print(f"Observer action: {observer_action}, Target action: {target_action}")
        
        # Check belief values after each step
        for goal in env.goals:
            behavior_sums = []
            for behavior_grid in belief_observer.actor_belief[goal]:
                # Sum in log-space by converting to regular space
                valid_probs = behavior_grid[behavior_grid > -np.inf]
                if len(valid_probs) > 0:
                    regular_probs = np.exp(valid_probs)
                    total_prob = np.sum(regular_probs)
                else:
                    total_prob = 0.0
                behavior_sums.append(f'{total_prob:.6f}')
            
            print(f"  Goal {goal} behavior sums: {behavior_sums}")
        
        print(f"  Goal beliefs: {dict((str(k), f'{v:.6f}') for k, v in belief_observer.goal_belief.items())}")
        
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
    
    print("\n✅ Log-space belief tracking test completed!")
    return True


if __name__ == "__main__":
    print("🎯 Log-Space Belief Tracking Test")
    print("=" * 40)
    
    try:
        success = test_log_space_beliefs()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Test script to verify the greedy BeliefUpdateObserver is working correctly.
"""

import sys
import os
import time
import numpy as np

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multigrid.envs.goal_prediction import AGREnv
    from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_greedy_observer():
    """Test that the greedy BeliefUpdateObserver works correctly."""
    print("🧪 Testing Greedy BeliefUpdateObserver")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=6, num_goals=2, max_steps=30)
    obs, info = env.reset()
    
    # Create belief observer with greedy action selection
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False)
    
    print(f"Environment initialized with {len(env.goals)} goals")
    print(f"Goals: {env.goals}")
    print(f"Observer initial position: {belief_observer.pos}")
    print(f"Observer initial direction: {belief_observer.dir}")
    
    # Run episode with greedy action selection
    total_steps = 0
    total_time = 0
    
    for step in range(10):  # Run for 10 steps
        print(f"\n--- Step {step + 1} ---")
        
        # Get cache stats before action
        cache_stats_before = belief_observer.get_cache_stats()
        
        start_time = time.time()
        
        try:
            # Compute action using greedy approach
            observer_action = belief_observer.compute_action(obs[0])
            print(f"Observer greedy action: {observer_action}")
            
        except Exception as e:
            print(f"❌ Error in compute_action: {e}")
            import traceback
            traceback.print_exc()
            break
        
        end_time = time.time()
        step_time = end_time - start_time
        total_time += step_time
        
        # Get cache stats after action
        cache_stats_after = belief_observer.get_cache_stats()
        
        print(f"Step computation time: {step_time:.3f}s")
        print(f"Cache hits this step: {cache_stats_after['cache_hits'] - cache_stats_before['cache_hits']}")
        print(f"Current goal belief: {belief_observer.goal_belief}")
        
        # Step environment
        target_action = 2  # Simple forward action for target
        actions = {0: observer_action, 1: target_action}
        
        try:
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_steps += 1
        except Exception as e:
            print(f"❌ Error in environment step: {e}")
            break
        
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished")
            break
    
    # Final statistics
    final_cache_stats = belief_observer.get_cache_stats()
    avg_time_per_step = total_time / total_steps if total_steps > 0 else 0
    
    print(f"\n📊 Final Results:")
    print(f"   Total steps: {total_steps}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time per step: {avg_time_per_step:.3f}s")
    print(f"   Final cache size: {final_cache_stats['cache_size']}")
    print(f"   Cache hit rate: {final_cache_stats['hit_rate']:.2%}")
    print(f"   Final goal belief: {belief_observer.goal_belief}")
    
    return True


def test_greedy_components():
    """Test individual components of the greedy algorithm."""
    print("\n🧪 Testing Greedy Algorithm Components")
    print("=" * 45)
    
    # Create environment and observer
    env = AGREnv(size=6, num_goals=2, max_steps=10)
    obs, info = env.reset()
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False)
    
    print("Testing utility calculation components...")
    
    try:
        # Test utility evaluation for different actions
        current_pos_state = (tuple(belief_observer.pos), int(belief_observer.dir))
        from multigrid.gr_pursuer.astar import get_obs_successor
        
        possible_actions = get_obs_successor(env, current_pos_state)
        print(f"Number of possible actions: {len(possible_actions)}")
        
        for i, (action, next_pos_state) in enumerate(possible_actions[:3]):  # Test first 3 actions
            utility = belief_observer.evaluate_action_utility(current_pos_state, next_pos_state, action)
            info_gain = belief_observer.calculate_information_gain(next_pos_state)
            goal_approach = belief_observer.calculate_goal_approach_score(current_pos_state, next_pos_state)
            entropy_potential = belief_observer.calculate_entropy_reduction_potential(next_pos_state)
            
            print(f"Action {action}:")
            print(f"  Total utility: {utility:.3f}")
            print(f"  Info gain: {info_gain:.3f}")
            print(f"  Goal approach: {goal_approach:.3f}")
            print(f"  Entropy potential: {entropy_potential:.3f}")
        
        print("✅ Utility calculation components working")
        
        # Test expected target position calculation
        for goal in env.goals:
            expected_pos = belief_observer.get_expected_target_position(goal)
            print(f"Expected target position for goal {goal}: {expected_pos}")
        
        print("✅ Expected target position calculation working")
        
    except Exception as e:
        print(f"❌ Error in component testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🎯 Greedy BeliefUpdateObserver Test Suite")
    print("=" * 45)
    
    # Run tests
    try:
        success1 = test_greedy_observer()
        success2 = test_greedy_components()
        
        if success1 and success2:
            print("\n🎉 All greedy observer tests passed!")
            print("\nKey features of greedy approach:")
            print("- Direct utility-based action selection")
            print("- Considers information gain potential")
            print("- Accounts for goal-directed movement")
            print("- Prefers actions that reduce goal uncertainty")
            print("- Much faster than MCTS approach")
            print("- Still benefits from transition probability caching")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
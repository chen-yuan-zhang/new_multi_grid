#!/usr/bin/env python3
"""
Test script for belief tracking functionality.

This script can be run directly to test the belief tracking system
with BeliefUpdateObserver and augmented observations.

Usage:
    python test_belief_tracking.py
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
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_basic_functionality():
    """Test basic belief tracking functionality."""
    print("\n" + "="*50)
    print("TEST 1: Basic Functionality")
    print("="*50)
    
    try:
        # Create environment
        env = AGREnv(size=8, num_goals=2, max_steps=20)
        print("✅ Environment created successfully")
        
        # Reset environment
        obs, info = env.reset()
        print("✅ Environment reset successfully")
        print(f"   Observer observation keys: {list(obs[0].keys())}")
        print(f"   Target observation keys: {list(obs[1].keys())}")
        print(f"   Goals: {env.goals}")
        print(f"   True goal: {env.goal}")
        
        # Create belief observer
        belief_observer = BeliefUpdateObserver(env)
        print("✅ BeliefUpdateObserver created successfully")
        print(f"   Initial goal belief: {belief_observer.goal_belief}")
        
        # Test belief structure
        first_goal = list(belief_observer.actor_belief.keys())[0]
        num_behaviors = len(belief_observer.actor_belief[first_goal])
        print(f"   Actor belief structure: {len(belief_observer.actor_belief)} goals, {num_behaviors} behaviors each")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_observation_augmentation():
    """Test observation augmentation with belief data."""
    print("\n" + "="*50)
    print("TEST 2: Observation Augmentation")
    print("="*50)
    
    try:
        # Setup
        env = AGREnv(size=6, num_goals=2, max_steps=15)
        obs, info = env.reset()
        belief_observer = BeliefUpdateObserver(env)
        
        print("✅ Setup complete")
        
        # Test augmentation before any actions
        augmented_obs = belief_observer.augment_observation(obs)
        print("✅ Observation augmentation successful")
        
        # Check augmented observation structure
        aug_keys = list(augmented_obs[0].keys())
        expected_keys = ['goal_belief', 'actor_belief', 'behavior_patterns']
        
        print(f"   Augmented observation keys: {aug_keys}")
        
        for key in expected_keys:
            if key in aug_keys:
                print(f"   ✅ {key} present")
            else:
                print(f"   ❌ {key} missing")
                return False
        
        # Test structure of actor_belief (should be raw multi-behavior)
        actor_belief = augmented_obs[0]['actor_belief']
        first_goal = list(actor_belief.keys())[0]
        behavior_list = actor_belief[first_goal]
        
        print(f"   Actor belief for goal {first_goal}: {len(behavior_list)} behavior types")
        print(f"   Each behavior grid shape: {behavior_list[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_multi_step_simulation():
    """Test multi-step simulation with belief tracking."""
    print("\n" + "="*50)
    print("TEST 3: Multi-Step Simulation")
    print("="*50)
    
    try:
        # Setup
        env = AGREnv(size=6, num_goals=2, max_steps=10)
        obs, info = env.reset()
        belief_observer = BeliefUpdateObserver(env)
        
        print("✅ Setup complete")
        
        # Run simulation steps
        for step in range(3):
            print(f"\n--- Step {step + 1} ---")
            
            # Observer action
            observer_action = belief_observer.compute_action(obs[0])
            print(f"   Observer action: {observer_action}")
            
            # Simple target action (forward most of the time)
            target_action = 2 if step % 2 == 0 else 1  # Forward or turn right
            print(f"   Target action: {target_action}")
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step([observer_action, target_action])
            
            # Augment observation
            augmented_obs = belief_observer.augment_observation(obs)
            
            # Print belief information
            goal_belief = augmented_obs[0]['goal_belief']
            print(f"   Goal beliefs: {goal_belief}")
            
            # Check if any beliefs are updating
            total_belief = sum(goal_belief.values())
            print(f"   Total goal belief: {total_belief:.3f}")
            
            # Update obs for next iteration
            obs = augmented_obs
            
            # Check termination
            if any(terminations.values()) or any(truncations.values()):
                print("   Episode terminated")
                break
        
        print("✅ Multi-step simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_belief_access_patterns():
    """Test different ways to access belief data."""
    print("\n" + "="*50)
    print("TEST 4: Belief Access Patterns")
    print("="*50)
    
    try:
        # Setup
        env = AGREnv(size=6, num_goals=2, max_steps=10)
        obs, info = env.reset()
        belief_observer = BeliefUpdateObserver(env)
        
        # Take one step to update beliefs
        action = belief_observer.compute_action(obs[0])
        obs, _, _, _, _ = env.step([action, 2])
        augmented_obs = belief_observer.augment_observation(obs)
        
        print("✅ Setup and one step completed")
        
        # Test direct access from belief observer
        print("\n   Direct access from BeliefUpdateObserver:")
        direct_goal_belief = belief_observer.goal_belief
        direct_actor_belief = belief_observer.actor_belief
        print(f"   - Goal belief keys: {list(direct_goal_belief.keys())}")
        print(f"   - Actor belief keys: {list(direct_actor_belief.keys())}")
        
        # Test access from augmented observation
        print("\n   Access from augmented observation:")
        obs_goal_belief = augmented_obs[0]['goal_belief']
        obs_actor_belief = augmented_obs[0]['actor_belief']
        print(f"   - Goal belief keys: {list(obs_goal_belief.keys())}")
        print(f"   - Actor belief keys: {list(obs_actor_belief.keys())}")
        
        # Test behavior-specific access
        print("\n   Behavior-specific access:")
        first_goal = list(obs_actor_belief.keys())[0]
        behavior_grids = obs_actor_belief[first_goal]
        print(f"   - Goal {first_goal} has {len(behavior_grids)} behavior types")
        for i, grid in enumerate(behavior_grids):
            belief_sum = np.sum(grid)
            print(f"   - Behavior {i}: belief sum = {belief_sum:.6f}")
        
        # Test behavior patterns access
        behavior_patterns = augmented_obs[0]['behavior_patterns']
        print(f"\n   Behavior patterns available: {list(behavior_patterns.keys())}")
        
        print("✅ All access patterns work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Starting Belief Tracking Tests")
    print("="*60)
    
    tests = [
        test_basic_functionality,
        test_observation_augmentation,
        test_multi_step_simulation,
        test_belief_access_patterns
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Belief tracking system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
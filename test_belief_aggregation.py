#!/usr/bin/env python3
"""
Test script to verify belief aggregation across goals, behaviors, and directions.
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


def test_belief_aggregation():
    """Test that belief aggregation correctly sums across all dimensions."""
    print("🧪 Testing Belief Aggregation")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=6, num_goals=2, max_steps=20)
    obs, info = env.reset()
    
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False)
    
    print(f"Goals: {env.goals}")
    print(f"Goal beliefs: {belief_observer.goal_belief}")
    
    # Manually inspect the belief structure
    print("\nBelief structure:")
    for goal in env.goals:
        print(f"  Goal {goal}:")
        print(f"    Number of behavior types: {len(belief_observer.actor_belief[goal])}")
        for i, behavior_grid in enumerate(belief_observer.actor_belief[goal]):
            total_belief_in_grid = np.sum(behavior_grid)
            print(f"    Behavior {i}: shape {behavior_grid.shape}, total belief: {total_belief_in_grid:.6f}")
            
            # Show some non-zero positions
            nonzero = np.argwhere(behavior_grid > 0)
            if len(nonzero) > 0:
                print(f"      Non-zero positions (first 5): {nonzero[:5]}")
    
    # Test the aggregation method
    most_likely_pos = belief_observer.get_most_likely_actor_position()
    print(f"\nMost likely actor position: {most_likely_pos}")
    
    # Manual verification: compute aggregation step by step
    print("\nManual verification of aggregation:")
    sample_goal = next(iter(belief_observer.actor_belief))
    sample_grid = belief_observer.actor_belief[sample_goal][0]
    height, width, num_directions = sample_grid.shape
    print(f"Grid dimensions: {height} x {width} x {num_directions}")
    
    # Manual aggregation
    manual_aggregated_belief = np.zeros((height, width))
    
    for goal in env.goals:
        goal_weight = belief_observer.goal_belief[goal]
        print(f"\nGoal {goal} (weight: {goal_weight:.3f}):")
        
        for behavior_idx, behavior_grid in enumerate(belief_observer.actor_belief[goal]):
            # Sum across directions at each position
            position_belief = np.sum(behavior_grid, axis=2)
            weighted_position_belief = goal_weight * position_belief
            manual_aggregated_belief += weighted_position_belief
            
            total_belief = np.sum(position_belief)
            print(f"  Behavior {behavior_idx}: total position belief = {total_belief:.6f}")
    
    # Find maximum in manual aggregation
    manual_max_pos = np.unravel_index(np.argmax(manual_aggregated_belief), manual_aggregated_belief.shape)
    manual_max_value = manual_aggregated_belief[manual_max_pos]
    
    print(f"\nManual aggregation results:")
    print(f"  Maximum position: {manual_max_pos}")
    print(f"  Maximum value: {manual_max_value:.6f}")
    print(f"  Method result: {most_likely_pos}")
    
    # Verify they match
    if most_likely_pos == manual_max_pos:
        print("✅ Belief aggregation is working correctly!")
    else:
        print("❌ Belief aggregation mismatch!")
        
    # Show top 3 positions
    print(f"\nTop 3 most likely positions:")
    flat_indices = np.argsort(manual_aggregated_belief.flatten())[::-1]
    for i in range(min(3, len(flat_indices))):
        pos = np.unravel_index(flat_indices[i], manual_aggregated_belief.shape)
        value = manual_aggregated_belief[pos]
        print(f"  {i+1}. Position {pos}: belief = {value:.6f}")

    return True


if __name__ == "__main__":
    print("🎯 Belief Aggregation Test")
    print("=" * 30)
    
    try:
        success = test_belief_aggregation()
        if success:
            print("\n🎉 Belief aggregation test completed!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
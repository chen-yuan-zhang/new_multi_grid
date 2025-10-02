#!/usr/bin/env python3
"""
Test script to check the shape of most_likely_actor_pos.
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


def test_most_likely_actor_pos_shape():
    """Test the shape and type of most_likely_actor_pos."""
    print("🧪 Testing most_likely_actor_pos Shape")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=6, num_goals=2, max_steps=20)
    obs, info = env.reset()
    
    belief_observer = BeliefUpdateObserver(env, use_neural_predictor=False)
    
    # Get the most likely actor position
    most_likely_actor_pos = belief_observer.get_most_likely_actor_position()
    
    print(f"most_likely_actor_pos: {most_likely_actor_pos}")
    print(f"Type: {type(most_likely_actor_pos)}")
    
    if most_likely_actor_pos is not None:
        print(f"Shape: {np.array(most_likely_actor_pos).shape}")
        print(f"Length: {len(most_likely_actor_pos)}")
        print(f"Element types: {[type(x) for x in most_likely_actor_pos]}")
        print(f"Element values: {[x for x in most_likely_actor_pos]}")
        
        # Check what np.unravel_index returns
        sample_goal = next(iter(belief_observer.actor_belief))
        sample_grid = belief_observer.actor_belief[sample_goal][0]
        height, width, num_directions = sample_grid.shape
        aggregated_belief = np.zeros((height, width))
        
        print(f"\nContext:")
        print(f"Grid dimensions: height={height}, width={width}, directions={num_directions}")
        print(f"aggregated_belief shape: {aggregated_belief.shape}")
        
        # Simulate np.unravel_index behavior
        test_index = np.argmax(aggregated_belief)
        test_pos = np.unravel_index(test_index, aggregated_belief.shape)
        print(f"Example np.unravel_index result: {test_pos}")
        print(f"Example type: {type(test_pos)}")
        print(f"Example shape: {np.array(test_pos).shape}")
        
    else:
        print("most_likely_actor_pos is None")

    return True


if __name__ == "__main__":
    print("🎯 most_likely_actor_pos Shape Test")
    print("=" * 40)
    
    try:
        success = test_most_likely_actor_pos_shape()
        if success:
            print("\n🎉 Shape test completed!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
"""
Test Environment Action Handling

This test verifies that the multi-agent environment correctly processes
observer and actor actions, ensuring that actions produce expected state changes.
"""

import numpy as np
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import Direction

def test_basic_movement():
    """Test basic forward movement for both observer and actor."""
    print("🧪 Testing Basic Movement")
    print("=" * 40)
    
    # Create a simple environment
    env = AGREnv(
        size=8,
        agents_start_pos=[(1, 1), (3, 3)],  # observer, target
        agents_start_dir=[Direction.right, Direction.down]  # facing right, down
    )
    
    obs, info = env.reset()
    print(f"Initial observer pos: {env.observer.pos}, dir: {env.observer.dir}")
    print(f"Initial target pos: {env.target.pos}, dir: {env.target.dir}")
    
    # Test observer forward movement (right)
    initial_obs_pos = tuple(env.observer.pos)
    initial_obs_dir = env.observer.dir
    
    actions = {
        0: Action.forward,  # observer moves forward (east)
        1: Action.stay      # target stays
    }
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    expected_obs_pos = (initial_obs_pos[0] + 1, initial_obs_pos[1])  # Move right
    actual_obs_pos = tuple(env.observer.pos)
    
    print(f"\n📍 Observer Movement Test:")
    print(f"  Expected position after forward (right): {expected_obs_pos}")
    print(f"  Actual position: {actual_obs_pos}")
    print(f"  Direction unchanged: {env.observer.dir} == {initial_obs_dir}")
    
    assert actual_obs_pos == expected_obs_pos, f"Observer forward failed: expected {expected_obs_pos}, got {actual_obs_pos}"
    assert env.observer.dir == initial_obs_dir, f"Observer direction changed unexpectedly"
    print("  ✅ Observer forward movement correct")
    
    env.close()

def test_rotation():
    """Test rotation actions for both agents."""
    print("\n🔄 Testing Rotation Actions")
    print("=" * 40)
    
    env = AGREnv(
        size=8,
        agents_start_pos=[(2, 2), (4, 4)],
        agents_start_dir=[Direction.right, Direction.up]
    )
    
    obs, info = env.reset()
    initial_obs_dir = env.observer.dir
    initial_target_dir = env.target.dir
    
    print(f"Initial directions - Observer: {initial_obs_dir}, Target: {initial_target_dir}")
    
    # Test left rotation
    actions = {
        0: Action.left,   # observer turns left (right -> up)
        1: Action.right   # target turns right (up -> right)
    }
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    expected_obs_dir = (initial_obs_dir - 1) % 4  # Turn left
    expected_target_dir = (initial_target_dir + 1) % 4  # Turn right
    
    print(f"\n🔄 Rotation Test:")
    print(f"  Observer: {initial_obs_dir} -> {env.observer.dir} (expected: {expected_obs_dir})")
    print(f"  Target: {initial_target_dir} -> {env.target.dir} (expected: {expected_target_dir})")
    
    assert env.observer.dir == expected_obs_dir, f"Observer rotation failed"
    assert env.target.dir == expected_target_dir, f"Target rotation failed"
    print("  ✅ Rotation actions correct")
    
    env.close()

def test_multi_agent_independent_actions():
    """Test that observer and target can move independently."""
    print("\n👥 Testing Independent Multi-Agent Actions")
    print("=" * 50)
    
    env = AGREnv(
        size=10,
        agents_start_pos=[(1, 1), (5, 5)],
        agents_start_dir=[Direction.right, Direction.left]
    )
    
    obs, info = env.reset()
    
    initial_obs_pos = tuple(env.observer.pos)
    initial_target_pos = tuple(env.target.pos)
    
    print(f"Initial positions - Observer: {initial_obs_pos}, Target: {initial_target_pos}")
    
    # Both agents move forward simultaneously
    actions = {
        0: Action.forward,  # observer moves right
        1: Action.forward   # target moves left
    }
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    expected_obs_pos = (initial_obs_pos[0] + 1, initial_obs_pos[1])  # Right
    expected_target_pos = (initial_target_pos[0] - 1, initial_target_pos[1])  # Left
    
    actual_obs_pos = tuple(env.observer.pos)
    actual_target_pos = tuple(env.target.pos)
    
    print(f"\n👥 Independent Movement Test:")
    print(f"  Observer: {initial_obs_pos} -> {actual_obs_pos} (expected: {expected_obs_pos})")
    print(f"  Target: {initial_target_pos} -> {actual_target_pos} (expected: {expected_target_pos})")
    
    assert actual_obs_pos == expected_obs_pos, f"Observer independent movement failed"
    assert actual_target_pos == expected_target_pos, f"Target independent movement failed"
    print("  ✅ Independent multi-agent movement correct")
    
    env.close()

def test_wall_collision():
    """Test that agents can't move through walls."""
    print("\n🚧 Testing Wall Collision Detection")
    print("=" * 42)
    
    # Create environment with known wall positions
    base_grid = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],  # Wall in middle
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    
    env = AGREnv(
        base_grid=base_grid,
        agents_start_pos=[(1, 1), (1, 2)],  # Observer next to wall
        agents_start_dir=[Direction.down, Direction.right]  # Observer facing wall
    )
    
    obs, info = env.reset()
    
    initial_obs_pos = tuple(env.observer.pos)
    print(f"Observer position: {initial_obs_pos}, facing down toward wall at (1, 2)")
    print(f"Wall grid:\n{base_grid}")
    
    # Try to move observer into wall
    actions = {
        0: Action.forward,  # observer tries to move into wall (down)
        1: Action.stay
    }
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    final_obs_pos = tuple(env.observer.pos)
    
    print(f"\n🚧 Wall Collision Test:")
    print(f"  Initial position: {initial_obs_pos}")
    print(f"  After forward into wall: {final_obs_pos}")
    
    # Position should remain the same (can't move through walls)
    assert final_obs_pos == initial_obs_pos, f"Observer moved through wall! {initial_obs_pos} -> {final_obs_pos}"
    print("  ✅ Wall collision correctly prevented movement")
    
    env.close()

def test_stay_action():
    """Test that stay action keeps agents in place."""
    print("\n⏸️  Testing Stay Action")
    print("=" * 30)
    
    env = AGREnv(
        size=6,
        agents_start_pos=[(2, 2), (4, 3)],
        agents_start_dir=[Direction.up, Direction.down]
    )
    
    obs, info = env.reset()
    
    initial_obs_pos = tuple(env.observer.pos)
    initial_target_pos = tuple(env.target.pos)
    initial_obs_dir = env.observer.dir
    initial_target_dir = env.target.dir
    
    print(f"Initial state - Observer: {initial_obs_pos}, dir: {initial_obs_dir}")
    print(f"                Target: {initial_target_pos}, dir: {initial_target_dir}")
    
    # Both agents stay
    actions = {
        0: Action.stay,
        1: Action.stay
    }
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    final_obs_pos = tuple(env.observer.pos)
    final_target_pos = tuple(env.target.pos)
    
    print(f"\n⏸️  Stay Action Test:")
    print(f"  Observer position unchanged: {initial_obs_pos} == {final_obs_pos}")
    print(f"  Target position unchanged: {initial_target_pos} == {final_target_pos}")
    print(f"  Directions unchanged: {initial_obs_dir} == {env.observer.dir}, {initial_target_dir} == {env.target.dir}")
    
    assert final_obs_pos == initial_obs_pos, "Observer moved when staying"
    assert final_target_pos == initial_target_pos, "Target moved when staying"
    assert env.observer.dir == initial_obs_dir, "Observer direction changed when staying"
    assert env.target.dir == initial_target_dir, "Target direction changed when staying"
    print("  ✅ Stay action correctly maintains positions and directions")
    
    env.close()

def test_direction_mapping():
    """Test that all directions work correctly."""
    print("\n🧭 Testing All Direction Movements")
    print("=" * 42)
    
    directions = [
        (Direction.right, (1, 0), "Right"),
        (Direction.down, (0, 1), "Down"),
        (Direction.left, (-1, 0), "Left"),
        (Direction.up, (0, -1), "Up")
    ]
    
    for direction, expected_delta, name in directions:
        env = AGREnv(
            size=8,
            agents_start_pos=[(3, 3), (5, 5)],  # Center positions
            agents_start_dir=[direction, Direction.right]
        )
        
        obs, info = env.reset()
        initial_pos = tuple(env.observer.pos)
        
        actions = {0: Action.forward, 1: Action.stay}
        obs, reward, terminated, truncated, info = env.step(actions)
        
        final_pos = tuple(env.observer.pos)
        expected_pos = (initial_pos[0] + expected_delta[0], initial_pos[1] + expected_delta[1])
        
        print(f"  {name:5}: {initial_pos} -> {final_pos} (expected: {expected_pos})")
        
        assert final_pos == expected_pos, f"{name} movement failed: expected {expected_pos}, got {final_pos}"
        
        env.close()
    
    print("  ✅ All directional movements correct")

def run_all_tests():
    """Run all environment action tests."""
    print("🔬 Environment Action Handling Tests")
    print("=" * 60)
    
    try:
        test_basic_movement()
        test_rotation()
        test_multi_agent_independent_actions()
        test_wall_collision()
        test_stay_action()
        test_direction_mapping()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Environment correctly handles actions.")
        print("✅ Forward movement works correctly")
        print("✅ Rotation actions work correctly") 
        print("✅ Multi-agent independence verified")
        print("✅ Wall collision detection works")
        print("✅ Stay action maintains state")
        print("✅ All directions mapped correctly")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
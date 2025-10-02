"""
Test to verify the improved greedy algorithm avoids infinite turning loops
"""
import numpy as np
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import Direction
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver

def test_greedy_observer_movement():
    """Test that the greedy observer doesn't get stuck in turning loops"""
    
    print("🔄 Testing Greedy Observer Movement")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=10)
    obs, info = env.reset()
    
    # Create observer agent
    observer = BeliefUpdateObserver(env)
    
    # Track observer movement over multiple steps
    positions = []
    directions = []
    actions_taken = []
    
    print(f"Initial observer: pos={tuple(env.observer.pos)}, dir={Direction(env.observer.dir).name}")
    print(f"Target position: {tuple(env.target.pos)}")
    print()
    
    # Simulate 20 steps to see if observer gets stuck
    for step in range(20):
        current_pos = tuple(env.observer.pos)
        current_dir = env.observer.dir
        
        positions.append(current_pos)
        directions.append(current_dir)
        
        # Get observer action using improved greedy
        observer_action = observer.greedy()
        actions_taken.append(observer_action)
        
        # Create action dict (observer acts, target stays)
        actions = {0: observer_action, 1: Action.stay}
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Update observer's internal state
        observer.pos = env.observer.pos
        observer.dir = env.observer.dir
        
        print(f"Step {step:2d}: pos={current_pos} → {tuple(env.observer.pos)}, "
              f"dir={Direction(current_dir).name} → {Direction(env.observer.dir).name}, "
              f"action={observer_action.name}")
        
        # Check for infinite turning (same position, different directions)
        if step >= 4:  # Check last 4 steps
            recent_positions = positions[-4:]
            recent_actions = actions_taken[-4:]
            
            # If stuck in same position with only turn actions
            if (len(set(recent_positions)) == 1 and 
                all(action in [Action.left, Action.right] for action in recent_actions)):
                print("❌ DETECTED: Infinite turning loop!")
                break
    
    print()
    
    # Analyze movement pattern
    unique_positions = len(set(positions))
    total_steps = len(positions)
    turn_actions = sum(1 for action in actions_taken if action in [Action.left, Action.right])
    forward_actions = sum(1 for action in actions_taken if action == Action.forward)
    
    print("📊 Movement Analysis:")
    print(f"  Total steps: {total_steps}")
    print(f"  Unique positions visited: {unique_positions}")
    print(f"  Forward actions: {forward_actions}")
    print(f"  Turn actions: {turn_actions}")
    print(f"  Movement efficiency: {unique_positions/total_steps*100:.1f}%")
    
    # Check for good behavior
    if unique_positions > total_steps * 0.3:  # Should visit at least 30% unique positions
        print("✅ Good movement: Observer explores different positions")
    else:
        print("⚠️ Poor movement: Observer is too static")
    
    if turn_actions <= forward_actions * 2:  # Reasonable turn-to-forward ratio
        print("✅ Good action balance: Not excessive turning")
    else:
        print("⚠️ Excessive turning: Too many turn actions relative to forward movement")
    
    env.close()

def test_directional_approach():
    """Test if observer efficiently approaches target from different directions"""
    
    print(f"\n🎯 Testing Directional Approach Efficiency")
    print("=" * 50)
    
    # Test multiple scenarios
    for test_case in range(3):
        print(f"\nTest case {test_case + 1}:")
        
        env = AGREnv(size=10)
        obs, info = env.reset()
        observer = BeliefUpdateObserver(env)
        
        initial_pos = tuple(env.observer.pos)
        target_pos = tuple(env.target.pos)
        initial_distance = abs(initial_pos[0] - target_pos[0]) + abs(initial_pos[1] - target_pos[1])
        
        print(f"  Observer: {initial_pos}, Target: {target_pos}, Distance: {initial_distance}")
        
        # Simulate 10 steps
        for step in range(10):
            observer_action = observer.greedy()
            actions = {0: observer_action, 1: Action.stay}
            obs, reward, terminated, truncated, info = env.step(actions)
            
            observer.pos = env.observer.pos
            observer.dir = env.observer.dir
            
            new_pos = tuple(env.observer.pos)
            new_distance = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
            
            print(f"    Step {step}: {observer_action.name:8} → pos={new_pos}, dist={new_distance}")
            
            # If we're getting closer consistently, that's good
            if new_distance < initial_distance * 0.7:  # Made significant progress
                print(f"  ✅ Made good progress (reduced distance by {initial_distance - new_distance})")
                break
        
        env.close()

if __name__ == "__main__":
    test_greedy_observer_movement()
    test_directional_approach()
    
    print(f"\n🎯 Summary:")
    print("The improved greedy algorithm should:")
    print("✅ Avoid infinite turning loops")
    print("✅ Balance forward movement with necessary turning")
    print("✅ Make progress toward target position")
    print("✅ Use action costs to prefer forward movement")
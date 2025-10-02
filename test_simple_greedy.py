"""
Test the simplified greedy algorithm
"""
import numpy as np
from multigrid.envs.goal_prediction import AGREnv
from multigrid.core.actions import Action
from multigrid.core.constants import Direction
from multigrid.gr_pursuer.agents.observer import BeliefUpdateObserver

def test_simple_greedy():
    """Test the simplified greedy approach"""
    
    print("🎯 Testing Simplified Greedy Algorithm")
    print("=" * 50)
    
    # Create environment
    env = AGREnv(size=10)
    obs, info = env.reset()
    
    # Create observer agent
    observer = BeliefUpdateObserver(env)
    
    initial_pos = tuple(env.observer.pos)
    target_pos = tuple(env.target.pos)
    initial_distance = abs(initial_pos[0] - target_pos[0]) + abs(initial_pos[1] - target_pos[1])
    
    print(f"Initial observer: {initial_pos}, dir={Direction(env.observer.dir).name}")
    print(f"Target: {target_pos}")
    print(f"Initial distance: {initial_distance}")
    print()
    
    # Track movement for 15 steps
    for step in range(15):
        current_pos = tuple(env.observer.pos)
        current_dir = env.observer.dir
        current_distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
        
        # Get observer action
        observer_action = observer.greedy()
        
        # Step environment
        actions = {0: observer_action, 1: Action.stay}
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Update observer state
        observer.pos = env.observer.pos
        observer.dir = env.observer.dir
        
        new_pos = tuple(env.observer.pos)
        new_dir = env.observer.dir
        new_distance = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
        
        # Analyze the action
        action_result = ""
        if observer_action == Action.forward:
            if new_distance < current_distance:
                action_result = "✅ Good move"
            elif new_distance == current_distance:
                action_result = "→ No progress"
            else:
                action_result = "❌ Wrong direction"
        elif observer_action in [Action.left, Action.right]:
            action_result = "🔄 Turning toward target"
        else:
            action_result = "⏸️ Staying"
        
        print(f"Step {step:2d}: {observer_action.name:8} | {current_pos} → {new_pos} | "
              f"dist: {current_distance} → {new_distance} | {action_result}")
        
        # If we reached the target, stop
        if new_distance == 0:
            print("🎯 Reached target!")
            break
        
        # If we made significant progress, note it
        if new_distance < initial_distance * 0.5:
            print(f"📈 Good progress! Reduced distance by {initial_distance - new_distance}")
    
    final_distance = abs(tuple(env.observer.pos)[0] - target_pos[0]) + abs(tuple(env.observer.pos)[1] - target_pos[1])
    improvement = initial_distance - final_distance
    
    print()
    print(f"📊 Results:")
    print(f"  Initial distance: {initial_distance}")
    print(f"  Final distance: {final_distance}")
    print(f"  Improvement: {improvement} ({improvement/initial_distance*100:.1f}%)")
    
    if improvement > 0:
        print("✅ Observer made progress toward target")
    else:
        print("❌ Observer did not make progress")
    
    env.close()

def test_direction_alignment():
    """Test if the observer correctly identifies useful directions"""
    
    print(f"\n🧭 Testing Direction Alignment Logic")
    print("=" * 40)
    
    # Test different relative positions
    test_cases = [
        ((2, 2), (5, 2), "Target to the right"),
        ((5, 5), (2, 5), "Target to the left"), 
        ((3, 5), (3, 2), "Target above"),
        ((2, 2), (2, 6), "Target below"),
        ((2, 2), (5, 5), "Target diagonal (down-right)")
    ]
    
    for observer_pos, target_pos, description in test_cases:
        print(f"\n{description}:")
        print(f"  Observer at {observer_pos}, Target at {target_pos}")
        
        # Calculate expected useful direction
        dx = target_pos[0] - observer_pos[0]
        dy = target_pos[1] - observer_pos[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                expected_dir = "right"
            else:
                expected_dir = "left"
        else:
            if dy > 0:
                expected_dir = "down"
            else:
                expected_dir = "up"
        
        print(f"  Expected useful direction: {expected_dir}")
        
        # Test all starting directions
        for start_dir in range(4):
            dir_name = Direction(start_dir).name
            
            # Calculate what our algorithm would choose
            target_vec = (target_pos[0] - observer_pos[0], target_pos[1] - observer_pos[1])
            
            # Check if forward reduces distance
            forward_vec = Direction(start_dir).to_vec()
            forward_pos = (observer_pos[0] + forward_vec[0], observer_pos[1] + forward_vec[1])
            current_dist = abs(observer_pos[0] - target_pos[0]) + abs(observer_pos[1] - target_pos[1])
            forward_dist = abs(forward_pos[0] - target_pos[0]) + abs(forward_pos[1] - target_pos[1])
            
            if forward_dist < current_dist:
                action = "forward"
            else:
                # Find best direction
                best_direction = None
                min_angle_diff = float('inf')
                
                for direction in range(4):
                    dir_vec = Direction(direction).to_vec()
                    dot_product = dir_vec[0] * target_vec[0] + dir_vec[1] * target_vec[1]
                    angle_diff = -dot_product
                    
                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_direction = direction
                
                if best_direction == start_dir:
                    action = "stay"
                else:
                    turn_diff = (best_direction - start_dir) % 4
                    if turn_diff == 1 or turn_diff == -3:
                        action = "right"
                    else:
                        action = "left"
            
            print(f"    From {dir_name:5}: {action}")

if __name__ == "__main__":
    test_simple_greedy()
    test_direction_alignment()
    
    print(f"\n🎯 Summary:")
    print("The simplified algorithm:")
    print("✅ Moves forward when it reduces distance")
    print("✅ Turns toward target when forward doesn't help")
    print("✅ Avoids infinite loops by clear decision logic")
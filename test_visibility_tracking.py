"""
Test the visibility tracking functionality
"""
import pandas as pd
from multigrid.experience.main_simple import evaluate_scenario

def test_visibility_tracking():
    """Test that visibility data is correctly captured and stored"""
    
    print("👁️  Testing Visibility Tracking")
    print("=" * 40)
    
    # Load a few test scenarios
    df = pd.read_csv('goal_recognition_dataset.csv')
    test_scenarios = df.head(3)
    
    print(f"Testing {len(test_scenarios)} scenarios...\n")
    
    for idx, row in test_scenarios.iterrows():
        print(f"Scenario {idx + 1}:")
        
        # Run evaluation with visibility tracking
        success, convergence_step, exec_time, analysis_data = evaluate_scenario(row, verbose=True)
        
        # Print visibility analysis
        print(f"  📊 Visibility Analysis:")
        print(f"     Total steps: {analysis_data['total_steps']}")
        print(f"     Visible steps: {analysis_data['visible_steps']}")
        print(f"     Visibility ratio: {analysis_data['visibility_ratio']:.1%}")
        print(f"     Visibility changes: {analysis_data['visibility_changes']}")
        print(f"     Visibility pattern: {analysis_data['visibility_history']}")
        
        # Print confidence evolution
        if analysis_data['confidence_history']:
            print(f"     Initial confidence: {analysis_data['confidence_history'][0]:.3f}")
            print(f"     Final confidence: {analysis_data['confidence_history'][-1]:.3f}")
            print(f"     Max confidence: {max(analysis_data['confidence_history']):.3f}")
        
        print(f"  🎯 Result: {'✅ Success' if success else '❌ Failed'}")
        if success:
            print(f"     Converged at step: {convergence_step}")
        print()

if __name__ == "__main__":
    test_visibility_tracking()
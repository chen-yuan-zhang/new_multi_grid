"""
Analysis: How Initial Distance Affects In-View Rate

This script analyzes the relationship between initial_distance and visibility_ratio
from evaluation results to understand how starting distance impacts observation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_evaluation_results(pattern="evaluation_results*.csv"):
    """Load all evaluation result files matching the pattern."""
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No evaluation files found matching: {pattern}")
        return None
    
    # Try to find a complete results file (not temp files)
    complete_files = [f for f in files if 'temp' not in f]
    
    if complete_files:
        # Use the most recent complete file
        file = max(complete_files, key=lambda x: Path(x).stat().st_mtime)
    else:
        # Use the most recent temp file
        file = max(files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"📖 Loading: {file}")
    df = pd.read_csv(file)
    
    # Check for required columns
    required_cols = ['initial_distance', 'visibility_ratio']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Missing columns. Available: {df.columns.tolist()}")
        return None
    
    print(f"✅ Loaded {len(df)} scenarios")
    return df

def analyze_distance_visibility(df):
    """Analyze relationship between initial distance and visibility."""
    
    print("\n" + "="*70)
    print("📊 ANALYSIS: Initial Distance vs In-View Rate")
    print("="*70)
    
    # Filter valid data
    valid_df = df.dropna(subset=['initial_distance', 'visibility_ratio'])
    print(f"\n📈 Valid scenarios: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("❌ No valid data to analyze")
        return
    
    # Group by initial distance
    distance_stats = valid_df.groupby('initial_distance').agg({
        'visibility_ratio': ['mean', 'std', 'min', 'max', 'count'],
        'visible_steps': 'mean',
        'total_trajectory_steps': 'mean',
        'eval_success': 'mean'
    }).round(4)
    
    print("\n" + "-"*70)
    print("📋 Statistics by Initial Distance:")
    print("-"*70)
    print(distance_stats)
    
    # Correlation analysis
    if len(valid_df['initial_distance'].unique()) > 1:
        correlation = valid_df['initial_distance'].corr(valid_df['visibility_ratio'])
        print(f"\n🔗 Correlation (initial_distance vs visibility_ratio): {correlation:.4f}")
        
        if abs(correlation) > 0.7:
            strength = "Strong"
        elif abs(correlation) > 0.4:
            strength = "Moderate"
        elif abs(correlation) > 0.2:
            strength = "Weak"
        else:
            strength = "Very weak"
        
        direction = "negative" if correlation < 0 else "positive"
        print(f"   → {strength} {direction} correlation")
    
    # Analysis by distance ranges
    print("\n" + "-"*70)
    print("📊 Detailed Analysis by Distance:")
    print("-"*70)
    
    for distance in sorted(valid_df['initial_distance'].unique()):
        subset = valid_df[valid_df['initial_distance'] == distance]
        
        avg_visibility = subset['visibility_ratio'].mean()
        avg_visible_steps = subset['visible_steps'].mean()
        avg_total_steps = subset['total_trajectory_steps'].mean()
        success_rate = subset['eval_success'].mean() if 'eval_success' in subset.columns else None
        
        print(f"\nDistance {int(distance)}:")
        print(f"  • Scenarios: {len(subset)}")
        print(f"  • Avg visibility ratio: {avg_visibility:.1%}")
        print(f"  • Avg visible steps: {avg_visible_steps:.1f} / {avg_total_steps:.1f} total")
        if success_rate is not None:
            print(f"  • Success rate: {success_rate:.1%}")
    
    # Additional insights
    print("\n" + "-"*70)
    print("💡 Key Insights:")
    print("-"*70)
    
    # Compare closest vs farthest
    distances = sorted(valid_df['initial_distance'].unique())
    if len(distances) >= 2:
        closest = distances[0]
        farthest = distances[-1]
        
        closest_vis = valid_df[valid_df['initial_distance'] == closest]['visibility_ratio'].mean()
        farthest_vis = valid_df[valid_df['initial_distance'] == farthest]['visibility_ratio'].mean()
        
        diff = closest_vis - farthest_vis
        pct_change = (diff / farthest_vis * 100) if farthest_vis > 0 else 0
        
        print(f"\n1. Distance Impact:")
        print(f"   • Closest distance ({int(closest)}): {closest_vis:.1%} visibility")
        print(f"   • Farthest distance ({int(farthest)}): {farthest_vis:.1%} visibility")
        print(f"   • Difference: {diff:+.1%} ({pct_change:+.1f}% change)")
    
    # Visibility changes vs distance
    if 'visibility_changes' in valid_df.columns:
        print(f"\n2. Visibility Dynamics:")
        for distance in sorted(valid_df['initial_distance'].unique()):
            subset = valid_df[valid_df['initial_distance'] == distance]
            avg_changes = subset['visibility_changes'].mean()
            print(f"   • Distance {int(distance)}: {avg_changes:.1f} avg visibility transitions")
    
    # Success rate correlation
    if 'eval_success' in valid_df.columns:
        print(f"\n3. Success Rate by Distance:")
        success_by_dist = valid_df.groupby('initial_distance')['eval_success'].agg(['mean', 'count'])
        for distance, row in success_by_dist.iterrows():
            print(f"   • Distance {int(distance)}: {row['mean']:.1%} success ({int(row['count'])} scenarios)")
    
    return valid_df

def create_visualizations(df):
    """Create visualization plots for the analysis."""
    
    print("\n" + "="*70)
    print("📊 Creating Visualizations...")
    print("="*70)
    
    valid_df = df.dropna(subset=['initial_distance', 'visibility_ratio'])
    
    if len(valid_df) == 0:
        print("❌ No valid data for visualization")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Initial Distance vs In-View Rate Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot: Visibility ratio by initial distance
    ax1 = axes[0, 0]
    valid_df.boxplot(column='visibility_ratio', by='initial_distance', ax=ax1)
    ax1.set_title('Visibility Ratio Distribution by Initial Distance')
    ax1.set_xlabel('Initial Distance')
    ax1.set_ylabel('Visibility Ratio')
    ax1.get_figure().suptitle('')  # Remove default title
    
    # 2. Scatter plot with trend line
    ax2 = axes[0, 1]
    ax2.scatter(valid_df['initial_distance'], valid_df['visibility_ratio'], alpha=0.5)
    
    # Add trend line
    z = np.polyfit(valid_df['initial_distance'], valid_df['visibility_ratio'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['initial_distance'].min(), valid_df['initial_distance'].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    
    ax2.set_title('Visibility Ratio vs Initial Distance (with trend)')
    ax2.set_xlabel('Initial Distance')
    ax2.set_ylabel('Visibility Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bar chart: Average visibility by distance
    ax3 = axes[1, 0]
    distance_avg = valid_df.groupby('initial_distance')['visibility_ratio'].mean()
    distance_avg.plot(kind='bar', ax=ax3, color='skyblue')
    ax3.set_title('Average Visibility Ratio by Initial Distance')
    ax3.set_xlabel('Initial Distance')
    ax3.set_ylabel('Average Visibility Ratio')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(distance_avg.items()):
        ax3.text(i, value + 0.01, f'{value:.2%}', ha='center', va='bottom')
    
    # 4. Heatmap: Distance vs Success (if available)
    ax4 = axes[1, 1]
    if 'eval_success' in valid_df.columns and 'hidden_cost_style' in valid_df.columns:
        pivot = valid_df.pivot_table(
            values='visibility_ratio',
            index='initial_distance',
            columns='hidden_cost_style',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Visibility Ratio'})
        ax4.set_title('Visibility Ratio by Distance and Behavior Style')
        ax4.set_xlabel('Behavior Style')
        ax4.set_ylabel('Initial Distance')
    else:
        # Alternative: Visible steps vs total steps
        distance_steps = valid_df.groupby('initial_distance').agg({
            'visible_steps': 'mean',
            'total_trajectory_steps': 'mean'
        })
        distance_steps.plot(kind='bar', ax=ax4)
        ax4.set_title('Average Steps by Initial Distance')
        ax4.set_xlabel('Initial Distance')
        ax4.set_ylabel('Steps')
        ax4.legend(['Visible Steps', 'Total Steps'])
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'distance_visibility_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_file}")
    
    plt.show()

def main():
    """Main analysis function."""
    print("="*70)
    print("🔍 Initial Distance vs In-View Rate Analysis")
    print("="*70)
    
    # Load data
    df = load_evaluation_results()
    
    if df is None:
        print("\n⚠️ Could not load evaluation results.")
        print("   Make sure you have run main.py to generate evaluation_results_*.csv")
        return
    
    # Perform analysis
    analyzed_df = analyze_distance_visibility(df)
    
    if analyzed_df is not None:
        # Create visualizations
        try:
            create_visualizations(analyzed_df)
        except Exception as e:
            print(f"\n⚠️ Could not create visualizations: {e}")
            print("   (Analysis results are still valid)")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    main()

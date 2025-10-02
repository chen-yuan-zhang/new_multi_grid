"""
Comprehensive Analysis Script for Goal Recognition Evaluation Results

This script analyzes the output CSV files from main.py and generates:
- Summary statistics tables
- Performance breakdown by grid size and initial distance
- Visualization plots (success rates, convergence, execution time, visibility)
- Comparative analysis across different scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Calculate convergence efficiency: (total_trajectory_steps - convergence_step) / total_trajectory_steps for successful cases, 0 otherwise
    df['convergence_efficiency'] = df.apply(
        lambda row: (row['total_trajectory_steps'] - row['eval_convergence_step']) / row['total_trajectory_steps'] 
                    if row['eval_success'] == True and row['total_trajectory_steps'] > 0 
                    else 0.0,
        axis=1
    )
    
    print(f"✅ Loaded {len(df)} scenarios from {csv_path}")
    return df

def print_overall_summary(df: pd.DataFrame) -> None:
    """Print overall summary statistics."""
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY STATISTICS")
    print("="*80)
    
    total_scenarios = len(df)
    success_count = df['eval_success'].sum()
    success_rate = (success_count / total_scenarios) * 100
    
    print(f"Total scenarios evaluated: {total_scenarios}")
    print(f"Successful scenarios: {success_count} ({success_rate:.2f}%)")
    print(f"Failed scenarios: {total_scenarios - success_count} ({100-success_rate:.2f}%)")
    
    # Convergence efficiency: (total_trajectory_steps - convergence_step) / total_trajectory_steps
    avg_convergence_efficiency = df['convergence_efficiency'].mean()
    print(f"\nConvergence efficiency: {avg_convergence_efficiency:.4f}")
    print(f"  (Formula: (total_trajectory_steps - convergence_step) / total_trajectory_steps for successes, 0 for failures)")
    
    # Execution time
    avg_exec_time = df['eval_execution_time'].mean()
    total_exec_time = df['eval_execution_time'].sum()
    print(f"\nExecution time:")
    print(f"  - Average per scenario: {avg_exec_time:.3f}s")
    print(f"  - Total time: {total_exec_time:.2f}s ({total_exec_time/60:.2f} minutes)")
    print(f"  - Min: {df['eval_execution_time'].min():.3f}s")
    print(f"  - Max: {df['eval_execution_time'].max():.3f}s")
    
    # Confidence statistics
    avg_final_conf = df['final_confidence'].mean()
    avg_max_conf = df['max_confidence_reached'].mean()
    print(f"\nConfidence:")
    print(f"  - Average final confidence: {avg_final_conf:.4f}")
    print(f"  - Average max confidence reached: {avg_max_conf:.4f}")
    
    # Visibility statistics
    avg_visibility = df['visibility_ratio'].mean()
    avg_visible_steps = df['visible_steps'].mean()
    avg_visibility_changes = df['visibility_changes'].mean()
    print(f"\nVisibility:")
    print(f"  - Average visibility ratio: {avg_visibility:.3f}")
    print(f"  - Average visible steps: {avg_visible_steps:.1f}")
    print(f"  - Average visibility changes: {avg_visibility_changes:.1f}")
    
    # Cache statistics
    if 'cache_hit_rate' in df.columns:
        avg_cache_hit_rate = df['cache_hit_rate'].mean()
        print(f"\nCache performance:")
        print(f"  - Average cache hit rate: {avg_cache_hit_rate:.2%}")

def analyze_by_grid_size(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by grid size."""
    print("\n" + "="*80)
    print("📏 ANALYSIS BY GRID SIZE")
    print("="*80)
    
    grouped = df.groupby('size').agg({
        'eval_success': ['count', 'sum', 'mean'],
        'convergence_efficiency': 'mean',
        'eval_execution_time': 'mean',
        'final_confidence': 'mean',
        'visibility_ratio': 'mean',
        'cache_hit_rate': 'mean'
    }).round(4)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={
        'eval_success_count': 'total_scenarios',
        'eval_success_sum': 'successes',
        'eval_success_mean': 'success_rate',
        'convergence_efficiency_mean': 'avg_convergence_efficiency',
        'eval_execution_time_mean': 'avg_exec_time',
        'final_confidence_mean': 'avg_final_confidence',
        'visibility_ratio_mean': 'avg_visibility_ratio',
        'cache_hit_rate_mean': 'avg_cache_hit_rate'
    }, inplace=True)
    
    print(grouped.to_string())
    return grouped

def analyze_by_initial_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by initial distance."""
    print("\n" + "="*80)
    print("📍 ANALYSIS BY INITIAL DISTANCE")
    print("="*80)
    
    grouped = df.groupby('initial_distance').agg({
        'eval_success': ['count', 'sum', 'mean'],
        'convergence_efficiency': 'mean',
        'eval_execution_time': 'mean',
        'final_confidence': 'mean',
        'visibility_ratio': 'mean',
        'cache_hit_rate': 'mean'
    }).round(4)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={
        'eval_success_count': 'total_scenarios',
        'eval_success_sum': 'successes',
        'eval_success_mean': 'success_rate',
        'convergence_efficiency_mean': 'avg_convergence_efficiency',
        'eval_execution_time_mean': 'avg_exec_time',
        'final_confidence_mean': 'avg_final_confidence',
        'visibility_ratio_mean': 'avg_visibility_ratio',
        'cache_hit_rate_mean': 'avg_cache_hit_rate'
    }, inplace=True)
    
    print(grouped.to_string())
    return grouped

def analyze_by_behavior_style(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by behavior style."""
    if 'hidden_cost_style' not in df.columns:
        return None
        
    print("\n" + "="*80)
    print("🎭 ANALYSIS BY BEHAVIOR STYLE")
    print("="*80)
    
    grouped = df.groupby('hidden_cost_style').agg({
        'eval_success': ['count', 'sum', 'mean'],
        'convergence_efficiency': 'mean',
        'eval_execution_time': 'mean',
        'final_confidence': 'mean',
        'visibility_ratio': 'mean'
    }).round(4)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={
        'eval_success_count': 'total_scenarios',
        'eval_success_sum': 'successes',
        'eval_success_mean': 'success_rate',
        'convergence_efficiency_mean': 'avg_convergence_efficiency',
        'eval_execution_time_mean': 'avg_exec_time',
        'final_confidence_mean': 'avg_final_confidence',
        'visibility_ratio_mean': 'avg_visibility_ratio'
    }, inplace=True)
    
    print(grouped.to_string())
    return grouped

def analyze_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results by combination of grid size and initial distance."""
    print("\n" + "="*80)
    print("📊 COMBINED ANALYSIS (Grid Size × Initial Distance)")
    print("="*80)
    
    # Create pivot table for success rate
    pivot_success = df.pivot_table(
        values='eval_success',
        index='size',
        columns='initial_distance',
        aggfunc='mean'
    ).round(4)
    
    print("\nSuccess Rate:")
    print(pivot_success.to_string())
    
    # Create pivot table for convergence efficiency
    pivot_convergence_eff = df.pivot_table(
        values='convergence_efficiency',
        index='size',
        columns='initial_distance',
        aggfunc='mean'
    ).round(4)
    
    print("\nAverage Convergence Efficiency:")
    print(pivot_convergence_eff.to_string())
    
    return pivot_success, pivot_convergence_eff

def plot_success_rate_by_size(df: pd.DataFrame, output_dir: str) -> None:
    """Plot success rate by grid size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby('size')['eval_success'].mean()
    grouped.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Grid Size', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(grouped.values):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_size.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: success_rate_by_size.png")
    plt.close()

def plot_success_rate_by_distance(df: pd.DataFrame, output_dir: str) -> None:
    """Plot success rate by initial distance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby('initial_distance')['eval_success'].mean()
    grouped.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Initial Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Initial Distance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(grouped.values):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_distance.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: success_rate_by_distance.png")
    plt.close()

def plot_heatmap_success_rate(df: pd.DataFrame, output_dir: str) -> None:
    """Plot heatmap of success rate by size and distance."""
    pivot = df.pivot_table(
        values='eval_success',
        index='size',
        columns='initial_distance',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1, 
                ax=ax, cbar_kws={'label': 'Success Rate'}, linewidths=0.5)
    
    ax.set_xlabel('Initial Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grid Size', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Heatmap (Grid Size × Initial Distance)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: success_rate_heatmap.png")
    plt.close()

def plot_convergence_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Plot convergence efficiency analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Convergence efficiency by grid size
    grouped_size = df.groupby('size')['convergence_efficiency'].mean()
    grouped_size.plot(kind='bar', ax=axes[0], color='lightgreen', edgecolor='black')
    axes[0].set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Convergence Efficiency', fontsize=12, fontweight='bold')
    axes[0].set_title('Convergence Efficiency by Grid Size', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_size.values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Convergence efficiency by initial distance
    grouped_dist = df.groupby('initial_distance')['convergence_efficiency'].mean()
    grouped_dist.plot(kind='bar', ax=axes[1], color='lightsalmon', edgecolor='black')
    axes[1].set_xlabel('Initial Distance', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Convergence Efficiency', fontsize=12, fontweight='bold')
    axes[1].set_title('Convergence Efficiency by Initial Distance', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_dist.values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: convergence_analysis.png")
    plt.close()

def plot_execution_time_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Plot execution time analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Execution time by grid size
    grouped_size = df.groupby('size')['eval_execution_time'].mean()
    grouped_size.plot(kind='bar', ax=axes[0], color='plum', edgecolor='black')
    axes[0].set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Execution Time (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_title('Execution Time by Grid Size', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_size.values):
        axes[0].text(i, v + 0.001, f'{v:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Execution time by initial distance
    grouped_dist = df.groupby('initial_distance')['eval_execution_time'].mean()
    grouped_dist.plot(kind='bar', ax=axes[1], color='khaki', edgecolor='black')
    axes[1].set_xlabel('Initial Distance', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Execution Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Execution Time by Initial Distance', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_dist.values):
        axes[1].text(i, v + 0.001, f'{v:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: execution_time_analysis.png")
    plt.close()

def plot_visibility_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Plot visibility ratio analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Visibility ratio by grid size
    grouped_size = df.groupby('size')['visibility_ratio'].mean()
    grouped_size.plot(kind='bar', ax=axes[0], color='lightblue', edgecolor='black')
    axes[0].set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Visibility Ratio', fontsize=12, fontweight='bold')
    axes[0].set_title('Visibility Ratio by Grid Size', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_size.values):
        axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Visibility ratio by initial distance
    grouped_dist = df.groupby('initial_distance')['visibility_ratio'].mean()
    grouped_dist.plot(kind='bar', ax=axes[1], color='peachpuff', edgecolor='black')
    axes[1].set_xlabel('Initial Distance', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Visibility Ratio', fontsize=12, fontweight='bold')
    axes[1].set_title('Visibility Ratio by Initial Distance', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(grouped_dist.values):
        axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visibility_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: visibility_analysis.png")
    plt.close()

def save_summary_tables(df: pd.DataFrame, output_dir: str) -> None:
    """Save summary tables to CSV files."""
    # Table 1: By grid size
    table_size = df.groupby('size').agg({
        'eval_success': ['count', 'sum', 'mean'],
        'eval_convergence_step': 'mean',
        'eval_execution_time': 'mean',
        'final_confidence': 'mean',
        'visibility_ratio': 'mean'
    }).round(4)
    table_size.columns = ['_'.join(col).strip() for col in table_size.columns.values]
    table_size.to_csv(os.path.join(output_dir, 'summary_by_size.csv'))
    print(f"  📄 Saved: summary_by_size.csv")
    
    # Table 2: By initial distance
    table_distance = df.groupby('initial_distance').agg({
        'eval_success': ['count', 'sum', 'mean'],
        'eval_convergence_step': 'mean',
        'eval_execution_time': 'mean',
        'final_confidence': 'mean',
        'visibility_ratio': 'mean'
    }).round(4)
    table_distance.columns = ['_'.join(col).strip() for col in table_distance.columns.values]
    table_distance.to_csv(os.path.join(output_dir, 'summary_by_distance.csv'))
    print(f"  📄 Saved: summary_by_distance.csv")
    
    # Table 3: Combined pivot table
    pivot_success = df.pivot_table(
        values='eval_success',
        index='size',
        columns='initial_distance',
        aggfunc='mean'
    ).round(4)
    pivot_success.to_csv(os.path.join(output_dir, 'success_rate_pivot.csv'))
    print(f"  📄 Saved: success_rate_pivot.csv")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze goal recognition evaluation results and generate reports'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (e.g., greedy_evaluation_results_test.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory to save analysis outputs (default: analysis_output)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}\n")
    
    # Load results
    df = load_results(args.input)
    
    # Print overall summary
    print_overall_summary(df)
    
    # Analyze by different groupings
    analyze_by_grid_size(df)
    analyze_by_initial_distance(df)
    analyze_by_behavior_style(df)
    analyze_combined(df)
    
    # Generate plots
    print("\n" + "="*80)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_success_rate_by_size(df, args.output_dir)
    plot_success_rate_by_distance(df, args.output_dir)
    plot_heatmap_success_rate(df, args.output_dir)
    plot_convergence_analysis(df, args.output_dir)
    plot_execution_time_analysis(df, args.output_dir)
    plot_visibility_analysis(df, args.output_dir)
    
    # Save summary tables
    print("\n" + "="*80)
    print("💾 SAVING SUMMARY TABLES")
    print("="*80)
    save_summary_tables(df, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"All outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - success_rate_by_size.png")
    print("    - success_rate_by_distance.png")
    print("    - success_rate_heatmap.png")
    print("    - convergence_analysis.png")
    print("    - execution_time_analysis.png")
    print("    - visibility_analysis.png")
    print("  Tables:")
    print("    - summary_by_size.csv")
    print("    - summary_by_distance.csv")
    print("    - success_rate_pivot.csv")

if __name__ == "__main__":
    main()

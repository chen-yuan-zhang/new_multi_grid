"""
Analyze Ablation Study Results

This script analyzes the results from the observer mode ablation study,
comparing different combinations of observer action modes and belief update modes.

It loads all result CSV files, computes comprehensive statistics, generates
comparison tables and visualizations, and produces a detailed analysis report.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Define all mode combinations
ACTION_MODES = ['greedy', 'stay', 'random']
BELIEF_MODES = ['bayesian', 'optimal', 'uniform']

# Nice labels for plotting
ACTION_MODE_LABELS = {
    'greedy': 'Greedy (Active)',
    'stay': 'Stay (Stationary)',
    'random': 'Random'
}

BELIEF_MODE_LABELS = {
    'bayesian': 'Bayesian (Full)',
    'optimal': 'Optimal (Point)',
    'uniform': 'Uniform (Baseline)'
}


def load_all_results(base_dir: Path, prefix: str = 'results') -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Load all result CSV files from the ablation study.
    
    Args:
        base_dir: Directory containing result files
        prefix: Filename prefix (default: 'results', can be 'symbolic_results')
        
    Returns:
        Dictionary mapping (action_mode, belief_mode) to DataFrame
    """
    results = {}
    missing = []
    
    for action_mode in ACTION_MODES:
        for belief_mode in BELIEF_MODES:
            filename = base_dir / f"{prefix}_{action_mode}_{belief_mode}.csv"
            
            if filename.exists():
                df = pd.read_csv(filename)
                results[(action_mode, belief_mode)] = df
                print(f"✓ Loaded: {filename.name} ({len(df)} scenarios)")
            else:
                missing.append(f"{action_mode}_{belief_mode}")
                print(f"✗ Missing: {filename.name}")
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} result files missing: {', '.join(missing)}")
    
    return results


def compute_summary_statistics(results: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """
    Compute summary statistics for all configurations.
    
    Returns:
        DataFrame with one row per configuration
    """
    summaries = []
    
    for (action_mode, belief_mode), df in results.items():
        # Basic success metrics
        total_scenarios = len(df)
        successful = df['eval_success'].sum()
        success_rate = df['eval_success'].mean()
        
        # Compute convergence rate: (total_steps - convergence_step) / total_steps
        # 0 if not successful
        convergence_rates = []
        for _, row in df.iterrows():
            if row['eval_success']:
                # Get total trajectory steps
                if 'total_trajectory_steps' in df.columns:
                    total_steps = row['total_trajectory_steps']
                elif 'total_steps' in df.columns:
                    total_steps = row['total_steps']
                else:
                    # Fallback: estimate from convergence step
                    total_steps = row['eval_convergence_step'] * 1.5
                
                convergence_step = row['eval_convergence_step']
                convergence_rate = (total_steps - convergence_step) / total_steps if total_steps > 0 else 0.0
            else:
                convergence_rate = 0.0
            convergence_rates.append(convergence_rate)
        
        df['convergence_rate'] = convergence_rates
        
        # Convergence rate statistics
        avg_convergence_rate = np.mean(convergence_rates)
        std_convergence_rate = np.std(convergence_rates)
        median_convergence_rate = np.median(convergence_rates)
        
        # Convergence rate for successful cases only
        successful_convergence_rates = [r for r, success in zip(convergence_rates, df['eval_success']) if success]
        if successful_convergence_rates:
            avg_convergence_rate_successful = np.mean(successful_convergence_rates)
            std_convergence_rate_successful = np.std(successful_convergence_rates)
        else:
            avg_convergence_rate_successful = 0.0
            std_convergence_rate_successful = 0.0
        
        # Convergence step metrics (for reference)
        successful_df = df[df['eval_success'] == True]
        if len(successful_df) > 0:
            avg_convergence_step = successful_df['eval_convergence_step'].mean()
            std_convergence_step = successful_df['eval_convergence_step'].std()
            median_convergence_step = successful_df['eval_convergence_step'].median()
        else:
            avg_convergence_step = np.nan
            std_convergence_step = np.nan
            median_convergence_step = np.nan
        
        # Execution time
        avg_exec_time = df['eval_execution_time'].mean()
        total_exec_time = df['eval_execution_time'].sum()
        
        # Visibility metrics
        avg_visibility = df['visibility_ratio'].mean()
        avg_visibility_changes = df['visibility_changes'].mean()
        
        # Confidence metrics
        avg_final_confidence = df['final_confidence'].mean()
        avg_max_confidence = df['max_confidence_reached'].mean()
        
        # Cache metrics (if available)
        if 'cache_hit_rate' in df.columns:
            avg_cache_hit_rate = df['cache_hit_rate'].mean()
        else:
            avg_cache_hit_rate = np.nan
        
        summaries.append({
            'action_mode': action_mode,
            'belief_mode': belief_mode,
            'config': f"{action_mode}_{belief_mode}",
            'total_scenarios': total_scenarios,
            'successful': successful,
            'success_rate': success_rate,
            'avg_convergence_rate': avg_convergence_rate,
            'std_convergence_rate': std_convergence_rate,
            'median_convergence_rate': median_convergence_rate,
            'avg_convergence_rate_successful': avg_convergence_rate_successful,
            'std_convergence_rate_successful': std_convergence_rate_successful,
            'avg_convergence_step': avg_convergence_step,
            'std_convergence_step': std_convergence_step,
            'median_convergence_step': median_convergence_step,
            'avg_execution_time': avg_exec_time,
            'total_execution_time': total_exec_time,
            'avg_visibility_ratio': avg_visibility,
            'avg_visibility_changes': avg_visibility_changes,
            'avg_final_confidence': avg_final_confidence,
            'avg_max_confidence': avg_max_confidence,
            'avg_cache_hit_rate': avg_cache_hit_rate
        })
    
    summary_df = pd.DataFrame(summaries)
    
    # Sort by success rate (descending) and then by convergence rate (descending - higher is better)
    summary_df = summary_df.sort_values(
        by=['success_rate', 'avg_convergence_rate'],
        ascending=[False, False]
    )
    
    return summary_df


def analyze_by_grid_size(results: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze performance by grid size for each configuration.
    """
    analyses = []
    
    for (action_mode, belief_mode), df in results.items():
        if 'size' not in df.columns:
            continue
        
        # Ensure convergence_rate is computed
        if 'convergence_rate' not in df.columns:
            convergence_rates = []
            for _, row in df.iterrows():
                if row['eval_success']:
                    if 'total_trajectory_steps' in df.columns:
                        total_steps = row['total_trajectory_steps']
                    elif 'total_steps' in df.columns:
                        total_steps = row['total_steps']
                    else:
                        total_steps = row['eval_convergence_step'] * 1.5
                    
                    convergence_step = row['eval_convergence_step']
                    convergence_rate = (total_steps - convergence_step) / total_steps if total_steps > 0 else 0.0
                else:
                    convergence_rate = 0.0
                convergence_rates.append(convergence_rate)
            df['convergence_rate'] = convergence_rates
            
        for size in sorted(df['size'].unique()):
            size_df = df[df['size'] == size]
            
            analyses.append({
                'action_mode': action_mode,
                'belief_mode': belief_mode,
                'config': f"{action_mode}_{belief_mode}",
                'grid_size': size,
                'scenarios': len(size_df),
                'success_rate': size_df['eval_success'].mean(),
                'avg_convergence_rate': size_df['convergence_rate'].mean(),
                'avg_convergence_step': size_df[size_df['eval_success']]['eval_convergence_step'].mean(),
                'avg_exec_time': size_df['eval_execution_time'].mean()
            })
    
    return pd.DataFrame(analyses)


def analyze_by_behavior(results: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze performance by behavior type for each configuration.
    """
    analyses = []
    
    for (action_mode, belief_mode), df in results.items():
        if 'hidden_cost_style' not in df.columns:
            continue
        
        # Ensure convergence_rate is computed
        if 'convergence_rate' not in df.columns:
            convergence_rates = []
            for _, row in df.iterrows():
                if row['eval_success']:
                    if 'total_trajectory_steps' in df.columns:
                        total_steps = row['total_trajectory_steps']
                    elif 'total_steps' in df.columns:
                        total_steps = row['total_steps']
                    else:
                        total_steps = row['eval_convergence_step'] * 1.5
                    
                    convergence_step = row['eval_convergence_step']
                    convergence_rate = (total_steps - convergence_step) / total_steps if total_steps > 0 else 0.0
                else:
                    convergence_rate = 0.0
                convergence_rates.append(convergence_rate)
            df['convergence_rate'] = convergence_rates
            
        for behavior in sorted(df['hidden_cost_style'].unique()):
            behavior_df = df[df['hidden_cost_style'] == behavior]
            
            analyses.append({
                'action_mode': action_mode,
                'belief_mode': belief_mode,
                'config': f"{action_mode}_{belief_mode}",
                'behavior': behavior,
                'scenarios': len(behavior_df),
                'success_rate': behavior_df['eval_success'].mean(),
                'avg_convergence_rate': behavior_df['convergence_rate'].mean(),
                'avg_convergence_step': behavior_df[behavior_df['eval_success']]['eval_convergence_step'].mean(),
                'avg_exec_time': behavior_df['eval_execution_time'].mean()
            })
    
    return pd.DataFrame(analyses)


def plot_success_rates(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create bar plot comparing success rates across configurations.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data
    configs = summary_df['config'].values
    success_rates = summary_df['success_rate'].values * 100
    
    # Color by action mode
    colors = []
    for _, row in summary_df.iterrows():
        if row['action_mode'] == 'greedy':
            colors.append('#2ecc71')  # Green
        elif row['action_mode'] == 'stay':
            colors.append('#3498db')  # Blue
        else:  # random
            colors.append('#e74c3c')  # Red
    
    bars = ax.bar(range(len(configs)), success_rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Goal Recognition Success Rate by Observer Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', label='Greedy (Active)'),
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='Stay (Stationary)'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Random')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / 'success_rates_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_convergence_speed(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create plot comparing convergence rate across configurations.
    """
    # Filter to only successful configurations
    successful_configs = summary_df[summary_df['success_rate'] > 0].copy()
    
    if len(successful_configs) == 0:
        print("⚠️  No successful configurations to plot convergence rate")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    configs = successful_configs['config'].values
    convergence_rates = successful_configs['avg_convergence_rate'].values * 100  # Convert to percentage
    
    # Color by belief mode
    colors = []
    for _, row in successful_configs.iterrows():
        if row['belief_mode'] == 'bayesian':
            colors.append('#9b59b6')  # Purple
        elif row['belief_mode'] == 'optimal':
            colors.append('#f39c12')  # Orange
        else:  # uniform
            colors.append('#95a5a6')  # Gray
    
    bars = ax.bar(range(len(configs)), convergence_rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, rate in zip(bars, convergence_rates):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Convergence Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Rate by Observer Configuration (Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylim(0, max(convergence_rates) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', alpha=0.7, edgecolor='black', label='Bayesian (Full)'),
        Patch(facecolor='#f39c12', alpha=0.7, edgecolor='black', label='Optimal (Point)'),
        Patch(facecolor='#95a5a6', alpha=0.7, edgecolor='black', label='Uniform (Baseline)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / 'convergence_rate_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_heatmap_matrix(summary_df: pd.DataFrame, metric: str, output_dir: Path):
    """
    Create heatmap showing metric values in action_mode x belief_mode matrix.
    """
    # Create pivot table
    pivot_data = summary_df.pivot(index='action_mode', columns='belief_mode', values=metric)
    
    # Reorder to match our preferred order
    pivot_data = pivot_data.reindex(index=ACTION_MODES, columns=BELIEF_MODES)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Choose colormap based on metric
    if 'success' in metric or 'confidence' in metric or 'convergence_rate' in metric:
        cmap = 'YlGn'  # Higher is better
        if 'rate' in metric:
            fmt = '.2%'
        elif 'convergence_rate' in metric:
            fmt = '.3f'
        else:
            fmt = '.3f'
    elif 'convergence' in metric or 'time' in metric:
        cmap = 'YlOrRd_r'  # Lower is better
        fmt = '.1f'
    else:
        cmap = 'coolwarm'
        fmt = '.3f'
    
    sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, 
                linewidths=0.5, ax=ax, cbar_kws={'label': metric})
    
    ax.set_xlabel('Belief Update Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observer Action Mode', fontsize=12, fontweight='bold')
    
    # Nice title
    title_map = {
        'success_rate': 'Success Rate',
        'avg_convergence_rate': 'Average Convergence Rate',
        'avg_convergence_step': 'Average Convergence Step',
        'avg_execution_time': 'Average Execution Time (s)',
        'avg_visibility_ratio': 'Average Visibility Ratio',
        'avg_final_confidence': 'Average Final Confidence'
    }
    title = title_map.get(metric, metric.replace('_', ' ').title())
    ax.set_title(f'{title} by Observer Configuration', fontsize=14, fontweight='bold')
    
    # Relabel axes with nice names
    ax.set_xticklabels([BELIEF_MODE_LABELS[b] for b in BELIEF_MODES])
    ax.set_yticklabels([ACTION_MODE_LABELS[a] for a in ACTION_MODES], rotation=0)
    
    plt.tight_layout()
    output_file = output_dir / f'heatmap_{metric}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_grid_size_comparison(grid_analysis: pd.DataFrame, output_dir: Path):
    """
    Plot success rate by grid size for each configuration.
    """
    if len(grid_analysis) == 0:
        print("⚠️  No grid size data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get unique configurations and grid sizes
    configs = grid_analysis['config'].unique()
    grid_sizes = sorted(grid_analysis['grid_size'].unique())
    
    # Plot each configuration
    for i, config in enumerate(configs):
        config_data = grid_analysis[grid_analysis['config'] == config]
        success_rates = []
        for size in grid_sizes:
            size_data = config_data[config_data['grid_size'] == size]
            if len(size_data) > 0:
                success_rates.append(size_data['success_rate'].values[0] * 100)
            else:
                success_rates.append(np.nan)
        
        ax.plot(grid_sizes, success_rates, marker='o', label=config, linewidth=2, markersize=6)
    
    ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Grid Size', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output_file = output_dir / 'success_by_grid_size.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_report(summary_df: pd.DataFrame, 
                   grid_analysis: pd.DataFrame,
                   behavior_analysis: pd.DataFrame,
                   output_dir: Path):
    """
    Generate a comprehensive markdown report.
    """
    report_lines = [
        "# Observer Mode Ablation Study - Analysis Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"This report analyzes {len(summary_df)} different observer configurations,",
        "comparing combinations of observer action modes (greedy, stay, random) and",
        "belief update modes (bayesian, optimal, uniform).",
        "",
        "## Overall Performance Ranking",
        "",
        "Configurations ranked by success rate:",
        ""
    ]
    
    # Add ranking table
    report_lines.append("| Rank | Configuration | Success Rate | Avg Convergence Rate | Avg Time (s) |")
    report_lines.append("|------|---------------|--------------|----------------------|--------------|")
    
    for i, (idx, row) in enumerate(summary_df.iterrows(), 1):
        conv_rate = f"{row['avg_convergence_rate']*100:.1f}%" if not pd.isna(row['avg_convergence_rate']) else "N/A"
        report_lines.append(
            f"| {i} | {row['config']} | {row['success_rate']*100:.1f}% | "
            f"{conv_rate} | {row['avg_execution_time']:.3f} |"
        )
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### 1. Effect of Observer Movement",
        ""
    ])
    
    # Compare action modes
    action_comparison = summary_df.groupby('action_mode').agg({
        'success_rate': 'mean',
        'avg_convergence_rate': 'mean',
        'avg_execution_time': 'mean'
    })
    
    report_lines.append("Average performance by action mode:")
    report_lines.append("")
    report_lines.append("| Action Mode | Avg Success Rate | Avg Convergence Rate | Avg Time |")
    report_lines.append("|-------------|------------------|----------------------|----------|")
    
    for action_mode in ACTION_MODES:
        if action_mode in action_comparison.index:
            row = action_comparison.loc[action_mode]
            conv_rate = f"{row['avg_convergence_rate']*100:.1f}%" if not pd.isna(row['avg_convergence_rate']) else "N/A"
            report_lines.append(
                f"| {ACTION_MODE_LABELS[action_mode]} | {row['success_rate']*100:.1f}% | "
                f"{conv_rate} | {row['avg_execution_time']:.3f}s |"
            )
    
    report_lines.extend([
        "",
        "### 2. Effect of Belief Update Strategy",
        ""
    ])
    
    # Compare belief modes
    belief_comparison = summary_df.groupby('belief_mode').agg({
        'success_rate': 'mean',
        'avg_convergence_rate': 'mean',
        'avg_execution_time': 'mean'
    })
    
    report_lines.append("Average performance by belief mode:")
    report_lines.append("")
    report_lines.append("| Belief Mode | Avg Success Rate | Avg Convergence Rate | Avg Time |")
    report_lines.append("|-------------|------------------|----------------------|----------|")
    
    for belief_mode in BELIEF_MODES:
        if belief_mode in belief_comparison.index:
            row = belief_comparison.loc[belief_mode]
            conv_rate = f"{row['avg_convergence_rate']*100:.1f}%" if not pd.isna(row['avg_convergence_rate']) else "N/A"
            report_lines.append(
                f"| {BELIEF_MODE_LABELS[belief_mode]} | {row['success_rate']*100:.1f}% | "
                f"{conv_rate} | {row['avg_execution_time']:.3f}s |"
            )
    
    # Add grid size analysis if available
    if len(grid_analysis) > 0:
        report_lines.extend([
            "",
            "### 3. Performance by Grid Size",
            "",
            "Success rates tend to vary with grid size. Larger grids generally present",
            "more challenging scenarios due to increased search space and longer trajectories.",
            ""
        ])
    
    # Add behavior analysis if available
    if len(behavior_analysis) > 0:
        report_lines.extend([
            "",
            "### 4. Performance by Behavior Type",
            "",
            "Different behavior types (like_wall, hate_wall, like_edge, hate_edge) may",
            "exhibit different difficulty levels for goal recognition.",
            ""
        ])
    
    report_lines.extend([
        "",
        "## Visualizations",
        "",
        "The following visualizations have been generated:",
        "",
        "1. `success_rates_comparison.png` - Bar chart of success rates",
        "2. `convergence_rate_comparison.png` - Convergence rate comparison",
        "3. `heatmap_success_rate.png` - Heatmap of success rates",
        "4. `heatmap_avg_convergence_rate.png` - Heatmap of convergence rates",
        "5. `success_by_grid_size.png` - Success rate trends by grid size",
        "",
        "## Conclusions",
        "",
        "This ablation study provides insights into which components of the observer",
        "system contribute most to goal recognition performance:",
        "",
        "- **Active movement** (greedy vs stay) shows the importance of information gathering",
        "- **Belief tracking** (bayesian vs optimal vs uniform) shows the value of inference",
        "- **Combined effects** reveal potential synergies or tradeoffs between components",
        "",
        "**Convergence Rate** measures how early in the trajectory the goal is identified:",
        "- Formula: (total_steps - convergence_step) / total_steps",
        "- Range: 0-1 (0% to 100%)",
        "- Higher is better (earlier identification)",
        "- 0 for failed scenarios",
        "",
        "## Recommendations",
        "",
        "Based on the results:",
        ""
    ])
    
    # Add recommendation based on best configuration
    best_config = summary_df.iloc[0]
    report_lines.append(
        f"- **Best overall:** `{best_config['config']}` "
        f"({best_config['success_rate']*100:.1f}% success rate)"
    )
    
    # Find fastest successful config
    successful = summary_df[summary_df['success_rate'] > 0.5]
    if len(successful) > 0:
        fastest = successful.loc[successful['avg_execution_time'].idxmin()]
        report_lines.append(
            f"- **Fastest (>50% success):** `{fastest['config']}` "
            f"({fastest['avg_execution_time']:.3f}s per scenario)"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "*This report was automatically generated by analyze_ablation_results.py*"
    ])
    
    # Write report
    report_file = output_dir / 'ablation_study_report.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from observer mode ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes all result CSV files from the ablation study and generates:
- Summary statistics table
- Comparison visualizations
- Performance rankings
- Detailed analysis report

Examples:
  # Analyze results in current directory with default prefix
  python analyze_ablation_results.py
  
  # Analyze symbolic_results files
  python analyze_ablation_results.py --prefix symbolic_results
  
  # Analyze results in specific directory
  python analyze_ablation_results.py --results-dir ../results --prefix results
  
  # Specify output directory for visualizations
  python analyze_ablation_results.py --output-dir analysis_output --prefix symbolic_results
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='.',
        help='Directory containing result CSV files (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ablation_analysis',
        help='Directory to save analysis outputs (default: ablation_analysis)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='results',
        help='Filename prefix for result files (default: results, use symbolic_results for symbolic results)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Observer Mode Ablation Study - Results Analysis")
    print("=" * 80)
    print(f"\n📂 Results directory: {results_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📝 Filename prefix: {args.prefix}\n")
    
    # Load all results
    print("📖 Loading result files...")
    results = load_all_results(results_dir, args.prefix)
    
    if len(results) == 0:
        print("\n❌ No result files found! Make sure you've run the ablation study first.")
        return
    
    print(f"\n✓ Loaded {len(results)} configurations\n")
    
    # Compute summary statistics
    print("📊 Computing summary statistics...")
    summary_df = compute_summary_statistics(results)
    
    # Save summary table
    summary_file = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved: {summary_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(summary_df.to_string(index=False))
    print()
    
    # Analyze by grid size
    print("📏 Analyzing by grid size...")
    grid_analysis = analyze_by_grid_size(results)
    if len(grid_analysis) > 0:
        grid_file = output_dir / 'analysis_by_grid_size.csv'
        grid_analysis.to_csv(grid_file, index=False)
        print(f"✓ Saved: {grid_file}")
    
    # Analyze by behavior
    print("🎭 Analyzing by behavior type...")
    behavior_analysis = analyze_by_behavior(results)
    if len(behavior_analysis) > 0:
        behavior_file = output_dir / 'analysis_by_behavior.csv'
        behavior_analysis.to_csv(behavior_file, index=False)
        print(f"✓ Saved: {behavior_file}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_success_rates(summary_df, output_dir)
    plot_convergence_speed(summary_df, output_dir)
    
    # Heatmaps for key metrics
    for metric in ['success_rate', 'avg_convergence_rate', 'avg_execution_time', 'avg_final_confidence']:
        if metric in summary_df.columns and not summary_df[metric].isna().all():
            plot_heatmap_matrix(summary_df, metric, output_dir)
    
    # Grid size comparison
    if len(grid_analysis) > 0:
        plot_grid_size_comparison(grid_analysis, output_dir)
    
    # Generate report
    print("\n📝 Generating analysis report...")
    generate_report(summary_df, grid_analysis, behavior_analysis, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - summary_statistics.csv")
    print("  - ablation_study_report.md")
    print("  - Various visualization PNG files")
    print()


if __name__ == "__main__":
    main()

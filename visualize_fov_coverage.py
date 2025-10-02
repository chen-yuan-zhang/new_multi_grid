"""
Visualize Field of View (FOV) Coverage

This script analyzes and visualizes why the in-view rate is high by:
1. Showing the FOV size and shape
2. Calculating FOV coverage percentage relative to grid size
3. Demonstrating FOV in different scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from multigrid.core.constants import Direction

def calculate_fov_coverage(grid_size, view_size):
    """Calculate what percentage of the grid is covered by FOV."""
    # FOV is view_size x view_size
    fov_cells = view_size * view_size
    total_cells = grid_size * grid_size
    coverage = (fov_cells / total_cells) * 100
    return coverage, fov_cells, total_cells

def get_fov_cells(observer_pos, observer_dir, view_size, grid_size):
    """
    Get all cells visible in the FOV.
    
    Based on the logic from observer.py update_belief function.
    """
    visible_cells = []
    
    # Direction vectors
    f_vec = Direction(observer_dir).to_vec()
    r_vec = np.array((-f_vec[1], f_vec[0]))
    
    # Top-left corner of view
    top_left = (
        observer_pos[0] + f_vec[0] * (view_size - 1) - r_vec[0] * (view_size // 2),
        observer_pos[1] + f_vec[1] * (view_size - 1) - r_vec[1] * (view_size // 2)
    )
    
    # For each cell in the FOV
    for vis_j in range(view_size):
        for vis_i in range(view_size):
            # Compute world coordinates
            abs_i = int(top_left[0] - f_vec[0] * vis_j + r_vec[0] * vis_i)
            abs_j = int(top_left[1] - f_vec[1] * vis_j + r_vec[1] * vis_i)
            
            # Check bounds
            if 0 <= abs_i < grid_size and 0 <= abs_j < grid_size:
                visible_cells.append((abs_i, abs_j))
    
    return visible_cells

def visualize_fov_scenarios():
    """Create visualizations showing FOV in different scenarios."""
    
    view_size = 5  # From goal_prediction.py line 220
    grid_sizes = [10, 12, 15]  # Common grid sizes from generator
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Field of View (FOV) Analysis: Why In-View Rate is High', 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    for grid_size in grid_sizes:
        # Calculate coverage statistics
        coverage, fov_cells, total_cells = calculate_fov_coverage(grid_size, view_size)
        
        # Show 3 different observer positions per grid size
        positions = [
            (grid_size // 2, grid_size // 2, 0, "Center"),  # Center, facing right
            (1, 1, 0, "Corner"),  # Corner
            (grid_size // 2, 1, 1, "Edge"),  # Edge, facing down
        ]
        
        for obs_pos_x, obs_pos_y, obs_dir, location in positions:
            ax = fig.add_subplot(gs[plot_idx])
            
            # Draw grid
            for i in range(grid_size + 1):
                ax.plot([0, grid_size], [i, i], 'k-', alpha=0.2, linewidth=0.5)
                ax.plot([i, i], [0, grid_size], 'k-', alpha=0.2, linewidth=0.5)
            
            # Get FOV cells
            observer_pos = (obs_pos_x, obs_pos_y)
            visible_cells = get_fov_cells(observer_pos, obs_dir, view_size, grid_size)
            
            # Draw FOV cells
            for cell_x, cell_y in visible_cells:
                rect = Rectangle((cell_x, cell_y), 1, 1, 
                               facecolor='yellow', alpha=0.5, edgecolor='orange', linewidth=1)
                ax.add_patch(rect)
            
            # Draw observer
            obs_rect = Rectangle((obs_pos_x, obs_pos_y), 1, 1,
                                facecolor='blue', alpha=0.8, edgecolor='darkblue', linewidth=2)
            ax.add_patch(obs_rect)
            
            # Draw direction indicator
            dir_vec = Direction(obs_dir).to_vec()
            ax.arrow(obs_pos_x + 0.5, obs_pos_y + 0.5,
                    dir_vec[0] * 0.7, dir_vec[1] * 0.7,
                    head_width=0.3, head_length=0.2, fc='darkblue', ec='darkblue', linewidth=2)
            
            # Set limits and labels
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            actual_coverage = (len(visible_cells) / total_cells) * 100
            
            ax.set_title(f'Grid {grid_size}×{grid_size} - Observer at {location}\n'
                        f'FOV: {len(visible_cells)}/{total_cells} cells ({actual_coverage:.1f}%)',
                        fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.8, label='Observer'),
                Patch(facecolor='yellow', alpha=0.5, edgecolor='orange', label='FOV Coverage')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            plot_idx += 1
    
    plt.savefig('fov_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved: fov_coverage_analysis.png")
    plt.show()

def analyze_fov_statistics():
    """Print detailed FOV statistics."""
    
    view_size = 5
    grid_sizes = [10, 12, 15]
    
    print("="*70)
    print("📊 FIELD OF VIEW (FOV) ANALYSIS")
    print("="*70)
    print(f"\n🔍 Observer FOV Size: {view_size}×{view_size} = {view_size**2} cells")
    print(f"   (Configured in goal_prediction.py: agent_view_size=[5, 5])")
    
    print("\n" + "-"*70)
    print("📐 FOV Coverage by Grid Size:")
    print("-"*70)
    
    for grid_size in grid_sizes:
        coverage, fov_cells, total_cells = calculate_fov_coverage(grid_size, view_size)
        print(f"\nGrid {grid_size}×{grid_size}:")
        print(f"  • Total cells: {total_cells}")
        print(f"  • FOV cells: {fov_cells}")
        print(f"  • Maximum theoretical coverage: {coverage:.1f}%")
        print(f"  • Grid-to-FOV ratio: 1:{total_cells/fov_cells:.1f}")
    
    print("\n" + "-"*70)
    print("💡 WHY IS IN-VIEW RATE HIGH?")
    print("-"*70)
    
    print("\n1. LARGE FOV RELATIVE TO GRID SIZE:")
    print("   • FOV is 5×5 = 25 cells")
    for grid_size in grid_sizes:
        coverage, _, _ = calculate_fov_coverage(grid_size, view_size)
        print(f"   • On {grid_size}×{grid_size} grid: {coverage:.1f}% max coverage")
    
    print("\n2. OBSERVER ACTIVELY PURSUES TARGET:")
    print("   • Observer uses greedy action selection (see observer.py greedy())")
    print("   • Observer moves toward most likely target position")
    print("   • This pursuit behavior keeps target in view more often")
    
    print("\n3. SMALL GRIDS + GOAL-DIRECTED MOVEMENT:")
    print("   • Target moves toward goal (predictable trajectory)")
    print("   • Observer predicts target location")
    print("   • On 10×10 grid, FOV covers 25% of space")
    print("   • High probability of intersection between:")
    print("     - Target's path to goal")
    print("     - Observer's pursuit path")
    
    print("\n4. CONTINUOUS OBSERVATION:")
    print("   • Both agents move simultaneously each step")
    print("   • Observer adjusts position to maintain visibility")
    print("   • FOV is forward-facing 5×5 cone")
    print("   • Coverage area moves with observer")
    
    print("\n" + "-"*70)
    print("📈 EXPECTED IN-VIEW RATES:")
    print("-"*70)
    
    print("\nBased on:")
    print("  • Active pursuit (observer moves toward target)")
    print("  • Large FOV (25 cells)")
    print("  • Small grids (10-15 cells)")
    print("  • Goal-directed behavior (predictable paths)")
    
    print("\nExpected visibility:")
    for grid_size in grid_sizes:
        coverage, _, _ = calculate_fov_coverage(grid_size, view_size)
        # Rough estimate: with active pursuit, expect 50-80% of theoretical max
        expected_low = coverage * 0.4
        expected_high = coverage * 0.7
        print(f"  • Grid {grid_size}×{grid_size}: {expected_low:.1f}% - {expected_high:.1f}%")
    
    print("\n" + "-"*70)
    print("🔧 TO REDUCE IN-VIEW RATE:")
    print("-"*70)
    
    print("\n1. Reduce FOV size:")
    print("   • Change agent_view_size from [5, 5] to [3, 3] in goal_prediction.py")
    print(f"   • This would reduce FOV from 25 to 9 cells ({(9/25)*100:.0f}% reduction)")
    
    print("\n2. Increase grid size:")
    print("   • Use larger grids (20×20, 25×25)")
    print("   • FOV coverage: 25/400 = 6.25% on 20×20 grid")
    
    print("\n3. Change observer behavior:")
    print("   • Use stationary observer instead of active pursuit")
    print("   • Use random movement instead of greedy pursuit")
    print("   • Set observer to patrol fixed positions")
    
    print("\n4. Add occlusions:")
    print("   • Add more walls (see_through_walls=False by default)")
    print("   • Walls block FOV and reduce visibility")
    
    print("\n" + "="*70)

def main():
    """Main function."""
    print("\n" + "="*70)
    print("🔍 FOV Coverage Analysis")
    print("="*70)
    
    # Print statistics
    analyze_fov_statistics()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    try:
        visualize_fov_scenarios()
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
        print("   (Statistics above are still valid)")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

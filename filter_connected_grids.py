"""
Filter CSV file to keep only instances with fully connected base grids.

This script:
1. Loads a CSV file containing grid instances
2. Parses the base_grid column (string representation of 2D arrays)
3. Checks connectivity using BFS from each free cell
4. Filters out instances with disconnected grids
5. Saves the filtered results to a new CSV file
"""

import pandas as pd
import ast
import numpy as np
from collections import deque
import argparse
import os

def parse_grid_string(grid_str):
    """Parse the string representation of a grid into a 2D numpy array."""
    try:
        # Use ast.literal_eval to safely parse the string representation
        grid_list = ast.literal_eval(grid_str)
        return np.array(grid_list)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing grid string: {e}")
        return None

def is_fully_connected(grid):
    """
    Check if all free cells (0s) in the grid are connected using BFS.
    
    Args:
        grid: 2D numpy array where 1 = wall, 0 = free space
        
    Returns:
        bool: True if all free cells are connected, False otherwise
    """
    if grid is None:
        return False
    
    rows, cols = grid.shape
    
    # Find all free cells
    free_cells = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                free_cells.append((i, j))
    
    if len(free_cells) == 0:
        return True  # No free cells, technically connected
    
    if len(free_cells) == 1:
        return True  # Only one free cell, connected by definition
    
    # BFS from the first free cell
    visited = set()
    queue = deque([free_cells[0]])
    visited.add(free_cells[0])
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        curr_row, curr_col = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = curr_row + dr, curr_col + dc
            
            # Check bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Check if it's a free cell and not visited
                if grid[new_row, new_col] == 0 and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
    
    # Check if all free cells were visited
    return len(visited) == len(free_cells)

def filter_connected_grids(input_csv, output_csv=None):
    """
    Filter CSV file to keep only instances with fully connected base grids.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional, will generate default name)
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    print(f"Initial number of instances: {initial_count}")
    
    # Track connectivity status
    connected_mask = []
    disconnected_count = 0
    parse_error_count = 0
    
    print("Checking grid connectivity...")
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{initial_count} instances...")
        
        grid_str = row['base_grid']
        grid = parse_grid_string(grid_str)
        
        if grid is None:
            connected_mask.append(False)
            parse_error_count += 1
            continue
        
        is_connected = is_fully_connected(grid)
        connected_mask.append(is_connected)
        
        if not is_connected:
            disconnected_count += 1
    
    print(f"Finished checking connectivity.")
    print(f"Disconnected grids found: {disconnected_count}")
    print(f"Parse errors: {parse_error_count}")
    
    # Filter the dataframe
    df_filtered = df[connected_mask].copy()
    filtered_count = len(df_filtered)
    
    print(f"\nFiltering results:")
    print(f"  Original instances: {initial_count}")
    print(f"  Connected instances: {filtered_count}")
    print(f"  Removed instances: {initial_count - filtered_count}")
    print(f"  Retention rate: {filtered_count/initial_count*100:.2f}%")
    
    # Generate output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_connected_only.csv"
    
    # Save filtered results
    df_filtered.to_csv(output_csv, index=False)
    print(f"\nFiltered data saved to: {output_csv}")
    
    return df_filtered, output_csv

def main():
    parser = argparse.ArgumentParser(description='Filter CSV file to keep only instances with fully connected base grids')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--test-sample', '-t', type=int, help='Test connectivity check on first N instances only')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    if args.test_sample:
        print(f"Testing connectivity check on first {args.test_sample} instances...")
        df = pd.read_csv(args.input)
        df_sample = df.head(args.test_sample)
        
        for idx, row in df_sample.iterrows():
            grid_str = row['base_grid']
            grid = parse_grid_string(grid_str)
            if grid is not None:
                is_connected = is_fully_connected(grid)
                print(f"Instance {idx}: Size {grid.shape}, Connected: {is_connected}")
                if not is_connected:
                    print(f"  Grid:\n{grid}")
        return
    
    # Filter the full dataset
    filter_connected_grids(args.input, args.output)

if __name__ == "__main__":
    main()
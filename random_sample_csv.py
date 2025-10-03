"""
Randomly sample a specified number of instances from a CSV file.

This script loads a CSV file and randomly selects N instances,
then saves them to a new file.
"""

import pandas as pd
import argparse
import os
import numpy as np

def random_sample_csv(input_csv, n_samples, output_csv=None, random_seed=42):
    """
    Randomly sample N instances from a CSV file.
    
    Args:
        input_csv: Path to input CSV file
        n_samples: Number of instances to sample
        output_csv: Path to output CSV file (optional)
        random_seed: Random seed for reproducibility
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    print(f"Total instances available: {initial_count}")
    
    if n_samples > initial_count:
        print(f"Warning: Requested {n_samples} samples but only {initial_count} available.")
        print(f"Using all {initial_count} instances.")
        n_samples = initial_count
        df_sampled = df.copy()
    else:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Randomly sample instances
        print(f"Randomly sampling {n_samples} instances (seed={random_seed})...")
        df_sampled = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    # Generate output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_sample_{n_samples}.csv"
    
    # Save sampled results
    df_sampled.to_csv(output_csv, index=False)
    
    print(f"\nSampling results:")
    print(f"  Original instances: {initial_count}")
    print(f"  Sampled instances: {len(df_sampled)}")
    print(f"  Sampling rate: {len(df_sampled)/initial_count*100:.2f}%")
    print(f"  Output saved to: {output_csv}")
    
    return df_sampled, output_csv

def main():
    parser = argparse.ArgumentParser(description='Randomly sample N instances from a CSV file')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--samples', '-n', type=int, default=1500, help='Number of samples to select (default: 1500)')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Sample the dataset
    random_sample_csv(args.input, args.samples, args.output, args.seed)

if __name__ == "__main__":
    main()
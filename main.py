"""
Relative Clause Extraction Main Script

This script provides the main entry point for relative clause extraction
from text files. It initializes the RelativeClause extractor, processes
all text files in the input directory, and generates comprehensive results.

Usage:
    python main.py

The script will:
1. Process all .txt files in the input_texts/ directory
2. Extract relative clauses using dual parsing (dependency + constituency)
3. Generate results in spaCy_RCs.csv
4. Display summary statistics
"""

from relative_clause_extractor import RelativeClause
import os
import pandas as pd

def main():
    """
    Main function for relative clause extraction.
    
    Initializes the extractor, processes input files, and generates
    comprehensive results with statistics.
    """
    # Define input and output folders
    input_texts = "input_texts"
    output_folder = "result"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize relative clause extractor
    print("Initializing relative clause extractor...")
    rc_extractor = RelativeClause(input_texts=input_texts, output_folder=output_folder)

    # Extract relative clauses
    print("Extracting relative clauses from input files...")
    df, message = rc_extractor.extract_relative_clauses()

    # Print results
    print(message)
    
    if not df.empty:
        print("\nSample of extracted relative clauses:")
        print(df.head())
        
        # Print some statistics
        print("\nStatistics:")
        print(f"Total number of relative clauses: {len(df)}")
        print("\nDistribution of RC types:")
        print(df['rc_type'].value_counts())
        print("\nDistribution of relativizers:")
        print(df['relativizer'].value_counts())
    else:
        print("\nNo relative clauses were found in the input files.")

if __name__ == "__main__":
    main()


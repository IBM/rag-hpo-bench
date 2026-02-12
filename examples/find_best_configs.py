"""
Example script to read RAG configurations summary from Hugging Face dataset
and save the best configuration for each dataset and split to CSV files.

This script demonstrates how to:
1. Download only the rag_configurations_summary.csv file from the RAG-HPO-bench dataset
2. Find the best configuration for each dataset/split combination based on different metrics
3. Save the results to CSV files with all hyperparameters and metric values
4. Display summary statistics

Dataset: https://huggingface.co/datasets/ibm-research/rag-hpo-bench
"""

import pandas as pd
from huggingface_hub import hf_hub_download


def load_rag_configurations_summary():
    """
    Load the RAG configurations summary from Hugging Face.
    Downloads only the rag_configurations_summary.csv file.
    
    Returns:
        pd.DataFrame: DataFrame containing all RAG configurations and their results
    """
    print("Downloading rag_configurations_summary.csv from Hugging Face...")
    
    # Download only the config summary CSV file from the dataset
    csv_path = hf_hub_download(
        repo_id="ibm-research/rag-hpo-bench",
        filename="rag_configurations_summary.csv",
        repo_type="dataset"
    )
    
    print(f"Downloaded to: {csv_path}")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} RAG configurations")
    print(f"Columns: {list(df.columns)}")
    
    return df


def find_best_configs(df, metric="LLMaaJ-AC", output_csv="best_configs.csv"):
    """
    Find the best configuration for each dataset and split based on a metric.
    Creates a DataFrame with the best configs and saves it to a CSV file.
    
    Args:
        df: DataFrame with RAG configurations
        metric: Metric to optimize (default: "LLMaaJ-AC")
        output_csv: Path to save the CSV file (default: "best_configs.csv")
        
    Returns:
        pd.DataFrame: Best configurations for each dataset/split
    """
    print(f"\nFinding best configurations based on metric: {metric}")
    
    # Group by Dataset and Split, then find the row with max metric value
    best_configs = df.loc[df.groupby(['Dataset', 'Split'])[metric].idxmax()]
    
    # Select columns that remain with the same name
    result_df = best_configs[[
        'Dataset', 'Split', 'Configuration ID', 'Chunk Size', 'Chunk Overlap',
        'Embedding Model', 'Top-K', 'Generative Model', 'LLMaaJ-AC',
        'Lexical-AC', 'Lexical-FF'
    ]].copy()
    
    # Add the objective metric column
    result_df['Objective Metric'] = metric
    
    # Reset index to make it cleaner
    result_df = result_df.reset_index(drop=True)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Best configurations saved to: {output_csv}")
    print(f"Total best configurations: {len(result_df)}")
    
    return result_df


def main():    
    # Load the dataset
    df = load_rag_configurations_summary()
    
    # Find and save best configurations for LLMaaJ-AC metric
    best_configs_llmaaj = find_best_configs(df, metric="LLMaaJ-AC", output_csv="best_configs_llmaaj_ac.csv")
    print(f"\nPreview of best configurations (LLMaaJ-AC):")
    print(best_configs_llmaaj.to_string())
    
    # Find and save best configurations for Lexical-AC metric
    best_configs_lexical = find_best_configs(df, metric="Lexical-AC", output_csv="best_configs_lexical_ac.csv")
    print(f"\nPreview of best configurations (Lexical-AC):")
    print(best_configs_lexical.to_string())
    
    # Find and save best configurations for Lexical-FF metric
    best_configs_ff = find_best_configs(df, metric="Lexical-FF", output_csv="best_configs_lexical_ff.csv")
    print(f"\nPreview of best configurations (Lexical-FF):")
    print(best_configs_ff.to_string())
    


if __name__ == "__main__":
    main()
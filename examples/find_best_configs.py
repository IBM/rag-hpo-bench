"""
Example script to read RAG configurations summary from Hugging Face dataset
and print the best configuration for each dataset and split.

This script demonstrates how to:
1. Load the rag_configurations_summary.csv from the RAG-HPO-bench dataset
2. Find the best configuration for each dataset/split combination
3. Display the results in a readable format

Dataset: https://huggingface.co/datasets/ibm-research/rag-hpo-bench
"""

import pandas as pd
from datasets import load_dataset


def load_rag_configurations():
    """
    Load the RAG configurations summary from Hugging Face.
    
    Returns:
        pd.DataFrame: DataFrame containing all RAG configurations and their results
    """
    print("Loading RAG-HPO-bench dataset from Hugging Face...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("ibm-research/rag-hpo-bench", split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    print(f"Loaded {len(df)} RAG configurations")
    print(f"Columns: {list(df.columns)}")
    
    return df


def find_best_configs(df, metric="LLMaaJ-AC"):
    """
    Find the best configuration for each dataset and split based on a metric.
    
    Args:
        df: DataFrame with RAG configurations
        metric: Metric to optimize (default: "LLMaaJ-AC")
        
    Returns:
        pd.DataFrame: Best configurations for each dataset/split
    """
    print(f"\nFinding best configurations based on metric: {metric}")
    
    # Group by Dataset and Split, then find the row with max metric value
    best_configs = df.loc[df.groupby(['Dataset', 'Split'])[metric].idxmax()]
    
    return best_configs


def print_best_configs(best_configs, metric="LLMaaJ-AC"):
    """
    Print the best configurations in a readable format.
    
    Args:
        best_configs: DataFrame with best configurations
        metric: Metric that was optimized
    """
    print(f"\n{'='*80}")
    print(f"BEST RAG CONFIGURATIONS (optimized for {metric})")
    print(f"{'='*80}\n")
    
    for _, row in best_configs.iterrows():
        print(f"Dataset: {row['Dataset']}")
        print(f"Split: {row['Split']}")
        print(f"Configuration ID: {row['Configuration ID']}")
        print(f"\nHyperparameters:")
        print(f"  - Chunk Size: {row['Chunk Size']}")
        print(f"  - Chunk Overlap: {row['Chunk Overlap']:.2f}")
        print(f"  - Embedding Model: {row['Embedding Model']}")
        print(f"  - Top-K: {row['Top-K']}")
        print(f"  - Generative Model: {row['Generative Model']}")
        print(f"\nMetrics:")
        print(f"  - {metric}: {row[metric]:.4f}")
        print(f"  - Lexical-AC: {row['Lexical-AC']:.4f}")
        print(f"  - Lexical-FF: {row['Lexical-FF']:.4f}")
        print(f"  - Context Correctness: {row['Context Correctness']:.4f}")
        print(f"\n{'-'*80}\n")


def compare_metrics(df):
    """
    Compare best configurations across different metrics.
    
    Args:
        df: DataFrame with RAG configurations
    """
    metrics = ["LLMaaJ-AC", "Lexical-AC", "Lexical-FF"]
    
    print(f"\n{'='*80}")
    print("COMPARISON: Best Configs Across Different Metrics")
    print(f"{'='*80}\n")
    
    for dataset_split in df.groupby(['Dataset', 'Split']).groups.keys():
        dataset, split = dataset_split
        subset = df[(df['Dataset'] == dataset) & (df['Split'] == split)]
        
        print(f"Dataset: {dataset}, Split: {split}")
        print(f"{'Metric':<20} {'Config ID':<12} {'Score':<10} {'Chunk Size':<12} {'Top-K':<8}")
        print("-" * 70)
        
        for metric in metrics:
            best_idx = subset[metric].idxmax()
            best_row = subset.loc[best_idx]
            print(f"{metric:<20} {best_row['Configuration ID']:<12} "
                  f"{best_row[metric]:<10.4f} {best_row['Chunk Size']:<12} "
                  f"{best_row['Top-K']:<8}")
        
        print()


def main():
    """Main function to demonstrate usage."""
    
    # Load the dataset
    df = load_rag_configurations()
    
    # Find and print best configurations for LLMaaJ-AC metric
    best_configs_llmaaj = find_best_configs(df, metric="LLMaaJ-AC")
    print_best_configs(best_configs_llmaaj, metric="LLMaaJ-AC")
    
    # Find and print best configurations for Lexical-AC metric
    best_configs_lexical = find_best_configs(df, metric="Lexical-AC")
    print_best_configs(best_configs_lexical, metric="Lexical-AC")
    
    # Compare metrics
    compare_metrics(df)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total configurations: {len(df)}")
    print(f"Unique datasets: {df['Dataset'].nunique()}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"\nConfigurations per dataset/split:")
    print(df.groupby(['Dataset', 'Split']).size())


if __name__ == "__main__":
    main()
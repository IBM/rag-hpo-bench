"""
Utility functions for loading RAG configurations from the HuggingFace dataset.

Dataset: https://huggingface.co/datasets/ibm-research/rag-hpo-bench
"""

import logging
from functools import lru_cache

import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_rag_configurations_summary() -> pd.DataFrame:
    """
    Load the RAG configurations summary from Hugging Face.
    Downloads only the rag_configurations_summary.csv file.
    
    This function is cached to avoid re-downloading the file on subsequent calls.
    
    Returns:
        pd.DataFrame: DataFrame containing all RAG configurations and their results
        
    Raises:
        Exception: If download or loading fails
    """
    logger.info("Downloading rag_configurations_summary.csv from Hugging Face...")
    
    try:
        # Download only the config summary CSV file from the dataset
        csv_path = hf_hub_download(
            repo_id="ibm-research/rag-hpo-bench",
            filename="rag_configurations_summary.csv",
            repo_type="dataset"
        )
        
        logger.info(f"Downloaded to: {csv_path}")
        
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Convert integer columns to int type to avoid float/int comparison issues
        int_columns = ["Chunk Size", "Top-K"]
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        logger.info(f"Loaded {len(df)} RAG configurations")
        logger.debug(f"Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load RAG configurations summary: {e}")
        raise
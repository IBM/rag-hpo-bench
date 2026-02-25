"""
Main entry point for running multiple HPO experiments.

This script runs multiple HPO experiments with different combinations of
datasets, algorithms, and optimization metrics.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from rag_hpo_bench.data_models import DatasetID, DatasetName
from rag_hpo_bench.hpo import AlgorithmConfig, TuneAndTestDataset, ExperimentsRunner
from rag_hpo_bench.hpo.search_space import SearchSpace, SearchSpaceParameter
from rag_hpo_bench.utils.logging_utils import init_logger

logger = logging.getLogger(__name__)


def create_search_space() -> SearchSpace:
    """
    Create the search space for RAG experiments.
    
    Returns:
        SearchSpace with configured parameters
    """
    
    parameters = [
        # Indexing parameters
        SearchSpaceParameter(
            path=["data_pipeline", "params", "indexing", "chunk_size"],
            values=[256, 384, 512],
        ),
        SearchSpaceParameter(
            path=["data_pipeline", "params", "indexing", "chunk_overlap"],
            values=[0, 0.25],
        ),
        SearchSpaceParameter(
            path=["data_pipeline", "params", "indexing", "vector_space", "embedding_model"],
            values=["local/e5_large", "local/bge_large_1_5", "local/granite_embedding_125m"],
        ),
        # Retrieval parameters
        SearchSpaceParameter(
            path=["inference_pipeline", "params", "retrieval", "top_k"],
            values=[3, 5, 10],
        ),
        # Generation parameters
        SearchSpaceParameter(
            path=["inference_pipeline", "params", "generation", "generative_model"],
            values=[
                "local/granite_3_1_8b_instruct",
                "local/llama_3_1_8b_instruct",
                "local/mistral_nemo_instruct",
            ],
        ),
    ]
    return SearchSpace(parameters=parameters)


def create_algorithm_configs() -> list[AlgorithmConfig]:
    algorithm_configs = [
        AlgorithmConfig(
            algorithm_type="grid",
            # Grid search is deterministic and doesn't use num_seeds
        ),
        AlgorithmConfig(
            algorithm_type="random",
            num_seeds=10,
            additional_params={"n_trials": 10},
        ),
        AlgorithmConfig(
            algorithm_type="greedy_m",
            num_seeds=10,
            additional_params={"n_trials": 10},
        ),
    ]
    
    return algorithm_configs


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run multiple HPO experiments with different configurations"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (default: run all experiments)",
    )
    return parser.parse_args()


def main():
    """Main entry point for running HPO experiments."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging
    init_logger(level=logging.INFO)
    
    # Define dataset pairs (tune and test)
    # Note: Split names must match the HuggingFace dataset: "Dev" for tuning, "Test" for testing
    dataset_pairs = [
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=DatasetName.ClapNQ,
                split="Dev",
            ),
            test=DatasetID(
                dataset_name=DatasetName.ClapNQ,
                split="Test",
            ),
        ),
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=DatasetName.AIArxiv,
                split="Dev",
            ),
            test=DatasetID(
                dataset_name=DatasetName.AIArxiv,
                split="Test",
            ),
        ),
    ]
    
    # Create algorithm configurations
    algorithm_configs = create_algorithm_configs()
    
    # Create search space
    search_space = create_search_space()
    
    # Define optimization metrics (available in rag_configurations_summary.csv)
    # These metrics are used to evaluate and compare RAG configurations:
    # - LLMaaJ-AC: LLM as a Judge - Answer Correctness
    # - Lexical-AC: Lexical Answer Correctness
    # - Lexical-FF: Lexical Faithfulness Score
    # - context_correctness: Context Correctness
    optimization_metrics = [
        "LLMaaJ-AC",
        "Lexical-AC",
        "Lexical-FF",
    ]
    
    # Create the ExperimentsRunner
    runner = ExperimentsRunner(
        search_space=search_space,
        dataset_pairs=dataset_pairs,
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=Path("./experiments_output"),
        skip_existing_tunes=True,
        skip_existing_test_results=True,
        clean_output_dir=False,
        max_experiments=args.max_experiments,
    )
    
    # Log configuration
    total_experiments = len(runner.hpo_experiments)
    logger.info(f"Total experiments created: {total_experiments}")
    if args.max_experiments is not None:
        experiments_to_run = min(args.max_experiments, total_experiments)
        logger.info(f"Will run: {experiments_to_run} experiments (limited by --max-experiments)")
    else:
        logger.info(f"Will run: all {total_experiments} experiments")
    logger.info(f"Dataset pairs: {len(dataset_pairs)}")
    logger.info(f"Algorithm configs: {len(algorithm_configs)}")
    logger.info(f"Optimization metrics: {len(optimization_metrics)}")
    
    # Run all experiments
    logger.info("Starting experiments...")
    results = runner.run()
    
    # Report results
    successful_results = [r for r in results if r is not None]
    logger.info(f"Completed: {len(successful_results)}/{len(results)} experiments successful")
    
    return results


if __name__ == "__main__":
    main()
"""
Main entry point for running multiple HPO experiments.

This script runs multiple HPO experiments with different combinations of
datasets, algorithms, and optimization metrics.
"""

import argparse
import logging
from pathlib import Path

from rag_hpo_bench.data_models import DatasetID, DatasetName
from rag_hpo_bench.hpo import AlgorithmConfig, ExperimentsRunner, TuneAndTestDataset
from rag_hpo_bench.hpo.search_space import SearchSpace, SearchSpaceParameter
from rag_hpo_bench.utils.analyze_test_results import run_analysis
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
            values=[
                "BAAI/bge-large-en-v1.5",
                "ibm/slate-125m-english-rtrvr",
                "intfloat/multilingual-e5-large",
            ],
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
                "ibm-granite/granite-3.1-8b-instruct",
                "meta-llama/llama-3-1-8b-instruct",
                "mistral_nemo_instruct",
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
            additional_params={"max_iterations": 10},
        ),
        AlgorithmConfig(
            algorithm_type="greedy_m",
            num_seeds=10,
            additional_params={"max_iterations": 10},
        ),
        AlgorithmConfig(
            algorithm_type="greedy_r",
            num_seeds=10,
            additional_params={"max_iterations": 10},
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
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Clean output directory before running experiments",
    )
    return parser.parse_args()


def run_experiments(
    dataset_pairs: list[TuneAndTestDataset],
    algorithm_configs: list[AlgorithmConfig],
    optimization_metrics: list[str],
    output_path: Path,
    max_experiments: int | None = None,
    skip_existing_tunes: bool = False,
    skip_existing_test_results: bool = False,
    clean_output_dir: bool = False,
):
    """
    Run HPO experiments with the given configuration.

    Args:
        dataset_pairs: List of tune/test dataset pairs
        algorithm_configs: List of algorithm configurations
        optimization_metrics: List of optimization metric IDs
        output_path: Base path for experiment outputs
        max_experiments: Maximum number of experiments to run (None for unlimited)
        skip_existing_tunes: Whether to skip existing tune results
        skip_existing_test_results: Whether to skip existing test results
        clean_output_dir: Whether to clean output directory before running

    Returns:
        List of experiment results
    """
    # Create search space
    search_space = create_search_space()

    # Create the ExperimentsRunner
    runner = ExperimentsRunner(
        search_space=search_space,
        dataset_pairs=dataset_pairs,
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=output_path,
        skip_existing_tunes=skip_existing_tunes,
        skip_existing_test_results=skip_existing_test_results,
        clean_output_dir=clean_output_dir,
        max_experiments=max_experiments,
    )

    # Log configuration
    total_experiments = len(runner.hpo_experiments)
    logger.info(f"Total experiments created: {total_experiments}")
    if max_experiments is not None:
        experiments_to_run = min(max_experiments, total_experiments)
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
    logger.info(f"Completed: {len(successful_results)}/{len(results)} experiments successfully.")

    return results


def run_hpo(output_path: Path, max_experiments: int | None = None, clean_output: bool = False):
    """Main entry point for running HPO experiments.

    Args:
        output_path: Base path for experiment outputs
        max_experiments: Maximum number of experiments to run (default: run all experiments)
        clean_output: Whether to clean output directory before running (default: False)
    """

    # Define dataset pairs (tune and test)
    # Note: Split names must match the HuggingFace dataset: "Dev" for tuning, "Test" for testing
    # Create pairs for all available datasets
    dataset_pairs = [
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=dataset_name,
                split="Dev",
            ),
            test=DatasetID(
                dataset_name=dataset_name,
                split="Test",
            ),
        )
        for dataset_name in DatasetName
    ]

    # Create algorithm configurations
    algorithm_configs = create_algorithm_configs()

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

    return run_experiments(
        dataset_pairs=dataset_pairs,
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=output_path,
        max_experiments=max_experiments,
        skip_existing_tunes=False,
        skip_existing_test_results=False,
        clean_output_dir=clean_output,
    )


def run_grid_search_on_test_sets(output_path: Path, clean_output: bool = False):
    """
    Run grid search directly on test datasets without tuning.

    This function evaluates all configurations in the search space on test datasets,
    useful for analyzing the full search space performance without HPO.

    Args:
        output_path: Base path for experiment outputs
        clean_output: Whether to clean output directory before running (default: False)
    """
    # Configure logging
    init_logger(level=logging.INFO)

    # Define test datasets only (no tuning)
    test_datasets = [
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=dataset_name,
                split="Test",  # Run the grid search on the test set
            ),
            test=None,  # No separate test dataset
        )
        for dataset_name in DatasetName
    ]

    # Create grid search algorithm configuration only
    algorithm_configs = [
        AlgorithmConfig(
            algorithm_type="grid",
            # Grid search is deterministic and doesn't use num_seeds
        ),
    ]

    # No optimization metrics needed for grid search on test sets
    # Grid search evaluates all configurations without optimization
    optimization_metrics = []

    # Run experiments using the utility function
    return run_experiments(
        dataset_pairs=test_datasets,
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=output_path,
        max_experiments=None,
        skip_existing_tunes=False,
        skip_existing_test_results=False,
        clean_output_dir=clean_output,
    )


if __name__ == "__main__":
    # Configure logging
    init_logger(level=logging.INFO)

    # Parse command-line arguments
    args = parse_args()

    # Define output path once
    output_path = Path("./experiments_output")

    # Run HPO experiments
    run_hpo(
        output_path=output_path,
        max_experiments=args.max_experiments,
        clean_output=args.clean_output,
    )

    # Run grid search on test sets
    run_grid_search_on_test_sets(output_path=output_path, clean_output=args.clean_output)

    # Analyze test results
    logger.info("Analyzing test results...")
    run_analysis(base_results_path=output_path)
    logger.info("Test results analysis complete")

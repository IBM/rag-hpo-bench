"""
Main entry point for running multiple HPO experiments.

This script runs multiple HPO experiments with different combinations of
datasets, algorithms, and optimization metrics.
"""

import logging
from pathlib import Path

from rag_hpo_bench.data_models import DatasetID, DatasetName
from rag_hpo_bench.hpo import AlgorithmConfig, TuneAndTestDataset, ExperimentsRunner
from rag_hpo_bench.hpo.search_space import SearchSpace, SearchSpaceParameter

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


def main():
    """Main entry point for running HPO experiments."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Define dataset pairs (tune and test)
    dataset_pairs = [
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=DatasetName.ClapNQ,
                split="train",
            ),
            test=DatasetID(
                dataset_name=DatasetName.ClapNQ,
                split="test",
            ),
        ),
        TuneAndTestDataset(
            tune=DatasetID(
                dataset_name=DatasetName.AIArxiv,
                split="train",
            ),
            test=DatasetID(
                dataset_name=DatasetName.AIArxiv,
                split="test",
            ),
        ),
    ]
    
    # Define algorithm configurations
    algorithm_configs = [
        AlgorithmConfig(
            algorithm_type="grid",
        ),
        AlgorithmConfig(
            algorithm_type="random",
            num_seeds=3,
            additional_params={"n_trials": 20},
        ),
        AlgorithmConfig(
            algorithm_type="bayesian",
            num_seeds=5,
            additional_params={"n_trials": 50},
        ),
    ]
    
    # Define optimization metrics
    optimization_metrics = [
        "answer_correctness",
        "answer_relevance",
        "faithfulness",
    ]
    
    # Create search space
    search_space = create_search_space()
    
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
    )
    
    # Log configuration
    logger.info(f"Total experiments to run: {len(runner.hpo_experiments)}")
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
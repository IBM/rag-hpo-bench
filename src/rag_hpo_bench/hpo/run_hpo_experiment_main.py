#!/usr/bin/env python3
"""
Main script for running a single HPO experiment.
"""
import logging
from pathlib import Path

from rag_hpo_bench.data_models.dataset_id import DatasetID
from rag_hpo_bench.data_models.dataset_names import DatasetName
from rag_hpo_bench.hpo.hpo_experiment import HpoExperiment
from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.search_space import SearchSpace, SearchSpaceParameter


def setup_logging():
    """Configure logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_search_space() -> SearchSpace:
    """
    Create a search space with hardcoded parameters.
    
    This example includes:
    - Chunking parameters (size and overlap)
    - Embedding model
    - Retrieval parameters (top-k)
    - Generation model
    """
    return SearchSpace(
        parameters=[
            # Indexing parameters
            SearchSpaceParameter(
                path=["indexing", "chunking", "size"],
                values=[512, 1024]
            ),
            SearchSpaceParameter(
                path=["indexing", "chunking", "overlap"],
                values=[50, 100]
            ),
            SearchSpaceParameter(
                path=["indexing", "embedding", "model"],
                values=["text-embedding-ada-002"]
            ),
            # Inference parameters
            SearchSpaceParameter(
                path=["inference", "retrieval", "top-k"],
                values=[5, 10]
            ),
            SearchSpaceParameter(
                path=["inference", "generation", "model"],
                values=["gpt-3.5-turbo"]
            ),
        ]
    )


def create_tune_dataset() -> DatasetID:
    """Create the tuning dataset configuration."""
    return DatasetID(
        dataset_name=DatasetName.AIArxiv,
        split="train"
    )


def create_test_dataset() -> DatasetID:
    """Create the test dataset configuration."""
    return DatasetID(
        dataset_name=DatasetName.AIArxiv,
        split="test"
    )


def create_algorithm_params() -> dict:
    """
    Create algorithm parameters for HPO.
    
    Available algorithm types:
    - "grid": Grid search (exhaustive)
    - "random": Random search
    - "bayesian": Bayesian optimization
    
    For random/bayesian, you can also specify:
    - "num_seeds": Number of random seeds to run (not compatible with grid)
    - "max_iterations": Maximum number of iterations
    """
    return {
        "algorithm_type": "grid",
        # Uncomment for random/bayesian search:
        # "num_seeds": 3,
        # "max_iterations": 20,
    }


def main():
    """Main entry point for running the HPO experiment."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting HPO Experiment")
    
    # Create search space
    search_space = create_search_space()
    logger.info(f"Created search space with {len(search_space.parameters)} parameters")
    
    # Create datasets
    tune_dataset = create_tune_dataset()
    test_dataset = create_test_dataset()
    logger.info(f"Tune dataset: {tune_dataset.as_string()}")
    logger.info(f"Test dataset: {test_dataset.as_string()}")
    
    # Create algorithm parameters
    algorithm_params = create_algorithm_params()
    logger.info(f"Algorithm: {algorithm_params['algorithm_type']}")
    
    # Define output path
    output_path = Path("./hpo_experiment_output")
    logger.info(f"Output path: {output_path.absolute()}")
    
    # Define optimization metric
    optimization_metric_id = "answer_correctness"
    logger.info(f"Optimization metric: {optimization_metric_id}")
    
    # Create and run the experiment
    experiment = HpoExperiment(
        search_space=search_space,
        tune_dataset=tune_dataset,
        test_dataset=test_dataset,
        algorithm_params=algorithm_params,
        optimization_metric_id=optimization_metric_id,
        output_path=output_path,
        skip_existing_tunes=False,
        skip_existing_test_results=False,
        clean_output_dir=False
    )
    
    logger.info("Running HPO experiment...")
    results = experiment.run()
    
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to: {experiment.output_path}")
    
    return results


if __name__ == "__main__":
    main()
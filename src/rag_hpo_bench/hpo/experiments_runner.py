"""
ExperimentsRunner for running multiple HpoExperiment experiments.

This module provides functionality to run multiple HPO experiments with different
combinations of datasets, algorithms, and optimization metrics.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_hpo_bench.data_models import DatasetID
from rag_hpo_bench.hpo.hpo_experiment import HpoExperiment
from rag_hpo_bench.hpo.search_space import SearchSpace

logger = logging.getLogger(__name__)


class TuneAndTestDataset(BaseModel):
    """
    Represents a pair of tune and test datasets for HPO experiments.
    
    Attributes:
        tune: Dataset to use for tuning/training
        test: Optional dataset to use for testing (can be None)
    """

    tune: DatasetID
    test: DatasetID | None = None
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __str__(self) -> str:
        test_str = self.test.as_string() if self.test else "None"
        return f"TuneAndTestDataset(tune={self.tune.as_string()}, test={test_str})"


class AlgorithmConfig(BaseModel):
    """
    Configuration for a single HPO algorithm.
    
    Attributes:
        algorithm_type: Type of algorithm (e.g., "grid", "random", "bayesian")
        num_seeds: Optional number of random seeds to run
        additional_params: Any additional algorithm-specific parameters
    """

    algorithm_type: str
    num_seeds: int | None = None
    additional_params: dict[str, Any] = {}
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format expected by HpoExperiment.
        
        Returns:
            Dictionary with algorithm configuration
        """
        result = {
            "algorithm_type": self.algorithm_type,
            **self.additional_params,
        }
        if self.num_seeds is not None:
            result["num_seeds"] = self.num_seeds
        return result

    def __str__(self) -> str:
        seeds_str = f", seeds={self.num_seeds}" if self.num_seeds else ""
        return f"AlgorithmConfig(type={self.algorithm_type}{seeds_str})"


@dataclass
class ExperimentsRunner:
    """
    Runs multiple HpoExperiment experiments with different combinations of:
    - Multiple tune/test dataset pairs
    - Multiple algorithm configurations
    - Multiple optimization metrics
    - A single search space (shared across all experiments)
    
    The runner creates and executes all combinations of the provided parameters,
    allowing for comprehensive HPO benchmarking across different configurations.
    
    Attributes:
        search_space: Single search space to use for all experiments
        dataset_pairs: List of tune/test dataset pairs
        algorithm_configs: List of algorithm configurations
        optimization_metrics: List of optimization metric IDs
        output_path: Base path for experiment outputs
        skip_existing_tunes: Whether to skip existing tune results
        skip_existing_test_results: Whether to skip existing test results
        clean_output_dir: Whether to clean output directory before running
    """

    search_space: SearchSpace
    dataset_pairs: list[TuneAndTestDataset]
    algorithm_configs: list[AlgorithmConfig]
    optimization_metrics: list[str]
    output_path: Path
    skip_existing_tunes: bool = False
    skip_existing_test_results: bool = False
    clean_output_dir: bool = False

    def __post_init__(self):
        """Initialize and create all HPO experiments."""
        self.hpo_experiments: list[HpoExperiment] = []
        self._create_experiments()

    def _create_experiments(self):
        """
        Create HpoExperiment instances for all combinations of:
        - dataset pairs
        - algorithm configs
        - optimization metrics
        
        Each combination creates a separate HPO experiment that will be run independently.
        """
        total_experiments = (
            len(self.dataset_pairs)
            * len(self.algorithm_configs)
            * len(self.optimization_metrics)
        )

        logger.info(
            f"Creating {total_experiments} HPO experiments:\n"
            f"  - {len(self.dataset_pairs)} dataset pair(s)\n"
            f"  - {len(self.algorithm_configs)} algorithm config(s)\n"
            f"  - {len(self.optimization_metrics)} optimization metric(s)"
        )

        experiment_count = 0
        for dataset_pair in self.dataset_pairs:
            for algorithm_config in self.algorithm_configs:
                for optimization_metric in self.optimization_metrics:
                    experiment_count += 1
                    
                    # Create algorithm params dict
                    algorithm_params = algorithm_config.to_dict()

                    # Create the HPO experiment
                    hpo_experiment = HpoExperiment(
                        search_space=self.search_space,
                        tune_dataset=dataset_pair.tune,
                        test_dataset=dataset_pair.test,
                        algorithm_params=algorithm_params,
                        optimization_metric_id=optimization_metric,
                        output_path=self.output_path,
                        skip_existing_tunes=self.skip_existing_tunes,
                        skip_existing_test_results=self.skip_existing_test_results,
                        clean_output_dir=self.clean_output_dir,
                    )

                    self.hpo_experiments.append(hpo_experiment)

                    logger.debug(
                        f"Created experiment {experiment_count}/{total_experiments}: "
                        f"{dataset_pair}, {algorithm_config}, metric={optimization_metric}"
                    )

        logger.info(f"Successfully created {len(self.hpo_experiments)} HPO experiments")

    def run(self) -> list[Any]:
        """
        Run all HPO experiments sequentially.
        
        Each experiment is run independently, and failures in one experiment
        do not prevent subsequent experiments from running.

        Returns:
            List of results from all experiments (None for failed experiments)
        """
        logger.info(f"Starting execution of {len(self.hpo_experiments)} HPO experiments")

        all_results = []
        successful_count = 0
        failed_count = 0

        for idx, hpo_experiment in enumerate(self.hpo_experiments, 1):
            logger.info(
                f"\n{'='*80}\n"
                f"Running experiment {idx}/{len(self.hpo_experiments)}\n"
                f"  Tune dataset: {hpo_experiment.tune_dataset.as_string()}\n"
                f"  Test dataset: {hpo_experiment.test_dataset.as_string() if hpo_experiment.test_dataset else 'None'}\n"
                f"  Algorithm: {hpo_experiment.algorithm_params['algorithm_type']}\n"
                f"  Metric: {hpo_experiment.optimization_metric_id}\n"
                f"{'='*80}"
            )

            try:
                result = hpo_experiment.run()
                all_results.append(result)
                successful_count += 1
                logger.info(
                    f"✓ Experiment {idx}/{len(self.hpo_experiments)} completed successfully"
                )
            except Exception as e:
                logger.error(
                    f"✗ Experiment {idx}/{len(self.hpo_experiments)} failed with error: {e}",
                    exc_info=True,
                )
                # Continue with next experiment even if one fails
                all_results.append(None)
                failed_count += 1

        logger.info(
            f"\n{'='*80}\n"
            f"All experiments completed:\n"
            f"  Successful: {successful_count}/{len(self.hpo_experiments)}\n"
            f"  Failed: {failed_count}/{len(self.hpo_experiments)}\n"
            f"{'='*80}"
        )

        return all_results
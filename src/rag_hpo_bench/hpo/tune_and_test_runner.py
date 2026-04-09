import logging
import random
from dataclasses import dataclass
from pathlib import Path

from rag_hpo_bench.data_models import DatasetID
from rag_hpo_bench.hpo.hpo_algorithm import HpoAlgorithmType
from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.pattern_results import MultiplePatternResults
from rag_hpo_bench.hpo.search_space import PatternParameters
from rag_hpo_bench.hpo.tuner import Tuner
from rag_hpo_bench.hpo.test_results import TestResults

logger = logging.getLogger(__name__)


@dataclass
class TuneAndTestRunner:

    tuner: Tuner
    test_dataset: DatasetID | None
    skip_existing_test_results: bool = False

    """
    The seed used for initializing the random generator that creates the seeds used
    per each run of the tuning algorithms.
    """
    seed_of_seeds: int = 17

    """
    The number of seeds used by the tuning algorithms. Each algorithm runs this number of
    times, each with a different seed.
    None: disables seed-based running. In this case the tuning algorithm runs once without receiving a seed.
    """
    num_seeds: int | None = None

    def __post_init__(self):
        if self.num_seeds:
            random.seed(self.seed_of_seeds)
            self.seeds = [random.randint(1, 10000) for _ in range(self.num_seeds)]
        else:
            self.seeds = []

    @property
    def output_path(self) -> Path:
        return self.tuner.output_path

    @output_path.setter
    def output_path(self, value: Path):
        if not value:
            raise ValueError("output_path cannot be empty")
        self.tuner.output_path = value

    @staticmethod
    def get_best_config(
        tune_result: HpoResults, optimization_metric_id: str, max_iterations: int | None
    ) -> PatternParameters:
        best_configs = tune_result.get_best_configs(
            metric_id=optimization_metric_id,
            num_best_configs_to_consider=1,
            max_iterations=max_iterations,
        )
        if len(best_configs) > 1:
            raise RuntimeError(
                f"Unexpected number of best configurations {len(best_configs)} (expected 1)."
            )
        return best_configs[0]

    def _run_single_seed(
        self, tuner_params: dict[str, any] | None = None
    ) -> HpoResults | TestResults:
        """Run tune and test for a single seed."""
        output_path = Path(self.tuner.output_path)
        self.tuner.output_path = output_path / "tuning"

        test_output_path = output_path / "test"
        test_results_path = TestResults.file_name(test_output_path)
        if test_results_path.exists() and self.skip_existing_test_results:
            logger.info(f"Loading existing test results from '{test_results_path}'..")
            return TestResults.from_csv(directory=test_output_path)

        logger.info("Running tuner ..")
        hpo_result = self.tuner.run(tuner_params)

        if not self.test_dataset:
            logger.info("No test dataset is defined, returning tune results.")
            return hpo_result

        per_iteration_best_configs = []
        is_grid = (
            HpoAlgorithmType(self.tuner.algorithm_params["algorithm_type"])
            == HpoAlgorithmType.GRID
        )
        if is_grid:
            # For grid search test only one best config.
            max_test_evals = 1
        else:
            max_test_evals = hpo_result.size()
        logger.debug(f"Getting best configuration, max_test_evals='{max_test_evals}'..")

        optimization_metric_id = self.tuner.algorithm_params["optimization_metric_id"]
        for iteration_i in range(1, max_test_evals + 1):
            max_iterations = (
                None if is_grid else iteration_i
            )  # For grid search, take the top configs of all iterations.
            best_config = self.get_best_config(
                hpo_result, optimization_metric_id, max_iterations=max_iterations
            )
            per_iteration_best_configs.append(best_config)

        test_runner = self.tuner.rag_runner
        best_configs_results = []
        for max_iteration, best_config in enumerate(
            per_iteration_best_configs, start=1
        ):
            logger.debug(
                f"Running on test, best configuration for metric '{optimization_metric_id}' "
                f"at max_iteration {max_iteration} is : '{best_config}'."
            )
            best_config_result = test_runner.run(
                self.test_dataset, pattern_parameters=best_config
            )
            if best_config_result:
                best_config_result.name = f"best_till_iteration_{max_iteration}"
                best_configs_results.append(best_config_result)

        test_results = TestResults.create(best_configs_results)
        if tuner_params:
            test_results.add_to_summary(tuner_params)
        test_results.add_to_summary(self.tuner.algorithm_params)
        test_results.to_csv(directory=test_output_path, file_name="test_results.csv")
        logger.info(f"Test results written to '{test_output_path}'.")
        return test_results

    def run(
        self, tuner_params: dict[str, any] | None = None
    ) -> HpoResults | TestResults | MultiplePatternResults:
        """
        Run tune and test, optionally with multiple seeds.
        
        If num_seeds is set, runs the experiment multiple times with different seeds
        and returns MultiplePatternResults. Otherwise, runs once and returns
        HpoResults or TestResults.
        """
        if not self.num_seeds:
            # Single run without seeds
            return self._run_single_seed(tuner_params)

        # Multi-seed run
        all_seeds_results_list = []
        num_runs = len(self.seeds)
        base_output_path = Path(self.tuner.output_path)
        logger.info(
            f"Running multi-seed experiment with {num_runs} runs, seeds: '{self.seeds}'."
        )
        
        for seed_i in range(num_runs):
            seed = self.seeds[seed_i]
            self.tuner.output_path = base_output_path / f"seed_{seed}"
            logger.info(
                f"Running tune and test {seed_i+1} out of {len(self.seeds)} "
                f"(with seed '{seed}')"
            )
            tuner_params_with_seed = dict(tuner_params) if tuner_params else dict()
            tuner_params_with_seed["seed"] = seed
            single_seed_result = self._run_single_seed(tuner_params_with_seed)
            all_seeds_results_list.append(single_seed_result)

        all_seeds_results = MultiplePatternResults.concat(all_seeds_results_list)
        all_seeds_results.to_csv(
            directory=base_output_path,
            file_name="test_multi_seed_results.csv",
            with_predictions=False,
        )
        logger.info(f"All seeds results written to '{base_output_path}'.")
        return all_seeds_results

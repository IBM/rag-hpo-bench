import logging
import random
from dataclasses import dataclass

from rag_hpo_bench.hpo.pattern_results import MultiplePatternResults
from rag_hpo_bench.hpo.tune_and_test_runner import TuneAndTestRunner

logger = logging.getLogger(__name__)


@dataclass
class MultipleSeedsRunner:

    tune_and_test_runner: TuneAndTestRunner
    output_path: str = ""

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
    num_seeds: int | None = 1

    def __post_init__(self):
        self.output_path = self.tune_and_test_runner.output_path
        if self.num_seeds:
            random.seed(self.seed_of_seeds)
            self.seeds = [random.randint(1, 10000) for _ in range(self.num_seeds)]
        else:
            self.seeds = []

    def run(self) -> MultiplePatternResults:
        all_seeds_results_list = []
        num_runs = len(self.seeds) if self.num_seeds else 1
        logger.info(
            f"Running multi-seed experiment with {num_runs} runs, seeds: '{self.seeds}'."
        )
        for seed_i in range(num_runs):
            seed = self.seeds[seed_i] if self.num_seeds else None
            self.tune_and_test_runner.output_path = self.output_path
            if seed:
                self.tune_and_test_runner.output_path /= f"seed_{seed}"
            logger.info(
                f"Running tune and test {seed_i+1} out of {len(self.seeds)}.. "
                f"(with seed '{seed}')"
                if seed
                else ""
            )
            tuner_params = dict()
            if seed is not None:
                tuner_params["seed"] = seed
            single_seed_result = self.tune_and_test_runner.run(tuner_params)
            all_seeds_results_list.append(single_seed_result)

        all_seeds_results = MultiplePatternResults.concat(all_seeds_results_list)
        all_seeds_results.to_csv(
            directory=self.output_path,
            file_name="test_multi_seed_results.csv",
            with_predictions=False,
        )
        logger.info(f"All seeds results written to '{self.output_path}'.")
        return all_seeds_results

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from rageval.flows.dataset_id import DatasetID
from rag_hpo_bench.hpo.hpo_algorithm import GreedyMHPO, GridHPO, HpoAlgorithmType, RandomHPO
from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.rag_runner import RagRunner
from rag_hpo_bench.hpo.search_space import PatternParameters, SearchSpace

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Tuner(ABC):

    output_path: str | Path
    skip_existing_tunes: bool = False
    rag_runner: RagRunner
    algorithm_params: dict[str, any]
    metric_defs: dict[str, any]

    def __post_init__(self):
        # Initialize the optimization_metric_id which is what is used by an HPO algorithm
        self.algorithm_params = deepcopy(self.algorithm_params)
        optimization_metric_name = self.algorithm_params.get("optimization_metric_name")
        if not optimization_metric_name:
            raise ValueError(
                f"Missing key 'optimization_metric_name' from algorithm params '{self.algorithm_params}'."
            )
        self.algorithm_params["optimization_metric_id"] = self.metric_defs[
            optimization_metric_name
        ]["metric_id"]

    @abstractmethod
    def run(self, tuner_params: [str, any] = None) -> HpoResults:
        pass


def new_hpo_algorithm(
    search_space: SearchSpace, objective_function, algorithm_params: dict[str, any]
):
    algorithm_type = HpoAlgorithmType(algorithm_params["algorithm_type"])
    algorithm_params_copy = deepcopy(algorithm_params)  # do not to change input dict
    del algorithm_params_copy["algorithm_type"]
    match algorithm_type:
        case HpoAlgorithmType.RANDOM:
            return RandomHPO(
                search_space=search_space,
                objective_function=objective_function,
                **algorithm_params_copy,
            )
        case HpoAlgorithmType.GRID:
            return GridHPO(
                search_space=search_space,
                objective_function=objective_function,
                **algorithm_params_copy,
            )
        case HpoAlgorithmType.GREEDY_M:
            return GreedyMHPO(
                search_space=search_space,
                objective_function=objective_function,
                **algorithm_params_copy,
            )
        case _:
            raise RuntimeError(f"Unexpected algorithm type '{algorithm_type}'.")


@dataclass(kw_only=True)
class SingleStageTuner(Tuner):
    """
    An object for running a single tune, using a single set of parameters.
    """

    search_space: SearchSpace
    tune_dataset: DatasetID

    def __post_init__(self):
        super().__post_init__()  # Must call manually otherwise the base class __post_init__ is not called.
        self.output_path = Path(self.output_path)  # We ensure it is a Path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self, tuner_params: [str, any] = None) -> HpoResults:
        self.search_space.serialize(output_dir=self.output_path)
        tune_results_path = HpoResults.file_name(path=self.output_path)
        if tune_results_path.exists() and self.skip_existing_tunes:
            logger.info(f"Loading existing results from '{tune_results_path}'..")
            return HpoResults.from_csv(directory=self.output_path)

        if tuner_params:
            self.algorithm_params.update(tuner_params)

        def objective_function(pattern_parameters: PatternParameters):
            return self.rag_runner.run(self.tune_dataset, pattern_parameters)

        algorithm_params = deepcopy(self.algorithm_params)
        del algorithm_params[
            "optimization_metric_name"
        ]  # Not needed for hpo algorithm initialization
        hpo_algorithm = new_hpo_algorithm(
            self.search_space, objective_function, algorithm_params
        )
        hpo_results = hpo_algorithm.search()
        hpo_results.add_to_summary(self.algorithm_params)
        hpo_results.to_csv(
            directory=self.output_path,
            with_predictions=True,
        )
        return hpo_results

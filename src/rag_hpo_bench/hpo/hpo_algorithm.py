import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.pattern_results import PatternResults
from rag_hpo_bench.hpo.search_space import (
    PatternParameters,
    RagParameter,
    RagParameterName,
    SearchSpace,
    SearchSpaceParameter,
)

logger = logging.getLogger(__name__)


class HpoAlgorithmType(str, Enum):
    RANDOM = "random"
    GRID = "grid"
    GREEDY_M = "greedy_m"
    GREEDY_R = "greedy_r"


class HpoAlgorithmModel(BaseModel):
    algorithm_type: HpoAlgorithmType = Field(
        ..., description="The hpo algorithm to use"
    )
    max_iterations: int | None = Field(
        default=None, description="The maximum number of iterations"
    )
    optimization_metric_name: str = Field(
        ..., description="The name of the metric to optimize"
    )
    num_seeds: int | None = Field(
        default=None, description="The number of seeds to use"
    )
    model_config = ConfigDict(extra="forbid")


class HpoAlgorithmsSection(BaseModel):
    hpo_algorithms: list[HpoAlgorithmModel]
    model_config = ConfigDict(extra="forbid")


@dataclass
class HpoAlgorithm(ABC):
    search_space: SearchSpace
    optimization_metric_id: str
    objective_function: Callable[[PatternParameters], PatternResults]
    max_iterations: int | None = None

    @abstractmethod
    def search(self) -> HpoResults:
        pass

    def run_objective(self, pattern_parameters: PatternParameters) -> PatternResults:
        return self.objective_function(pattern_parameters)


@dataclass(kw_only=True)
class RandomHPO(HpoAlgorithm):
    """
    Run a random search over all search space parameter combinations.

    """

    seed: int

    def search(self) -> HpoResults:
        all_pattern_parameters = self.search_space.all_combinations()
        random.seed(self.seed)
        random.shuffle(all_pattern_parameters)
        pattern_results = []
        if self.max_iterations:
            all_pattern_parameters = all_pattern_parameters[: self.max_iterations]
        for pattern_index, pattern_parameters in enumerate(all_pattern_parameters):
            pattern_result = self.run_objective(pattern_parameters)
            pattern_result.name = f"Pattern_{pattern_index}"
            pattern_results.append(pattern_result)
        return HpoResults.create(pattern_results)


class GridHPO(HpoAlgorithm):
    """
    Run a grid search over all search space parameter combinations.

    """

    def search(self) -> HpoResults:
        all_pattern_parameters = self.search_space.all_combinations()
        logger.info(
            f"Running grid search over {len(all_pattern_parameters)} RAG configurations."
        )
        pattern_results = []
        assert (
            self.max_iterations is None
        ), f"We can not run grid search with max_iterations not None - actual value '{self.max_iterations}'"
        all_pattern_parameters = all_pattern_parameters[: self.max_iterations]
        for pattern_index, pattern_parameters in enumerate(all_pattern_parameters):
            pattern_result = self.run_objective(pattern_parameters)
            pattern_result.name = f"Pattern_{pattern_index}"
            pattern_results.append(pattern_result)
        return HpoResults.create(pattern_results)


@dataclass(kw_only=True)
class GreedyHPO(HpoAlgorithm):
    """
    Base class for greedy search algorithms over search space parameter combinations.
    
    Greedy algorithms optimize parameters sequentially in a specified order, fixing each
    parameter to its optimal value before moving to the next.
    """

    seed: int

    def __post_init__(self):
        assert (
            self.max_iterations
        ), f"It does not make sense to run Greedy search without max_iterations limit - use {HpoAlgorithmType.RANDOM} type"
        random.seed(self.seed)

        # Help dict
        self.param_to_values: dict[RagParameterName, list[RagParameter]] = dict()
        param: SearchSpaceParameter
        for param in self.search_space.parameters:
            pattern_parameter_list: list[RagParameter] = param.as_single_values()
            # We sort according to the value
            pattern_parameter_list.sort(key=lambda parameter: parameter.value)
            # Extract parameter name from the last element of the path
            param_name = param.path[-1]
            # Convert string to RagParameterName enum if needed
            if isinstance(param_name, str):
                param_name = RagParameterName(param_name)
            self.param_to_values[param_name] = pattern_parameter_list

    @abstractmethod
    def get_parameter_order(self) -> list[RagParameterName]:
        """Return the order in which parameters should be optimized."""
        pass

    def _get_possible_params_combination(
        self,
        to_be_optimized: RagParameterName,
        optimized_params: dict[RagParameterName, RagParameter],
    ) -> list[PatternParameters]:

        run_parameters: dict[RagParameterName, RagParameter] = dict()
        for parameter_name in self.param_to_values.keys():
            if parameter_name != to_be_optimized:  # We skip the param to be optimized
                if (
                    parameter_name in optimized_params
                ):  # We have already "fixed" this parameter to its optimum
                    run_parameters[parameter_name] = optimized_params[parameter_name]
                else:  # We randomly choose
                    run_parameters[parameter_name] = random.choice(
                        self.param_to_values[parameter_name]
                    )
        # We add all the values of the parameter to be optimized
        to_be_optimized_values = self.param_to_values[to_be_optimized]
        random.shuffle(to_be_optimized_values)

        combinations: list[dict[RagParameterName, RagParameter]] = list()
        for v in to_be_optimized_values:
            full_run_parameters = run_parameters.copy()
            full_run_parameters[to_be_optimized] = v

            combinations.append(full_run_parameters)

        return [PatternParameters(list(c.values())) for c in combinations]

    def search(self) -> HpoResults:

        pattern_results = []

        optimized_params: dict[RagParameterName, RagParameter] = dict()
        seen_configs = set()
        pattern_index = 0
        done: bool = False
        param: RagParameterName
        for param in self.get_parameter_order():
            spanned_search_space: list[PatternParameters] = (
                self._get_possible_params_combination(
                    to_be_optimized=param, optimized_params=optimized_params
                )
            )
            optimum_score: float | None = None
            pattern_parameters: PatternParameters
            for pattern_parameters in spanned_search_space:
                if done:
                    break
                hash_dict = pattern_parameters.to_hash()
                if hash_dict in seen_configs:
                    # We saw this config in the past
                    continue
                seen_configs.add(hash_dict)

                pattern_result: PatternResults = self.run_objective(pattern_parameters)
                pattern_result.name = f"Pattern_{pattern_index}"
                pattern_results.append(pattern_result)
                score = pattern_result.metric_stats[self.optimization_metric_id]["mean"]
                pattern_index += 1
                done = self.max_iterations is not None and (
                    pattern_index >= self.max_iterations
                )  # We have reached the limit to explore
                if not optimum_score or score < optimum_score:
                    # We got a better score for this configuration - we keep the parameter value
                    optimum_score = score
                    # We extract the value for this param:
                    for search_parameter in pattern_parameters.pattern_params:
                        # Compare with the last element of the path
                        search_param_name = search_parameter.path[-1]
                        if isinstance(search_param_name, str):
                            search_param_name = RagParameterName(search_param_name)
                        if param == search_param_name:
                            optimized_params[param] = search_parameter
                            break

        return HpoResults.create(pattern_results)



@dataclass(kw_only=True)
class GreedyMHPO(GreedyHPO):
    """
    Greedy search that optimizes Model parameters first, then Retrieval parameters.
    
    Parameter optimization order:
    1. GENERATIVE_MODEL (Generation model)
    2. EMBEDDING_MODEL (Retrieval)
    3. CHUNK_SIZE (Retrieval)
    4. CHUNK_OVERLAP (Retrieval)
    5. TOP_K (Retrieval)
    """

    def get_parameter_order(self) -> list[RagParameterName]:
        """Return model-first parameter optimization order."""
        return [
            RagParameterName.GENERATIVE_MODEL,
            RagParameterName.EMBEDDING_MODEL,
            RagParameterName.CHUNK_SIZE,
            RagParameterName.CHUNK_OVERLAP,
            RagParameterName.TOP_K,
        ]


@dataclass(kw_only=True)
class GreedyRHPO(GreedyHPO):
    """
    Greedy search that optimizes Retrieval parameters first, then Generation parameters.
    
    Parameter optimization order:
    1. EMBEDDING_MODEL (Retrieval)
    2. CHUNK_SIZE (Retrieval)
    3. CHUNK_OVERLAP (Retrieval)
    4. GENERATIVE_MODEL (Generation model)
    5. TOP_K (Number of retrieved chunks)
    """

    def get_parameter_order(self) -> list[RagParameterName]:
        """Return retrieval-first parameter optimization order."""
        return [
            RagParameterName.EMBEDDING_MODEL,
            RagParameterName.CHUNK_SIZE,
            RagParameterName.CHUNK_OVERLAP,
            RagParameterName.GENERATIVE_MODEL,
            RagParameterName.TOP_K,
        ]

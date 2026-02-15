import ast
import copy
import itertools
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from rageval.experiment_setup import ExperimentSetup
from rageval.utils.hash_utils import get_hash_dict

logger = logging.getLogger(__name__)


class RagParameterName(str, Enum):
    VECTOR_SPACE = "vector_space"
    VENDOR = "vendor"
    CHUNK_SIZE = "chunk_size"
    CHUNK_OVERLAP = "chunk_overlap"
    URI = "uri"
    CHUNK_UNIT = "chunk_unit"
    EMBEDDING_MODEL = "embedding_model"
    TOP_K = "top_k"
    TEMPERATURE = "temperature"
    MIN_NEW_TOKENS = "min_new_tokens"
    MAX_NEW_TOKENS = "max_new_tokens"
    GENERATIVE_MODEL = "generative_model"
    INFERENCE_TYPE = "inference_type"
    DocumentLoader = "document_loader"


class BaseParameter(BaseModel):
    """
    Base class for parameters with common attributes.
    """

    path: list[str]
    model_config = ConfigDict(extra="forbid")


class RagParameter(BaseParameter):
    """
    A single parameter used in a pattern, containing one specific value.

    The single value may be a simple type (str, int, float) or a compound type (list, dict, etc).
    """

    value: Any
    model_config = ConfigDict(extra="forbid")

    def paths_match(self, required_path: list[str]):
        """
        self.path = ["retrieval", "top-k"]
        self.value = 5
        ->
        required_path = ["retrieval"]  -> True
        required_path = ["retrieval", "top-k"]  -> True
        required_path = ["generation"]  -> False
        """
        match = True
        for required_path_item, path_item in zip(required_path, self.path):
            if required_path_item != path_item:
                match = False
                break
        return match

    def as_nested_dict(self):
        """
        self.path = ["retrieval", "top-k"]
        self.value = 5
        ->
        {
            "retrieval": {
                "top-k": 5
            }
        }
        """
        result = {}
        current = result
        for path_item in self.path[:-1]:
            current[path_item] = {}
            current = current[path_item]
        current[self.path[-1]] = self.value
        return result


class SearchSpaceParameter(BaseParameter):
    """
    A single parameter that can take part in a search space.
    Has a list of potential values.
    """

    values: Any
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context) -> None:
        # We force values to be a list of .
        if not isinstance(self.values, list):
            self.values = [self.values]

    def as_single_values(self) -> list[RagParameter]:
        return [RagParameter(path=self.path, value=value) for value in self.values]


@dataclass
class PatternParameters:
    """
    All parameters needed to run one rag pattern.
    """

    pattern_params: list["RagParameter"]

    def __post_init__(self):
        for pattern_param in self.pattern_params:
            param_value = pattern_param.value
            if isinstance(param_value, str) and param_value.startswith("["):
                # Its a list, convert it
                pattern_param.value = ast.literal_eval(pattern_param.value)

    def get_path_to_values_dict(self):
        return {
            ".".join(pattern_param.path): pattern_param.value
            for pattern_param in self.pattern_params
        }

    def get_params(self, *required_path) -> dict[str : [int | str | float]]:
        non_strings = [arg for arg in required_path if not isinstance(arg, str)]

        if non_strings:
            print(f"Non-string arguments: {non_strings}")
            raise TypeError(
                f"Required path must be strings. Non-string values: {non_strings}. All args: {required_path}."
            )

        required_path = list(required_path)

        matching_params = [
            pattern_parameter
            for pattern_parameter in self.pattern_params
            if pattern_parameter.paths_match(required_path)
        ]
        if not matching_params:
            return {}

        nested_dicts = [
            mathing_param.as_nested_dict() for mathing_param in matching_params
        ]

        result = nested_dicts[0] if nested_dicts else {}
        for nested_dict in nested_dicts[1:]:
            self.merge_dicts(destination=result, source=nested_dict)

        for require_path_item in required_path:
            result = result[require_path_item]
        return result

    @staticmethod
    def merge_dicts(destination: dict, source: dict):
        for k, v in source.items():
            if k not in destination:
                destination[k] = copy.deepcopy(v)
            else:
                if isinstance(destination[k], Mapping) and isinstance(v, Mapping):
                    PatternParameters.merge_dicts(destination=destination[k], source=v)
                else:
                    raise TypeError(
                        f"Conflict at key '{k}': cannot merge non mapping type: {type(destination[k])} and {type(v)}.\n"
                        f"destination: {destination[k]}\nsource: {v}"
                    )

    def to_hash(self) -> str:
        return get_hash_dict({"_".join(p.path): p.value for p in self.pattern_params})

    @staticmethod
    def from_dict(pattern_parameters_dict: dict):
        rag_parameters = PatternParameters._from_dict(pattern_parameters_dict)
        return PatternParameters(rag_parameters)

    @staticmethod
    def _from_dict(pattern_parameters_dict: dict, current_path: list[str] = None):
        if not current_path:
            current_path = []
        rag_parameters = []
        for path_item, value in pattern_parameters_dict.items():
            new_path = current_path.copy()
            new_path.append(path_item)
            if isinstance(value, dict):
                rag_parameters.extend(
                    PatternParameters._from_dict(value, current_path=new_path)
                )
            else:
                rag_parameters.append(RagParameter(path=new_path, value=value))
        return rag_parameters


class SearchSpace(BaseModel):

    parameters: list[SearchSpaceParameter]
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def create_search_space_params(
        d: dict, prefix: list[str] = None
    ) -> list[SearchSpaceParameter]:
        if prefix is None:
            prefix = []

        items = []
        for k, v in d.items():
            current_path = prefix + [k]
            if isinstance(v, dict):
                items.extend(SearchSpace.create_search_space_params(v, current_path))
            else:
                items.append(SearchSpaceParameter(path=current_path, values=v))
        return items

    @staticmethod
    def from_experiment_setup(experiment_setup: ExperimentSetup):
        search_space_dict = experiment_setup.get_search_space_dict()
        search_space_parameters = SearchSpace.create_search_space_params(
            search_space_dict
        )
        logger.info(
            f"Creating a search space from {len(search_space_parameters)} rag parameters."
        )
        return SearchSpace(parameters=search_space_parameters)

    def serialize(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "search_space.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [p.model_dump_json() for p in self.parameters],
                f,
                indent=4,
                ensure_ascii=False,
            )

    def all_combinations(self) -> list[PatternParameters]:
        """
        Return all possible configurations that are in the search space.
        """
        # Create a list of lists, where each inner list represents all the values of one parameter
        parameter_single_values: list[list[RagParameter]] = [
            parameter.as_single_values() for parameter in self.parameters
        ]
        pattern_parameters = itertools.product(*parameter_single_values)
        return [PatternParameters(list(p)) for p in pattern_parameters]

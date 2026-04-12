import dataclasses
import re
from dataclasses import dataclass, field
from typing import Literal

from rag_hpo_bench.data_models.data_sampling_params import DataSamplingParams
from rag_hpo_bench.data_models.dataset_names import DatasetName


@dataclass
class DatasetID:
    dataset_name: DatasetName | str
    split: Literal["Dev", "Test"] | None = None
    sampling_params: DataSamplingParams = field(default_factory=DataSamplingParams)

    def __post_init__(self):
        if isinstance(self.dataset_name, str):
            try:
                self.dataset_name = self._dataset_str_to_dataset(self.dataset_name)
            except ValueError as e:
                raise ValueError(
                    f"Unexpected dataset name value '{self.dataset_name}' not found in catalog ({[item.value for item in DatasetName]})"
                ) from e

    @staticmethod
    def _dataset_str_to_dataset(dataset_str: str) -> DatasetName:
        return DatasetName(dataset_str)

    @staticmethod
    def create(dataset_setup: dict[str, any]):
        dataset_name_str = dataset_setup["id"]
        dataset_name = DatasetID._dataset_str_to_dataset(dataset_name_str)

        data_sampling_params = DataSamplingParams(**(dataset_setup.get("sampling") or {}))

        return DatasetID(
            dataset_name=dataset_name,
            split=dataset_setup.get("split"),
            sampling_params=data_sampling_params,
        )

    def _format_dataset_name(self):
        if isinstance(self.dataset_name, DatasetName):
            return self.dataset_name.value
        else:
            return self.dataset_name

    def as_string(self):
        dataset_name = self._format_dataset_name()

        dataset_id = f"name-{dataset_name}"

        if self.split:
            dataset_id += f"_split-{self.split}"

        sample_id = self.sampling_params.as_id()
        if sample_id:
            dataset_id += f"_{sample_id}"

        return dataset_id

    def dataset_name_value_as_string(self) -> str:
        if isinstance(self.dataset_name, DatasetName):
            return self.dataset_name.value
        elif isinstance(self.dataset_name, str):
            return self.dataset_name
        return "NONE"

    @staticmethod
    def from_string(dataset_id_str):
        """
        Examples:
            name-clap_nq_v1
            name-clap_nq_v1_split-train
            name-clap_nq_v1_split-train_q-1000_seed-43
            name-clap_nq_v1_split-train_q-1000_docs-factor-9_seed-43
        """
        pattern = r"^name-([a-zA-Z_0-9]+)(?:_split-([a-zA-Z]+))?(_q-(\d+))?(_docs-factor-(\d+))?(_seed-(\d+))?$"

        # Apply regex to the input string
        match = re.match(pattern, dataset_id_str)

        if match:
            dataset_name = match.group(1)
            split = match.group(2) if match.group(2) else None
            limited_questions = int(match.group(4)) if match.group(4) else None
            docs_factor = int(match.group(6)) if match.group(6) else None
            seed = int(match.group(8)) if match.group(8) else None

            sampling_params = DataSamplingParams(
                question_limit=limited_questions,
                document_factor=docs_factor,
                seed=seed,
            )
            return DatasetID(
                dataset_name=dataset_name,
                split=split,
                sampling_params=sampling_params,
            )
        else:
            raise RuntimeError(f"Unable to construct a dataset id from string '{str}'.")

    def as_dict(self):
        return dataclasses.asdict(self)

    def as_cache_key(self):
        return self.as_dict()

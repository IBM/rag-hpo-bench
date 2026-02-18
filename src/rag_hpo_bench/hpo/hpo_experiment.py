import logging
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from rag_hpo_bench.data_models import DatasetID
from rag_hpo_bench.hpo.hpo_algorithm import HpoAlgorithmType
from rag_hpo_bench.hpo.rag_runner import RagRunner
from rag_hpo_bench.hpo.search_space import SearchSpace
from rag_hpo_bench.hpo.tuner import Tuner
from rag_hpo_bench.hpo.tune_and_test_runner import TuneAndTestRunner

logger = logging.getLogger(__name__)


@dataclass
class HpoExperiment:
    search_space: SearchSpace
    tune_dataset: DatasetID
    test_dataset: DatasetID | None
    algorithm_params: dict
    optimization_metric_id: str
    output_path: Path
    skip_existing_tunes: bool = False
    skip_existing_test_results: bool = False
    clean_output_dir: bool = False

    def run(self):
        algorithm_type = HpoAlgorithmType(self.algorithm_params["algorithm_type"])
        self.output_path = self.output_path / algorithm_type.value
        self.output_path = self.output_path / self.tune_dataset.as_string()
        if self.test_dataset:
            self.output_path = self.output_path / self.test_dataset.as_string()
        # We check the content of output_path:
        if os.path.exists(self.output_path) and len(os.listdir(self.output_path)) > 0:
            if self.clean_output_dir:
                shutil.rmtree(self.output_path)
            else:
                logger.warning(
                    f"Output directory {self.output_path} exists and contains results before the run!"
                )
        self.output_path.mkdir(parents=True, exist_ok=True)

        # before pop() make a copy to avoid affecting objects that also
        # have access to this dict
        algorithm_params = deepcopy(self.algorithm_params)
        num_seeds = (
            algorithm_params.pop("num_seeds")
            if "num_seeds" in algorithm_params
            else None
        )
        assert not (
            algorithm_type == HpoAlgorithmType.GRID and num_seeds is not None
        ), f"Can not set HPO Algorithm to be {algorithm_type.value} and num_seeds is not None (num_seeds = {num_seeds})."

        tuner = Tuner(
            search_space=self.search_space,
            rag_runner=RagRunner(),
            algorithm_params=algorithm_params,
            tune_dataset=self.tune_dataset,
            output_path=self.output_path,
            skip_existing_tunes=self.skip_existing_tunes,
            optimization_metric_id=self.optimization_metric_id,
        )

        tune_and_test_runner = TuneAndTestRunner(
            tuner=tuner,
            test_dataset=self.test_dataset,
            skip_existing_test_results=self.skip_existing_test_results,
            num_seeds=int(num_seeds) if num_seeds else None,
        )

        all_seeds_results = tune_and_test_runner.run()
        logger.info(f"Experiment output written to '{self.output_path}'.")
        return all_seeds_results

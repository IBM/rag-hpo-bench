from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rag_hpo_bench.hpo.search_space import PatternParameters


@dataclass
class PatternResults:

    pattern_parameters: PatternParameters
    evaluated_benchmark: pd.DataFrame
    metric_stats: dict[str, dict[str, float]]
    name: str = ""

    @staticmethod
    def create(
        pattern_parameters: PatternParameters,
        evaluated_benchmark: list[dict[str, Any]],
        metric_stats: dict[str, dict[str, float]],
    ) -> "PatternResults":
        evaluated_benchmark = pd.DataFrame(evaluated_benchmark)

        _required_result_fields = [
            "q_id",
            "question",
            "ground_truths",
            "ground_truths_context_ids",
            "prompt",
            "prompt_tokens",
            "answer",
            "output_tokens",
            "input_tokens",
        ]
        missing_result_fields = [
            result_field
            for result_field in _required_result_fields
            if result_field not in evaluated_benchmark.columns
        ]
        if missing_result_fields:
            raise RuntimeError(
                f"Missing results fields: '{missing_result_fields}'. Input results have fields: '{evaluated_benchmark.columns}'."
            )

        return PatternResults(pattern_parameters, evaluated_benchmark, metric_stats)

    def write_results_files(self, csv_path: Path):
        csv_path.parent.mkdir(exist_ok=True, parents=True)

        json_report = self.evaluated_benchmark
        json_path = csv_path.with_suffix(".json")
        json_report.to_json(json_path, orient="records", lines=True, force_ascii=False)

        csv_report = json_report.drop(["trajectory", "logtrace"], axis=1)
        csv_report.to_csv(csv_path)

    def get_evaluation_stats(self):
        return self.metric_stats


@dataclass
class MultiplePatternResults:
    """
    A summary of the pattern, including the input parameters and evaluation metrics results per pattern.
    """

    _results_summary: pd.DataFrame

    """
    Detailed results of the tested RAG patterns.
    """
    patterns_results: list[PatternResults] = field(default_factory=list)

    default_file_name: str = "pattern_results"

    def size(self):
        return len(self._results_summary)

    @classmethod
    def file_name(cls, path: Path, file_name: str = None) -> Path:
        if not file_name:
            file_name = f"{cls.default_file_name}.csv"
        return Path(path / file_name)

    @classmethod
    def file_path(cls, directory: Path, file_name=None):
        if not file_name:
            file_path = cls.file_name(directory)
        else:
            file_path = directory / file_name
        return file_path

    @classmethod
    def from_csv(cls, directory: Path, file_name=None) -> "MultiplePatternResults":
        file_path = cls.file_path(directory, file_name)
        return cls(
            _results_summary=pd.read_csv(file_path),
            patterns_results=[],
        )

    @classmethod
    def create(cls, patterns_results: list[PatternResults]):
        return cls(
            _results_summary=cls._create_results_summary(patterns_results),
            patterns_results=patterns_results,
        )

    def to_csv(self, directory: Path, file_name: str = None, with_predictions=True):
        directory.mkdir(parents=True, exist_ok=True)
        if with_predictions:
            for pattern_result in self.patterns_results:
                assert pattern_result.name != ""
                pattern_result.write_results_files(
                    directory / f"{pattern_result.name}.csv"
                )
        self._results_summary.to_csv(self.file_name(directory, file_name))

    @staticmethod
    def _create_results_summary(patterns_results: list[PatternResults]) -> pd.DataFrame:
        results_summary = pd.DataFrame()
        results_summary["name"] = [pattern.name for pattern in patterns_results]

        # Add the parameters used in each pattern
        per_pattern_parameters = pd.DataFrame(
            [
                pattern.pattern_parameters.get_path_to_values_dict()
                for pattern in patterns_results
            ]
        )
        results_summary = pd.concat([results_summary, per_pattern_parameters], axis=1)

        # Add metric stats
        all_patterns_evaluation_stats = [
            pattern_results.get_evaluation_stats()
            for pattern_results in patterns_results
        ]
        metric_and_stat_to_score = [
            {
                f"{metric_id}_{stat_name}": metric_score
                for metric_id, metric_scores in evaluation_stats.items()
                for stat_name, metric_score in metric_scores.items()
            }
            for evaluation_stats in all_patterns_evaluation_stats
        ]
        results_summary = pd.concat(
            [results_summary, pd.DataFrame(metric_and_stat_to_score)], axis=1
        )

        results_summary["input_tokens"] = [
            pattern.evaluated_benchmark["input_tokens"].mean()
            for pattern in patterns_results
        ]
        results_summary["output_tokens"] = [
            pattern.evaluated_benchmark["output_tokens"].mean()
            for pattern in patterns_results
        ]

        return results_summary

    def add_to_summary(self, params: dict[str, Any]):
        for key, value in params.items():
            self._results_summary[key] = value

    @staticmethod
    def concat(tune_results: list["MultiplePatternResults"]):
        frames = [result._results_summary for result in tune_results]

        return MultiplePatternResults(
            _results_summary=pd.concat(frames),
            patterns_results=[
                pattern_result
                for tune_result in tune_results
                for pattern_result in tune_result.patterns_results
            ],
        )
import logging
from dataclasses import dataclass
from typing import Any

from rag_hpo_bench.hpo.pattern_results import MultiplePatternResults
from rag_hpo_bench.hpo.search_space import PatternParameters

logger = logging.getLogger(__name__)


@dataclass
class HpoResults(MultiplePatternResults):
    default_file_name: str = "hpo_results"

    @staticmethod
    def add_sub_keys(item: dict[str, Any], key: str):
        source = item[key]
        for sub_key, value in source.items():
            item[f"{sub_key}"] = value

    def get_best_configs(
        self,
        metric_id: str,
        num_best_configs_to_consider: int = 1,
        max_iterations: int = None,
    ) -> list[PatternParameters]:
        per_iteration_results = self._results_summary
        if max_iterations:
            per_iteration_results = per_iteration_results.head(max_iterations)
        sorted_by_metric = per_iteration_results.sort_values(
            by=f"{metric_id}_mean", ascending=False
        )
        best_configs = sorted_by_metric.head(n=num_best_configs_to_consider)
        results = []
        assert all(pattern.name != "" for pattern in self.patterns_results)
        patterns_by_name = {pattern.name: pattern for pattern in self.patterns_results}
        for _, config_row in best_configs.iterrows():
            best_pattern_name = config_row["name"]
            best_pattern = patterns_by_name[best_pattern_name]
            best_pattern_parameters = best_pattern.pattern_parameters
            results.append(best_pattern_parameters)
        return results

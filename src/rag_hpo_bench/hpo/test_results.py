import logging
from dataclasses import dataclass

from rag_hpo_bench.hpo.pattern_results import MultiplePatternResults

logger = logging.getLogger(__name__)


@dataclass
class TestResults(MultiplePatternResults):
    default_file_name: str = "test_results"

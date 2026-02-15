from rageval.flows.dataset_id import DatasetID

from rag_hpo_bench.hpo.pattern_results import PatternResults
from rag_hpo_bench.hpo.search_space import PatternParameters
from rag_hpo_bench.hpo.single_pattern_rag import SinglePatternRAG


class RagRunner:
    """
    Runner for executing RAG patterns by loading cached results from HuggingFace.
    """

    def run(
        self, dataset_id: DatasetID, pattern_parameters: PatternParameters
    ) -> PatternResults:
        """
        Run RAG pattern by loading cached results from HuggingFace dataset.
        
        Args:
            dataset_id: Dataset identifier
            pattern_parameters: RAG pattern parameters
            
        Returns:
            PatternResults with cached evaluation metrics
        """
        single_rag_pattern = SinglePatternRAG(
            dataset_id=dataset_id,
            pattern_parameters=pattern_parameters,
        )
        return single_rag_pattern.run()

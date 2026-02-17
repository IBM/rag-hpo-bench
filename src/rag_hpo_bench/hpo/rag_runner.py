"""
Runner for executing RAG patterns by loading cached results from HuggingFace.
"""
import logging
from typing import Any

import pandas as pd
from rageval.flows.dataset_id import DatasetID

from rag_hpo_bench.hpo.pattern_results import PatternResults
from rag_hpo_bench.hpo.search_space import PatternParameters
from rag_hpo_bench.utils.dataset_loader import load_rag_configurations_summary

logger = logging.getLogger(__name__)


class RagRunner:
    """
    Runner for executing RAG patterns by loading cached results from HuggingFace.
    
    This class reads from the RAG-HPO-bench HuggingFace dataset to retrieve cached
    evaluation results instead of running the actual RAG pipeline.
    
    Dataset: https://huggingface.co/datasets/ibm-research/rag-hpo-bench
    """

    def run(
        self, dataset_id: DatasetID, pattern_parameters: PatternParameters
    ) -> PatternResults | None:
        """
        Run RAG pattern by loading cached results from HuggingFace dataset.
        
        Args:
            dataset_id: Dataset identifier
            pattern_parameters: RAG pattern parameters
            
        Returns:
            PatternResults with cached evaluation metrics, or None if no match found
            
        Raises:
            Exception: If loading from HuggingFace fails or no match is found
        """
        # Load the summary from HuggingFace (cached after first call)
        df = load_rag_configurations_summary()
        
        # Extract parameters from pattern_parameters
        params_dict = pattern_parameters.get_path_to_values_dict()
        
        # Map parameter paths to column names in the summary file
        # Based on analyze_configs_distribution.py structure
        param_mapping = {
            "indexing.chunking.size": "chunking_size",
            "indexing.chunking.overlap": "chunking_overlap",
            "indexing.embedding.model": "embedding_model",
            "inference.retrieval.top-k": "top-k",
            "inference.generation.model": "generator",
        }
        
        # Build filter conditions
        filters = {}
        for param_path, col_name in param_mapping.items():
            if param_path in params_dict:
                filters[col_name] = params_dict[param_path]
        
        # Add dataset_id filter
        dataset_id_str = dataset_id.as_string() if hasattr(dataset_id, 'as_string') else str(dataset_id)
        filters["dataset_id"] = dataset_id_str
        
        # Filter the dataframe
        logger.info(f"Filtering with: {filters}")
        mask = pd.Series([True] * len(df))
        for col_name, value in filters.items():
            if col_name not in df.columns:
                logger.warning(f"Column '{col_name}' not found in summary file. Available columns: {df.columns.tolist()}")
                continue
            mask &= df[col_name] == value
        
        matched_rows = df[mask]
        
        if len(matched_rows) == 0:
            logger.warning(
                f"No matching configuration found in summary file for filters: {filters}"
            )
            return None
        
        if len(matched_rows) > 1:
            logger.warning(
                f"Multiple matching configurations found ({len(matched_rows)}). Using the first one."
            )
        
        # Get the first matching row
        row = matched_rows.iloc[0]
        
        # Extract metric results
        metric_stats = self._extract_metric_stats(row)
        
        # Create a minimal evaluated_benchmark DataFrame
        # Since we don't have per-question data, create a summary row
        evaluated_benchmark = self._create_evaluated_benchmark(row, metric_stats)
        
        # Create and return PatternResults
        pattern_results = PatternResults.create(
            pattern_parameters=pattern_parameters,
            evaluated_benchmark=evaluated_benchmark,
            metric_stats=metric_stats,
        )
        
        logger.info(f"Successfully loaded cached results with metrics: {list(metric_stats.keys())}")
        return pattern_results
    
    def _extract_metric_stats(self, row: pd.Series) -> dict[str, dict[str, float]]:
        """
        Extract metric statistics from the summary row.
        
        Args:
            row: A row from the summary DataFrame
            
        Returns:
            Dictionary mapping metric names to their statistics (mean, std, etc.)
        """
        metric_stats = {}
        
        # Common metric column names from analyze_configs_distribution.py
        metric_columns = {
            "Lexical-AC": "answer_correctness",
            "Lexical-FF": "faithfulness",
            "LLMaaJ-AC": "ragas.answer_correctness.gpt-4o-mini-2024-07-18",
            "context_correctness": "context_correctness",
        }
        
        for col_name, metric_id in metric_columns.items():
            if col_name in row.index and pd.notna(row[col_name]):
                # Store as mean value (summary file typically has mean scores)
                metric_stats[metric_id] = {
                    "mean": float(row[col_name]),
                    "std": 0.0,  # Not available in summary
                    "min": float(row[col_name]),
                    "max": float(row[col_name]),
                }
        
        return metric_stats
    
    def _create_evaluated_benchmark(
        self, row: pd.Series, metric_stats: dict[str, dict[str, float]]
    ) -> list[dict[str, Any]]:
        """
        Create a minimal evaluated benchmark list from the summary row.
        
        Args:
            row: A row from the summary DataFrame
            metric_stats: Extracted metric statistics
            
        Returns:
            List of dictionaries representing evaluated questions
        """
        # Create a single summary entry since we don't have per-question data
        benchmark_entry = {
            "q_id": "summary",
            "question": "Summary from cached results",
            "ground_truths": [],
            "ground_truths_context_ids": [],
            "prompt": "",
            "prompt_tokens": 0,
            "answer": "",
            "output_tokens": 0,
            "input_tokens": 0,
            "trajectory": [],
            "logtrace": {},
        }
        
        # Add metric scores
        for metric_id, stats in metric_stats.items():
            benchmark_entry[metric_id] = stats["mean"]
        
        return [benchmark_entry]

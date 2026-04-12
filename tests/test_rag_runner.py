"""
Tests for RagRunner.run method.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from rag_hpo_bench.data_models.dataset_id import DatasetID
from rag_hpo_bench.data_models.dataset_names import DatasetName
from rag_hpo_bench.hpo.pattern_results import PatternResults
from rag_hpo_bench.hpo.rag_runner import RagRunner
from rag_hpo_bench.hpo.search_space import PatternParameters, RagParameter


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame mimicking the HuggingFace dataset structure."""
    return pd.DataFrame(
        [
            {
                "Dataset": "AIArxiv",
                "Split": "Test",
                "Chunk Size": 512,
                "Chunk Overlap": 0,
                "Embedding Model": "text-embedding-ada-002",
                "Top-K": 5,
                "Generative Model": "gpt-3.5-turbo",
                "Lexical-AC": 0.75,
                "Lexical-FF": 0.82,
                "LLMaaJ-AC": 0.78,
                "context_correctness": 0.85,
            },
            {
                "Dataset": "AIArxiv",
                "Split": "Test",
                "Chunk Size": 1024,
                "Chunk Overlap": 0.25,
                "Embedding Model": "text-embedding-ada-002",
                "Top-K": 10,
                "Generative Model": "gpt-4",
                "Lexical-AC": 0.80,
                "Lexical-FF": 0.88,
                "LLMaaJ-AC": 0.83,
                "context_correctness": 0.90,
            },
            {
                "Dataset": "BioASQ",
                "Split": "Dev",
                "Chunk Size": 512,
                "Chunk Overlap": 0,
                "Embedding Model": "text-embedding-ada-002",
                "Top-K": 5,
                "Generative Model": "gpt-3.5-turbo",
                "Lexical-AC": 0.70,
                "Lexical-FF": 0.77,
                "LLMaaJ-AC": 0.72,
                "context_correctness": 0.80,
            },
        ]
    )


@pytest.fixture
def sample_pattern_parameters():
    """Create sample pattern parameters."""
    return PatternParameters(
        [
            RagParameter(path=["indexing", "chunking", "size"], value=512),
            RagParameter(path=["indexing", "chunking", "overlap"], value=50),
            RagParameter(path=["indexing", "embedding", "model"], value="text-embedding-ada-002"),
            RagParameter(path=["inference", "retrieval", "top-k"], value=5),
            RagParameter(path=["inference", "generation", "model"], value="gpt-3.5-turbo"),
        ]
    )


@pytest.fixture
def sample_dataset_id():
    """Create a sample dataset ID."""
    return DatasetID(
        dataset_name=DatasetName.AIArxiv,
        split="Test",
    )


class TestRagRunnerRun:
    """Test suite for RagRunner.run method."""

    def test_run_successful_match(
        self, sample_dataframe, sample_pattern_parameters, sample_dataset_id
    ):
        """Test successful run with matching configuration."""
        runner = RagRunner()

        with patch(
            "rag_hpo_bench.hpo.rag_runner.load_rag_configurations_summary",
            return_value=sample_dataframe,
        ):
            result = runner.run(sample_dataset_id, sample_pattern_parameters)

        # Verify result is not None
        assert result is not None
        assert isinstance(result, PatternResults)

        # Verify pattern parameters are preserved
        assert result.pattern_parameters == sample_pattern_parameters

        # Verify metric stats are extracted correctly
        assert "Lexical-AC" in result.metric_stats
        assert "Lexical-FF" in result.metric_stats
        assert result.metric_stats["Lexical-AC"]["mean"] == 0.75
        assert result.metric_stats["Lexical-FF"]["mean"] == 0.82

        # Verify evaluated_benchmark structure
        assert isinstance(result.evaluated_benchmark, pd.DataFrame)
        assert len(result.evaluated_benchmark) == 1
        assert "q_id" in result.evaluated_benchmark.columns
        assert result.evaluated_benchmark.iloc[0]["q_id"] == "summary"

    def test_run_no_match_raises_error(self, sample_dataframe, sample_dataset_id):
        """Test that run raises ValueError when no matching configuration is found."""
        runner = RagRunner()

        # Create parameters that don't match any row
        non_matching_params = PatternParameters(
            [
                RagParameter(path=["indexing", "chunking", "size"], value=2048),
                RagParameter(path=["indexing", "chunking", "overlap"], value=200),
                RagParameter(path=["indexing", "embedding", "model"], value="non-existent-model"),
                RagParameter(path=["inference", "retrieval", "top-k"], value=100),
                RagParameter(
                    path=["inference", "generation", "model"], value="non-existent-generator"
                ),
            ]
        )

        with patch(
            "rag_hpo_bench.hpo.rag_runner.load_rag_configurations_summary",
            return_value=sample_dataframe,
        ):
            with pytest.raises(ValueError) as exc_info:
                runner.run(sample_dataset_id, non_matching_params)

            # Verify error message mentions no matching configuration
            assert "No matching configuration found" in str(exc_info.value)

    def test_run_with_partial_parameters_raises_error(self, sample_dataset_id):
        """Test that partial parameters that match multiple rows raise an error."""
        runner = RagRunner()

        # Create a dataframe where partial parameters match multiple rows
        df_with_duplicates = pd.DataFrame(
            [
                {
                    "Dataset": "AIArxiv",
                    "Split": "Test",
                    "Chunk Size": 512,
                    "Chunk Overlap": 0,
                    "Embedding Model": "text-embedding-ada-002",
                    "Top-K": 5,
                    "Generative Model": "gpt-3.5-turbo",
                    "Lexical-AC": 0.75,
                },
                {
                    "Dataset": "AIArxiv",
                    "Split": "Test",
                    "Chunk Size": 512,
                    "Chunk Overlap": 0,
                    "Embedding Model": "text-embedding-ada-002",
                    "Top-K": 5,
                    "Generative Model": "gpt-4",  # Different generator
                    "Lexical-AC": 0.80,
                },
            ]
        )

        # Only specify parameters that match both rows (missing generator)
        partial_params = PatternParameters(
            [
                RagParameter(path=["indexing", "chunking", "size"], value=512),
                RagParameter(path=["indexing", "chunking", "overlap"], value=50),
                RagParameter(
                    path=["indexing", "embedding", "model"], value="text-embedding-ada-002"
                ),
                RagParameter(path=["inference", "retrieval", "top-k"], value=5),
            ]
        )

        with patch(
            "rag_hpo_bench.hpo.rag_runner.load_rag_configurations_summary",
            return_value=df_with_duplicates,
        ):
            # Should raise ValueError because partial params match multiple rows
            with pytest.raises(ValueError) as exc_info:
                runner.run(sample_dataset_id, partial_params)

            assert "Multiple matching configurations found" in str(exc_info.value)
            assert "missing parameters" in str(exc_info.value)

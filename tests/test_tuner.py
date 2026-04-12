"""
Tests for Tuner.run method.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from rag_hpo_bench.data_models.dataset_id import DatasetID
from rag_hpo_bench.data_models.dataset_names import DatasetName
from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.rag_runner import RagRunner
from rag_hpo_bench.hpo.search_space import (
    SearchSpace,
    SearchSpaceParameter,
)
from rag_hpo_bench.hpo.tuner import Tuner


@pytest.fixture
def sample_hf_dataframe():
    """Create a sample DataFrame mimicking the HuggingFace dataset structure."""
    # Create a comprehensive dataset with all combinations of the search space
    data = []
    for chunk_size in [512, 1024]:
        for chunk_overlap_ratio in [0.25, 0.5]:
            for top_k in [5, 10]:
                data.append(
                    {
                        "Dataset": "AIArxiv",
                        "Split": "Test",
                        "Chunk Size": chunk_size,
                        "Chunk Overlap": chunk_overlap_ratio,
                        "Embedding Model": "text-embedding-ada-002",
                        "Top-K": top_k,
                        "Generative Model": "gpt-3.5-turbo",
                        "Lexical-AC": 0.70 + (chunk_size / 10000) + (top_k / 100),
                        "Lexical-FF": 0.75 + (chunk_size / 10000) + (top_k / 100),
                        "LLMaaJ-AC": 0.72 + (chunk_size / 10000) + (top_k / 100),
                        "context_correctness": 0.80 + (chunk_size / 10000) + (top_k / 100),
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def sample_search_space():
    """Create a sample search space with multiple parameters."""
    # Note: chunk_overlap values must be ratios that match the dataframe (0.25 and 0.5)
    return SearchSpace(
        parameters=[
            SearchSpaceParameter(
                path=["data_pipeline", "params", "indexing", "chunk_size"], values=[512, 1024]
            ),
            SearchSpaceParameter(
                path=["data_pipeline", "params", "indexing", "chunk_overlap"], values=[0.25, 0.5]
            ),
            SearchSpaceParameter(
                path=["data_pipeline", "params", "indexing", "vector_space", "embedding_model"],
                values=["text-embedding-ada-002"],
            ),
            SearchSpaceParameter(
                path=["inference_pipeline", "params", "retrieval", "top_k"], values=[5, 10]
            ),
            SearchSpaceParameter(
                path=["inference_pipeline", "params", "generation", "generative_model"],
                values=["gpt-3.5-turbo"],
            ),
        ]
    )


@pytest.fixture
def sample_algorithm_params():
    """Create sample algorithm parameters for grid search."""
    return {
        "algorithm_type": "grid",
    }


@pytest.fixture
def sample_dataset_id():
    """Create a sample dataset ID."""
    return DatasetID(
        dataset_name=DatasetName.AIArxiv,
        split="Test",
    )


class TestTunerRun:
    """Test suite for Tuner.run method."""

    def test_run_with_valid_inputs(
        self,
        tmp_path,
        sample_hf_dataframe,
        sample_search_space,
        sample_algorithm_params,
        sample_dataset_id,
    ):
        """Test successful run with valid inputs using actual RagRunner and HPO algorithm."""
        # Setup
        output_path = tmp_path / "tuner_output"

        # Create actual RagRunner (not mocked)
        rag_runner = RagRunner()

        # Create tuner with actual objects
        tuner = Tuner(
            output_path=output_path,
            skip_existing_tunes=False,
            rag_runner=rag_runner,
            algorithm_params=sample_algorithm_params,
            optimization_metric_id="answer_correctness",
            search_space=sample_search_space,
            tune_dataset=sample_dataset_id,
        )

        # Mock only the HuggingFace dataset loading
        with patch(
            "rag_hpo_bench.hpo.rag_runner.load_rag_configurations_summary",
            return_value=sample_hf_dataframe,
        ):
            # Execute - this will use actual GridHPO algorithm
            result = tuner.run()

        # Verify result is HpoResults
        assert result is not None
        assert isinstance(result, HpoResults)

        # Verify all combinations were tested (2 chunk_size × 2 chunk_overlap × 2 top_k = 8)
        assert result.size() == 8

        # Verify search space was serialized
        search_space_file = output_path / "search_space.json"
        assert search_space_file.exists()

        # Verify HPO results were saved to CSV
        hpo_results_file = output_path / "hpo_results.csv"
        assert hpo_results_file.exists()

        # Verify CSV content
        df = pd.read_csv(hpo_results_file)
        assert len(df) == 8
        # Check for actual metric column names from the HuggingFace dataset
        assert "Lexical-AC_mean" in df.columns
        assert "Lexical-FF_mean" in df.columns
        assert "LLMaaJ-AC_mean" in df.columns
        assert "context_correctness_mean" in df.columns

        # Verify all parameter combinations are present
        assert set(df["data_pipeline.params.indexing.chunk_size"].unique()) == {512, 1024}
        # Check that we have 2 unique overlap ratios (0.25 and 0.5)
        assert set(df["data_pipeline.params.indexing.chunk_overlap"].unique()) == {0.25, 0.5}
        assert set(df["inference_pipeline.params.retrieval.top_k"].unique()) == {5, 10}

        # Verify optimization_metric_id was added to algorithm params
        assert "optimization_metric_id" in df.columns
        assert all(df["optimization_metric_id"] == "answer_correctness")

        # Verify pattern results files were created
        for i in range(8):
            pattern_file = output_path / f"Pattern_{i}.csv"
            assert pattern_file.exists()

        # Verify metrics are present and valid
        assert all(df["Lexical-AC_mean"] > 0)
        assert all(df["Lexical-FF_mean"] > 0)
        assert all(df["context_correctness_mean"] > 0)

        # Verify best config can be retrieved using one of the actual metric IDs
        best_configs = result.get_best_configs(
            metric_id="context_correctness", num_best_configs_to_consider=1
        )
        assert len(best_configs) == 1
        assert best_configs[0] is not None

        # Verify the best config has the highest chunk_size and top_k
        # (based on our synthetic data formula)
        best_params = best_configs[0].get_path_to_values_dict()
        assert best_params["data_pipeline.params.indexing.chunk_size"] == 1024
        assert best_params["inference_pipeline.params.retrieval.top_k"] == 10

    def test_run_with_optimization_metric_id(
        self,
        tmp_path,
        sample_hf_dataframe,
        sample_search_space,
        sample_algorithm_params,
        sample_dataset_id,
    ):
        """Test that optimization_metric_id is properly used."""
        output_path = tmp_path / "tuner_output"
        rag_runner = RagRunner()

        # Create tuner with optimization_metric_id
        tuner = Tuner(
            output_path=output_path,
            skip_existing_tunes=False,
            rag_runner=rag_runner,
            algorithm_params=sample_algorithm_params,
            optimization_metric_id="faithfulness",
            search_space=sample_search_space,
            tune_dataset=sample_dataset_id,
        )

        # Mock the HuggingFace dataset loading
        with patch(
            "rag_hpo_bench.hpo.rag_runner.load_rag_configurations_summary",
            return_value=sample_hf_dataframe,
        ):
            result = tuner.run()

        # Verify optimization_metric_id is in algorithm_params
        assert tuner.algorithm_params["optimization_metric_id"] == "faithfulness"

        # Verify results were generated
        assert result is not None
        assert isinstance(result, HpoResults)

"""
Tests for ExperimentsRunner with empty optimization metrics.
"""

from pathlib import Path

import pytest

from rag_hpo_bench.data_models import DatasetID, DatasetName
from rag_hpo_bench.hpo import AlgorithmConfig, ExperimentsRunner, TuneAndTestDataset
from rag_hpo_bench.hpo.hpo_algorithm import HpoAlgorithmType
from rag_hpo_bench.hpo.hpo_experiment import HpoExperiment
from rag_hpo_bench.hpo.search_space import SearchSpace, SearchSpaceParameter


@pytest.fixture
def simple_search_space():
    """Create a simple search space for testing."""
    return SearchSpace(
        parameters=[
            SearchSpaceParameter(
                path=["data_pipeline", "params", "indexing", "chunk_size"],
                values=[256, 512],
            ),
        ]
    )


@pytest.fixture
def sample_dataset_pair():
    """Create a sample dataset pair."""
    return TuneAndTestDataset(
        tune=DatasetID(dataset_name=DatasetName.AIArxiv, split="Test"),
        test=None,
    )


def test_empty_optimization_metrics_with_grid(simple_search_space, sample_dataset_pair):
    """Test that empty optimization_metrics works with grid search."""
    algorithm_configs = [AlgorithmConfig(algorithm_type="grid")]
    optimization_metrics = []

    runner = ExperimentsRunner(
        search_space=simple_search_space,
        dataset_pairs=[sample_dataset_pair],
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=Path("./test_output"),
    )

    # Should create one experiment
    assert len(runner.hpo_experiments) == 1

    # Check that experiment has empty optimization_metric_id
    exp = runner.hpo_experiments[0]
    assert exp.optimization_metric_id == ""


def test_empty_optimization_metrics_with_non_grid_fails(simple_search_space, sample_dataset_pair):
    """Test that empty optimization_metrics fails with non-grid algorithms."""
    algorithm_configs = [
        AlgorithmConfig(
            algorithm_type="random", num_seeds=5, additional_params={"max_iterations": 10}
        )
    ]
    optimization_metrics = []

    with pytest.raises(
        ValueError, match="Empty optimization_metrics list is only allowed for grid search"
    ):
        ExperimentsRunner(
            search_space=simple_search_space,
            dataset_pairs=[sample_dataset_pair],
            algorithm_configs=algorithm_configs,
            optimization_metrics=optimization_metrics,
            output_path=Path("./test_output"),
        )


def test_empty_optimization_metrics_with_multiple_algorithms_fails(
    simple_search_space, sample_dataset_pair
):
    """Test that empty optimization_metrics fails when any algorithm is not grid."""
    algorithm_configs = [
        AlgorithmConfig(algorithm_type="grid"),
        AlgorithmConfig(
            algorithm_type="random", num_seeds=5, additional_params={"max_iterations": 10}
        ),
    ]
    optimization_metrics = []

    with pytest.raises(
        ValueError, match="Empty optimization_metrics list is only allowed for grid search"
    ):
        ExperimentsRunner(
            search_space=simple_search_space,
            dataset_pairs=[sample_dataset_pair],
            algorithm_configs=algorithm_configs,
            optimization_metrics=optimization_metrics,
            output_path=Path("./test_output"),
        )


def test_non_empty_optimization_metrics_works_with_any_algorithm(
    simple_search_space, sample_dataset_pair
):
    """Test that non-empty optimization_metrics works with any algorithm."""
    algorithm_configs = [
        AlgorithmConfig(algorithm_type="grid"),
        AlgorithmConfig(
            algorithm_type="random", num_seeds=5, additional_params={"max_iterations": 10}
        ),
    ]
    optimization_metrics = ["Lexical-AC"]

    runner = ExperimentsRunner(
        search_space=simple_search_space,
        dataset_pairs=[sample_dataset_pair],
        algorithm_configs=algorithm_configs,
        optimization_metrics=optimization_metrics,
        output_path=Path("./test_output"),
    )

    # Should create 2 experiments (1 dataset * 2 algorithms * 1 metric)
    assert len(runner.hpo_experiments) == 2

    # All experiments should have the optimization metric
    for exp in runner.hpo_experiments:
        assert exp.optimization_metric_id == "Lexical-AC"


def test_output_path_without_optimization_metric():
    """Test that output path doesn't include metric when optimization_metric_id is empty."""
    tune_dataset = DatasetID(dataset_name=DatasetName.AIArxiv, split="Test")

    # Test with empty optimization_metric_id
    output_path = HpoExperiment.get_output_path(
        base_output_path=Path("./test_output"),
        algorithm_type=HpoAlgorithmType.GRID,
        tune_dataset=tune_dataset,
        optimization_metric_id="",
    )

    expected_path = Path("./test_output/grid/name-AIArxiv_split-Test")
    assert output_path == expected_path


def test_output_path_with_optimization_metric():
    """Test that output path includes metric when optimization_metric_id is provided."""
    tune_dataset = DatasetID(dataset_name=DatasetName.AIArxiv, split="Test")

    output_path = HpoExperiment.get_output_path(
        base_output_path=Path("./test_output"),
        algorithm_type=HpoAlgorithmType.GRID,
        tune_dataset=tune_dataset,
        optimization_metric_id="Lexical-AC",
    )

    expected_path = Path("./test_output/grid/Lexical-AC/name-AIArxiv_split-Test")
    assert output_path == expected_path


def test_output_path_with_test_dataset():
    """Test that output path includes test dataset when provided."""
    tune_dataset = DatasetID(dataset_name=DatasetName.AIArxiv, split="Dev")
    test_dataset = DatasetID(dataset_name=DatasetName.AIArxiv, split="Test")

    # With optimization metric
    output_path = HpoExperiment.get_output_path(
        base_output_path=Path("./test_output"),
        algorithm_type=HpoAlgorithmType.GRID,
        tune_dataset=tune_dataset,
        optimization_metric_id="Lexical-AC",
        test_dataset=test_dataset,
    )

    expected_path = Path(
        "./test_output/grid/Lexical-AC/name-AIArxiv_split-Dev/name-AIArxiv_split-Test"
    )
    assert output_path == expected_path

    # Without optimization metric
    output_path_no_metric = HpoExperiment.get_output_path(
        base_output_path=Path("./test_output"),
        algorithm_type=HpoAlgorithmType.GRID,
        tune_dataset=tune_dataset,
        optimization_metric_id="",
        test_dataset=test_dataset,
    )

    expected_path_no_metric = Path(
        "./test_output/grid/name-AIArxiv_split-Dev/name-AIArxiv_split-Test"
    )
    assert output_path_no_metric == expected_path_no_metric

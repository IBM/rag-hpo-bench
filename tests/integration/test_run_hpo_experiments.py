"""
Integration test for run_hpo_experiments.py main() function.

This test verifies the end-to-end HPO experiment pipeline including:
1. Running HPO experiments with tune/test splits
2. Running grid search on test sets
3. Analyzing and summarizing results
"""

import shutil

import pytest

from rag_hpo_bench.hpo.run_hpo_experiments import main


@pytest.fixture
def test_output_path(tmp_path):
    """Create a temporary output directory for test results."""
    output_dir = tmp_path / "integration_test_output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_main_function_full_pipeline(test_output_path):
    """
    Integration test that runs the complete HPO experiment pipeline.

    This test runs the actual main() function to verify the entire
    pipeline works end-to-end:
    1. HPO experiments with tune/test splits
    2. Grid search on test sets
    3. Analysis and result summarization

    This test runs all experiments to ensure complete coverage.
    """
    # Run the full pipeline with all experiments
    main(
        output_path=test_output_path,
        max_experiments=None,  # Run all experiments
        clean_output=True,
    )

    # Verify output directory structure was created
    assert test_output_path.exists(), "Output directory should exist"

    # Verify that at least one algorithm directory was created
    # (grid search is always run)
    grid_dir = test_output_path / "grid"
    assert grid_dir.exists(), "Grid search directory should exist"

    # Verify that some results were generated
    # The exact structure depends on the datasets and algorithms,
    # but we should have at least some subdirectories
    subdirs = list(test_output_path.iterdir())
    assert len(subdirs) > 0, "Should have created at least one subdirectory"

    # Verify analysis directory was created
    analysis_dir = test_output_path / "analysis"
    assert analysis_dir.exists(), "Analysis directory should exist"

    # Verify that analysis results were generated
    analysis_subdirs = list(analysis_dir.iterdir())
    assert len(analysis_subdirs) > 0, "Analysis should have generated results"

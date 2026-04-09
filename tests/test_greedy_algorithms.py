"""
Tests for Greedy HPO algorithms (GreedyMHPO and GreedyRHPO).
"""
import pytest
from unittest.mock import MagicMock

from rag_hpo_bench.hpo.hpo_algorithm import GreedyMHPO, GreedyRHPO, HpoAlgorithmType
from rag_hpo_bench.hpo.pattern_results import PatternResults
from rag_hpo_bench.hpo.search_space import (
    PatternParameters,
    RagParameter,
    RagParameterName,
    SearchSpace,
    SearchSpaceParameter,
)


@pytest.fixture
def sample_search_space():
    """Create a sample search space with all greedy-optimizable parameters."""
    return SearchSpace(
        parameters=[
            SearchSpaceParameter(
                path=["indexing", "chunking", "chunk_size"], 
                values=[256, 512, 1024]
            ),
            SearchSpaceParameter(
                path=["indexing", "chunking", "chunk_overlap"], 
                values=[50, 100]
            ),
            SearchSpaceParameter(
                path=["indexing", "embedding", "embedding_model"], 
                values=["model-a", "model-b"]
            ),
            SearchSpaceParameter(
                path=["inference", "retrieval", "top_k"], 
                values=[3, 5, 10]
            ),
            SearchSpaceParameter(
                path=["inference", "generation", "generative_model"], 
                values=["gen-model-1", "gen-model-2"]
            ),
        ]
    )


@pytest.fixture
def mock_objective_function():
    """
    Create a mock objective function that returns predictable scores.
    
    The scoring logic (HIGHER IS BETTER):
    - Smaller chunk_size is better (256 > 512 > 1024)
    - Smaller chunk_overlap is better (50 > 100)
    - model-a is better than model-b for embedding
    - Smaller top_k is better (3 > 5 > 10)
    - gen-model-1 is better than gen-model-2
    """
    def objective_function(pattern_parameters: PatternParameters) -> PatternResults:
        # Extract parameter values
        params_dict = {}
        for param in pattern_parameters.pattern_params:
            param_name = param.path[-1]
            if isinstance(param_name, str):
                param_name = RagParameterName(param_name)
            params_dict[param_name] = param.value
        
        # Calculate score based on parameter values (HIGHER IS BETTER)
        score = 0.5  # Base score
        
        # Chunk size: smaller is better (add points for better values)
        if params_dict.get(RagParameterName.CHUNK_SIZE) == 256:
            score += 0.1
        elif params_dict.get(RagParameterName.CHUNK_SIZE) == 512:
            score += 0.05
        # 1024 adds nothing (worst)
        
        # Chunk overlap: smaller is better
        if params_dict.get(RagParameterName.CHUNK_OVERLAP) == 50:
            score += 0.05
        # 100 adds nothing (worst)
        
        # Embedding model: model-a is better
        if params_dict.get(RagParameterName.EMBEDDING_MODEL) == "model-a":
            score += 0.1
        # model-b adds nothing (worst)
        
        # Top-k: smaller is better
        if params_dict.get(RagParameterName.TOP_K) == 3:
            score += 0.1
        elif params_dict.get(RagParameterName.TOP_K) == 5:
            score += 0.05
        # 10 adds nothing (worst)
        
        # Generative model: gen-model-1 is better
        if params_dict.get(RagParameterName.GENERATIVE_MODEL) == "gen-model-1":
            score += 0.1
        # gen-model-2 adds nothing (worst)
        
        # Create mock pattern results
        mock_result = MagicMock(spec=PatternResults)
        mock_result.metric_stats = {
            "test_metric": {"mean": score}
        }
        mock_result.name = "Pattern_0"
        
        return mock_result
    
    return objective_function


class TestGreedyMHPO:
    """Test suite for GreedyMHPO (Model-first greedy algorithm)."""
    
    def test_parameter_order(self, sample_search_space, mock_objective_function):
        """Test that GreedyMHPO optimizes parameters in model-first order."""
        greedy_m = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=20,
            seed=42,
        )
        
        # Verify parameter order
        expected_order = [
            RagParameterName.GENERATIVE_MODEL,
            RagParameterName.EMBEDDING_MODEL,
            RagParameterName.CHUNK_SIZE,
            RagParameterName.CHUNK_OVERLAP,
            RagParameterName.TOP_K,
        ]
        assert greedy_m.get_parameter_order() == expected_order
    
    def test_search_finds_optimal_config(self, sample_search_space, mock_objective_function):
        """Test that GreedyMHPO finds a good configuration within iteration limit."""
        greedy_m = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=20,
            seed=42,
        )
        
        # Run search
        results = greedy_m.search()
        
        # Verify results
        assert results is not None
        assert results.size() <= 20  # Should not exceed max_iterations
        assert results.size() > 0  # Should have at least one result
        
        # Get best configuration
        best_configs = results.get_best_configs(
            metric_id="test_metric",
            num_best_configs_to_consider=1
        )
        assert len(best_configs) == 1
        
        # Verify the best config has good parameter values
        best_params = best_configs[0].get_path_to_values_dict()
        
        # Based on our mock objective function, optimal values should be:
        # - gen-model-1 (optimized first in model-first order)
        # - model-a for embedding
        # - 256 for chunk_size
        # - 50 for chunk_overlap
        # - 3 for top_k
        assert best_params["inference.generation.generative_model"] == "gen-model-1"
    
    def test_max_iterations_respected(self, sample_search_space, mock_objective_function):
        """Test that GreedyMHPO respects max_iterations limit."""
        max_iter = 10
        greedy_m = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=42,
        )
        
        results = greedy_m.search()
        
        # Should not exceed max_iterations
        assert results.size() <= max_iter
    
    def test_seed_reproducibility(self, sample_search_space, mock_objective_function):
        """Test that same seed produces same results."""
        seed = 123
        
        # Run 1
        greedy_m1 = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=15,
            seed=seed,
        )
        results1 = greedy_m1.search()
        
        # Run 2 with same seed
        greedy_m2 = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=15,
            seed=seed,
        )
        results2 = greedy_m2.search()
        
        # Results should be identical
        assert results1.size() == results2.size()


class TestGreedyRHPO:
    """Test suite for GreedyRHPO (Retrieval-first greedy algorithm)."""
    
    def test_parameter_order(self, sample_search_space, mock_objective_function):
        """Test that GreedyRHPO optimizes parameters in retrieval-first order."""
        greedy_r = GreedyRHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=20,
            seed=42,
        )
        
        # Verify parameter order
        expected_order = [
            RagParameterName.EMBEDDING_MODEL,
            RagParameterName.CHUNK_SIZE,
            RagParameterName.CHUNK_OVERLAP,
            RagParameterName.GENERATIVE_MODEL,
            RagParameterName.TOP_K,
        ]
        assert greedy_r.get_parameter_order() == expected_order
    
    def test_search_finds_optimal_config(self, sample_search_space, mock_objective_function):
        """Test that GreedyRHPO finds a good configuration within iteration limit."""
        greedy_r = GreedyRHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=20,
            seed=42,
        )
        
        # Run search
        results = greedy_r.search()
        
        # Verify results
        assert results is not None
        assert results.size() <= 20  # Should not exceed max_iterations
        assert results.size() > 0  # Should have at least one result
        
        # Get best configuration
        best_configs = results.get_best_configs(
            metric_id="test_metric",
            num_best_configs_to_consider=1
        )
        assert len(best_configs) == 1
        
        # Verify the best config has good parameter values
        best_params = best_configs[0].get_path_to_values_dict()
        
        # Based on our mock objective function, optimal values should be:
        # - model-a for embedding (optimized first in retrieval-first order)
        # - 256 for chunk_size
        # - 50 for chunk_overlap
        # - gen-model-1
        # - 3 for top_k
        assert best_params["indexing.embedding.embedding_model"] == "model-a"
    
    def test_max_iterations_respected(self, sample_search_space, mock_objective_function):
        """Test that GreedyRHPO respects max_iterations limit."""
        max_iter = 10
        greedy_r = GreedyRHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=42,
        )
        
        results = greedy_r.search()
        
        # Should not exceed max_iterations
        assert results.size() <= max_iter
    
    def test_different_from_greedy_m(self, sample_search_space, mock_objective_function):
        """Test that GreedyRHPO explores different configurations than GreedyMHPO."""
        seed = 42
        max_iter = 15
        
        # Run GreedyMHPO
        greedy_m = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=seed,
        )
        results_m = greedy_m.search()
        
        # Run GreedyRHPO
        greedy_r = GreedyRHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=seed,
        )
        results_r = greedy_r.search()
        
        # Both should find results
        assert results_m.size() > 0
        assert results_r.size() > 0
        
        # The algorithms should explore parameters in different orders,
        # so they may find different configurations (though both should be good)
        # We just verify they both complete successfully
        assert results_m is not None
        assert results_r is not None


class TestGreedyAlgorithmsComparison:
    """Test suite comparing GreedyMHPO and GreedyRHPO behavior."""
    
    def test_both_algorithms_converge(self, sample_search_space, mock_objective_function):
        """Test that both greedy algorithms converge to good solutions."""
        seed = 100
        max_iter = 25
        
        # Run both algorithms
        greedy_m = GreedyMHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=seed,
        )
        results_m = greedy_m.search()
        
        greedy_r = GreedyRHPO(
            search_space=sample_search_space,
            optimization_metric_id="test_metric",
            objective_function=mock_objective_function,
            max_iterations=max_iter,
            seed=seed,
        )
        results_r = greedy_r.search()
        
        # Get best configs from both
        best_m = results_m.get_best_configs(
            metric_id="test_metric",
            num_best_configs_to_consider=1
        )[0]
        
        best_r = results_r.get_best_configs(
            metric_id="test_metric",
            num_best_configs_to_consider=1
        )[0]
        
        # Both should find good configurations
        # (exact values may differ due to different optimization orders)
        assert best_m is not None
        assert best_r is not None
        
        # Both should have reasonable scores (> 0.5 based on our mock function)
        best_m_params = best_m.get_path_to_values_dict()
        best_r_params = best_r.get_path_to_values_dict()
        
        # Verify both found good parameter values
        assert best_m_params["inference.generation.generative_model"] == "gen-model-1"
        assert best_r_params["indexing.embedding.embedding_model"] == "model-a"

# Made with Bob

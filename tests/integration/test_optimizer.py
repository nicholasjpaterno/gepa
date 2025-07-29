"""Integration tests for GEPA optimizer."""

import pytest
from unittest.mock import AsyncMock, patch
from typing import List, Dict, Any

from gepa.core.optimizer import GEPAOptimizer, OptimizationResult
from gepa.config import GEPAConfig
from gepa.core.system import CompoundAISystem
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score


@pytest.mark.integration
class TestGEPAOptimizer:
    """Integration tests for GEPA optimizer."""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(
        self,
        sample_config: GEPAConfig,
        sample_evaluator: SimpleEvaluator,
        mock_inference_client
    ):
        """Test optimizer initialization."""
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=mock_inference_client
        )
        
        assert optimizer.config == sample_config
        assert optimizer.evaluator == sample_evaluator
        assert optimizer.inference_client == mock_inference_client
        assert optimizer.rollouts_used == 0
        assert optimizer.total_cost == 0.0
        assert len(optimizer.optimization_history) == 0
    
    @pytest.mark.asyncio
    async def test_basic_optimization_run(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test a basic optimization run."""
        # Use smaller budget for faster testing
        sample_config.optimization.budget = 10
        sample_config.optimization.pareto_set_size = 3
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        result = await optimizer.optimize(sample_system, sample_dataset)
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert result.best_system is not None
        assert result.best_score >= 0
        assert result.total_rollouts > 0
        assert result.total_rollouts <= sample_config.optimization.budget
        assert result.total_cost >= 0
        assert result.pareto_frontier.size() > 0
    
    @pytest.mark.asyncio
    async def test_optimization_with_limited_generations(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test optimization with generation limit."""
        sample_config.optimization.budget = 50
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        result = await optimizer.optimize(
            sample_system, 
            sample_dataset,
            max_generations=3
        )
        
        assert result.total_rollouts > 0
        # Should stop after 3 generations regardless of budget
        assert len(optimizer.optimization_history) <= 3 * 2  # Up to 2 candidates per generation
    
    @pytest.mark.asyncio
    async def test_optimization_budget_enforcement(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test that optimization respects budget limits."""
        budget = 15
        sample_config.optimization.budget = budget
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        result = await optimizer.optimize(sample_system, sample_dataset)
        
        assert result.total_rollouts <= budget
        assert optimizer.rollouts_used <= budget
    
    @pytest.mark.asyncio
    async def test_pareto_frontier_evolution(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test that Pareto frontier evolves during optimization."""
        sample_config.optimization.budget = 20
        sample_config.optimization.pareto_set_size = 5
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        # Track frontier size changes
        initial_frontier_size = optimizer.pareto_frontier.size()
        
        result = await optimizer.optimize(sample_system, sample_dataset)
        
        final_frontier_size = result.pareto_frontier.size()
        
        assert final_frontier_size > initial_frontier_size
        assert final_frontier_size <= sample_config.optimization.pareto_set_size
    
    @pytest.mark.asyncio
    async def test_crossover_enabled(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test optimization with crossover enabled."""
        sample_config.optimization.enable_crossover = True
        sample_config.optimization.crossover_probability = 0.5
        sample_config.optimization.budget = 15
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        with patch.object(optimizer, '_perform_crossover') as mock_crossover:
            mock_crossover.return_value = sample_system
            
            result = await optimizer.optimize(sample_system, sample_dataset)
            
            # Crossover might be called (depending on random probability)
            assert result.total_rollouts > 0
    
    @pytest.mark.asyncio
    async def test_crossover_disabled(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test optimization with crossover disabled."""
        sample_config.optimization.enable_crossover = False
        sample_config.optimization.budget = 15
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        with patch.object(optimizer, '_perform_crossover') as mock_crossover:
            result = await optimizer.optimize(sample_system, sample_dataset)
            
            # Crossover should never be called
            mock_crossover.assert_not_called()
            assert result.total_rollouts > 0
    
    @pytest.mark.asyncio
    async def test_mutation_types(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test that different mutation types are used."""
        sample_config.optimization.mutation_types = ["rewrite", "insert"]
        sample_config.optimization.budget = 15
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client,
            reflection_client=test_inference_client
        )
        
        with patch.object(optimizer.mutator, 'mutate_prompt') as mock_mutate:
            mock_mutate.return_value = "Mutated prompt"
            
            result = await optimizer.optimize(sample_system, sample_dataset)
            
            # Mutation should be called
            assert mock_mutate.call_count > 0
            assert result.total_rollouts > 0
    
    @pytest.mark.asyncio
    async def test_optimization_statistics(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test optimization statistics collection."""
        sample_config.optimization.budget = 15
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        result = await optimizer.optimize(sample_system, sample_dataset)
        
        stats = optimizer.get_statistics()
        
        assert "rollouts_used" in stats
        assert "total_cost" in stats
        assert "pareto_frontier" in stats
        assert "generations" in stats
        assert "successful_mutations" in stats
        
        assert stats["rollouts_used"] == result.total_rollouts
        assert stats["total_cost"] == result.total_cost
        assert stats["generations"] > 0
    
    @pytest.mark.asyncio
    async def test_empty_dataset_handling(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test optimization with empty dataset."""
        sample_config.optimization.budget = 5
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        result = await optimizer.optimize(sample_system, [])
        
        # Should handle empty dataset gracefully
        assert result.best_system == sample_system  # Original system returned
        assert result.total_rollouts == 0
    
    @pytest.mark.asyncio
    async def test_early_stopping(
        self,
        sample_config: GEPAConfig,
        sample_system: CompoundAISystem,
        sample_dataset: List[Dict[str, Any]],
        sample_evaluator: SimpleEvaluator,
        test_inference_client
    ):
        """Test early stopping mechanism."""
        sample_config.optimization.budget = 50  # Large budget
        
        optimizer = GEPAOptimizer(
            config=sample_config,
            evaluator=sample_evaluator,
            inference_client=test_inference_client
        )
        
        # Mock should_stop_early to return True after a few generations
        original_should_stop = optimizer._should_stop_early
        call_count = 0
        
        def mock_should_stop_early():
            nonlocal call_count
            call_count += 1
            return call_count > 3  # Stop after 3 calls
        
        optimizer._should_stop_early = mock_should_stop_early
        
        result = await optimizer.optimize(sample_system, sample_dataset)
        
        # Should stop early, using less than full budget
        assert result.total_rollouts < sample_config.optimization.budget


@pytest.mark.integration
class TestOptimizationResult:
    """Test OptimizationResult data structure."""
    
    def test_optimization_result_creation(
        self,
        sample_system: CompoundAISystem,
        sample_candidates
    ):
        """Test OptimizationResult creation and access."""
        from gepa.core.pareto import ParetoFrontier
        
        frontier = ParetoFrontier(max_size=5)
        for candidate in sample_candidates[:3]:
            frontier.add_candidate(candidate)
        
        result = OptimizationResult(
            best_system=sample_system,
            best_score=0.85,
            pareto_frontier=frontier,
            total_rollouts=25,
            total_cost=0.15,
            optimization_history=[
                {"generation": 1, "best_score": 0.7},
                {"generation": 2, "best_score": 0.8},
                {"generation": 3, "best_score": 0.85}
            ]
        )
        
        assert result.best_system == sample_system
        assert result.best_score == 0.85
        assert result.pareto_frontier == frontier
        assert result.total_rollouts == 25
        assert result.total_cost == 0.15
        assert len(result.optimization_history) == 3
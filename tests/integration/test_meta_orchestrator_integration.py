"""Integration tests for MetaOrchestrator with real components."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.gepa.meta_orchestrator import (
    MetaOrchestrator,
    MetaOrchestratorConfig,
    ConfigProfiles
)
from src.gepa.meta_orchestrator.config import OptimizationMode, BudgetStrategy
from src.gepa.core.system import CompoundAISystem
from src.gepa.evaluation.base import Evaluator


class TestMetaOrchestratorIntegration:
    """Integration tests for MetaOrchestrator with real components."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return [
            {
                "input": "Summarize the key benefits of renewable energy.",
                "expected": "Renewable energy offers environmental benefits, cost savings, and energy independence."
            },
            {
                "input": "Explain the concept of machine learning in simple terms.",
                "expected": "Machine learning is a method where computers learn patterns from data to make predictions."
            },
            {
                "input": "What are the main components of a computer?",
                "expected": "The main components include CPU, memory, storage, and input/output devices."
            }
        ]
    
    @pytest.fixture
    def mock_system(self):
        """Create a mock compound AI system."""
        system = Mock(spec=CompoundAISystem)
        system.modules = {
            "analyzer": Mock(),
            "generator": Mock(),
            "validator": Mock()
        }
        system.get_module_prompts.return_value = {
            "analyzer": "Analyze the following input carefully and extract key information.",
            "generator": "Generate a comprehensive response based on the analysis.",
            "validator": "Validate the response for accuracy and completeness."
        }
        system.update_module.return_value = system
        return system
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator that simulates realistic performance."""
        evaluator = Mock(spec=Evaluator)
        
        # Simulate variable performance with slight improvements over time
        performance_sequence = [0.65, 0.67, 0.72, 0.68, 0.74, 0.76, 0.73, 0.78, 0.75, 0.80]
        call_count = 0
        
        async def mock_evaluate(system, dataset, **kwargs):
            nonlocal call_count
            score = performance_sequence[call_count % len(performance_sequence)]
            call_count += 1
            
            return {
                "score": score,
                "metrics": {
                    "accuracy": score,
                    "fluency": score + 0.05,
                    "relevance": score - 0.02
                },
                "individual_scores": [score + i * 0.01 for i in range(len(dataset))]
            }
        
        evaluator.evaluate = mock_evaluate
        return evaluator
    
    @pytest.fixture
    def mock_inference_client(self):
        """Create a mock inference client."""
        client = Mock()
        
        responses = [
            "This is a comprehensive analysis of the input.",
            "Based on the analysis, here's a detailed response.",
            "The response has been validated and is accurate."
        ]
        
        async def mock_generate(prompt, **kwargs):
            return responses[hash(prompt) % len(responses)]
        
        client.generate = mock_generate
        return client


class TestConfigurationLoading:
    """Test configuration loading from various sources."""
    
    def test_load_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "enabled": True,
            "optimization_mode": "exploration",
            "max_optimization_rounds": 150,
            "total_compute_budget": 300.0,
            "rl_config": {
                "learning_rate": 0.001,
                "buffer_capacity": 5000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = MetaOrchestratorConfig.from_file(config_file)
            
            assert config.enabled is True
            assert config.optimization_mode == OptimizationMode.EXPLORATION
            assert config.max_optimization_rounds == 150
            assert config.total_compute_budget == 300.0
            assert config.rl_config.learning_rate == 0.001
            assert config.rl_config.buffer_capacity == 5000
        finally:
            Path(config_file).unlink()
    
    def test_load_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        config_yaml = """
        enabled: true
        optimization_mode: balanced
        budget_allocation_strategy: adaptive
        max_optimization_rounds: 200
        total_compute_budget: 400.0
        detailed_logging: false
        topology_config:
          mutation_rate: 0.4
          crossover_rate: 0.8
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_yaml)
            config_file = f.name
        
        try:
            config = MetaOrchestratorConfig.from_file(config_file)
            
            assert config.enabled is True
            assert config.optimization_mode == OptimizationMode.BALANCED
            assert config.budget_allocation_strategy == BudgetStrategy.ADAPTIVE
            assert config.max_optimization_rounds == 200
            assert config.total_compute_budget == 400.0
            assert config.detailed_logging is False
            assert config.topology_config.mutation_rate == 0.4
            assert config.topology_config.crossover_rate == 0.8
        finally:
            Path(config_file).unlink()
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = ConfigProfiles.get_profile("development")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # Save as JSON
            config.to_file(config_file, format="json")
            assert Path(config_file).exists()
            
            # Load back and verify
            loaded_config = MetaOrchestratorConfig.from_file(config_file)
            assert loaded_config.optimization_mode == config.optimization_mode
            assert loaded_config.max_optimization_rounds == config.max_optimization_rounds
        finally:
            Path(config_file).unlink()
    
    @patch.dict('os.environ', {
        'GEPA_META_ENABLED': 'true',
        'GEPA_META_MODE': 'aggressive',
        'GEPA_META_MAX_ROUNDS': '500',
        'GEPA_META_COMPUTE_BUDGET': '1000.0',
        'GEPA_META_DETAILED_LOGGING': 'false'
    })
    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        config = MetaOrchestratorConfig.from_env("GEPA_META")
        
        assert config.enabled is True
        assert config.optimization_mode == OptimizationMode.AGGRESSIVE
        assert config.max_optimization_rounds == 500
        assert config.total_compute_budget == 1000.0
        assert config.detailed_logging is False


class TestMetaOrchestratorWorkflow:
    """Test complete MetaOrchestrator workflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_basic_optimization_workflow(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test basic optimization workflow with minimal configuration."""
        config = ConfigProfiles.get_profile("minimal")
        config.max_optimization_rounds = 2  # Very short for testing to avoid timeouts
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=5  # Reduced budget for faster testing
        )
        
        assert isinstance(result, dict)
        assert "best_score" in result
        assert "generations" in result
    
    @pytest.mark.asyncio
    async def test_development_profile_workflow(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test optimization with development profile."""
        config = ConfigProfiles.get_profile("development")
        config.max_optimization_rounds = 5  # Reduced for testing
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=50
        )
        
        assert isinstance(result, dict)
        # Development profile should enable detailed logging
        assert config.detailed_logging is True
        assert config.checkpoint_enabled is True
    
    @pytest.mark.asyncio
    async def test_conservative_profile_workflow(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test optimization with conservative profile (resource-constrained)."""
        config = ConfigProfiles.get_profile("conservative")
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=30
        )
        
        assert isinstance(result, dict)
        # Conservative profile should use fewer resources
        assert config.max_parallel_components == 2
        assert config.optimization_mode == OptimizationMode.CONSERVATIVE
    
    @pytest.mark.asyncio
    async def test_research_profile_workflow(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test optimization with research profile (exploration-focused)."""
        config = ConfigProfiles.get_profile("research")
        config.max_optimization_rounds = 10  # Reduced for testing
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=200
        )
        
        assert isinstance(result, dict)
        # Research profile should focus on exploration
        assert config.optimization_mode == OptimizationMode.EXPLORATION
        assert config.component_metrics is True
        assert config.export_metrics is True


class TestComponentInteraction:
    """Test interactions between MetaOrchestrator components."""
    
    @pytest.mark.asyncio
    async def test_rl_selector_and_hyperopt_coordination(self, mock_system, mock_evaluator, mock_inference_client):
        """Test coordination between RL selector and hyperparameter optimizer."""
        config = ConfigProfiles.get_profile("minimal")
        config.max_optimization_rounds = 3
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        # Test algorithm selection
        mock_state = Mock()
        mock_state.encode.return_value = [0.5] * 13
        mock_state.budget_remaining = 20
        
        algorithm, value_estimate = await orchestrator._select_algorithm(mock_state)
        
        assert algorithm in orchestrator.available_algorithms
        assert isinstance(value_estimate, (int, float))
        
        # Test hyperparameter suggestion for selected algorithm
        hyperparams, fidelity = await orchestrator.hyperopt.suggest_hyperparameters(
            algorithm, mock_state
        )
        
        assert isinstance(hyperparams, dict)
        assert fidelity in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_prompt_evolution_integration(self, mock_system, mock_evaluator, mock_inference_client):
        """Test prompt evolution integration with other components."""
        config = ConfigProfiles.get_profile("minimal")
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        # Mock prompts and performance feedback
        current_prompts = mock_system.get_module_prompts()
        performance_feedback = [0.7, 0.8, 0.6, 0.9, 0.75]
        
        evolved_structure = await orchestrator.prompt_evolver.evolve_prompt_structure(
            current_prompts, performance_feedback
        )
        
        # Evolution might or might not occur based on thresholds
        if evolved_structure:
            assert "evolution_type" in evolved_structure
            assert "updated_prompts" in evolved_structure
            assert "fitness_score" in evolved_structure
    
    def test_resource_allocation_coordination(self, mock_evaluator, mock_inference_client):
        """Test resource allocation based on optimization mode."""
        modes_to_test = [
            OptimizationMode.EXPLORATION,
            OptimizationMode.EXPLOITATION, 
            OptimizationMode.BALANCED,
            OptimizationMode.CONSERVATIVE,
            OptimizationMode.AGGRESSIVE
        ]
        
        for mode in modes_to_test:
            config = MetaOrchestratorConfig(
                enabled=True,
                optimization_mode=mode
            )
            
            orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
            
            allocation = config.get_resource_allocation()
            
            # All allocations should sum to 1.0
            assert abs(sum(allocation.values()) - 1.0) < 0.01
            
            # Check mode-specific allocations
            if mode == OptimizationMode.EXPLORATION:
                assert allocation["topology_evolver"] > allocation["hyperopt"]
            elif mode == OptimizationMode.EXPLOITATION:
                assert allocation["hyperopt"] >= allocation["topology_evolver"]


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.asyncio
    async def test_performance_with_increasing_complexity(self, mock_evaluator, mock_inference_client):
        """Test performance with increasing system complexity."""
        dataset_sizes = [1, 3, 5]  # Small datasets for testing
        
        for dataset_size in dataset_sizes:
            dataset = [
                {"input": f"Test input {i}", "expected": f"Expected output {i}"}
                for i in range(dataset_size)
            ]
            
            # Mock system with multiple modules
            mock_system = Mock(spec=CompoundAISystem)
            mock_system.modules = {f"module_{i}": Mock() for i in range(dataset_size)}
            mock_system.get_module_prompts.return_value = {
                f"module_{i}": f"Prompt for module {i}" for i in range(dataset_size)
            }
            mock_system.update_module.return_value = mock_system
            
            config = ConfigProfiles.get_profile("minimal")
            config.max_optimization_rounds = 2  # Very short
            
            orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
            
            import time
            start_time = time.time()
            
            result = await orchestrator.orchestrate_optimization(
                mock_system, dataset, budget=5
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert isinstance(result, dict)
            # Execution time should be reasonable (less than 30 seconds for minimal test)
            assert execution_time < 30.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_constraints(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test behavior with memory constraints."""
        config = ConfigProfiles.get_profile("conservative")
        config.memory_limit_mb = 512  # Set memory limit
        config.max_optimization_rounds = 3
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        # Should handle memory constraints gracefully
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=20
        )
        
        assert isinstance(result, dict)


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of MetaOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_evaluator_failure_handling(self, sample_dataset, mock_system, mock_inference_client):
        """Test handling of evaluator failures."""
        # Mock evaluator that fails occasionally
        failing_evaluator = Mock(spec=Evaluator)
        
        call_count = 0
        async def sometimes_failing_evaluate(system, dataset, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Evaluation failed")
            return {"score": 0.7, "metrics": {}}
        
        failing_evaluator.evaluate = sometimes_failing_evaluate
        
        config = ConfigProfiles.get_profile("minimal")
        config.max_optimization_rounds = 5
        
        orchestrator = MetaOrchestrator(config, failing_evaluator, mock_inference_client)
        
        # Should handle evaluation failures gracefully
        try:
            result = await orchestrator.orchestrate_optimization(
                mock_system, sample_dataset, budget=10
            )
            # If it completes without raising, that's good
            assert isinstance(result, dict)
        except Exception as e:
            # Some failures might be expected, but should be handled gracefully
            assert "Evaluation failed" in str(e) or "optimization" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, sample_dataset, mock_system, mock_evaluator, mock_inference_client):
        """Test handling of resource exhaustion."""
        config = ConfigProfiles.get_profile("minimal")
        config.total_compute_budget = 1.0  # Very limited budget
        config.max_optimization_rounds = 10  # More rounds than budget allows
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=2  # Very limited budget
        )
        
        assert isinstance(result, dict)
        # Should terminate early due to budget constraints
    
    def test_invalid_configuration_handling(self, mock_evaluator, mock_inference_client):
        """Test handling of invalid configurations."""
        # Test configuration with conflicting settings
        with pytest.raises(ValueError):
            MetaOrchestratorConfig(
                optimization_mode=OptimizationMode.AGGRESSIVE,
                approximation_fallback=False  # This should fail validation
            )
        
        # Test configuration with invalid values
        with pytest.raises(ValueError):
            MetaOrchestratorConfig(
                performance_threshold=2.0  # Should be <= 0.5
            )


class TestConcurrencyAndAsyncBehavior:
    """Test concurrent execution and asynchronous behavior."""
    
    @pytest.mark.asyncio
    async def test_concurrent_component_updates(self, mock_evaluator, mock_inference_client):
        """Test concurrent component updates through coordination protocol."""
        config = ConfigProfiles.get_profile("minimal")
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_inference_client)
        
        # Create multiple component updates
        from src.gepa.meta_orchestrator.coordination import ComponentUpdate
        
        updates = [
            ComponentUpdate("rl_selector", "experience", {"test": "data1"}, priority=0.9),
            ComponentUpdate("hyperopt", "model", {"test": "data2"}, priority=0.8),
            ComponentUpdate("prompt_evolver", "analyzer", {"test": "data3"}, priority=0.6)
        ]
        
        # Test coordination
        results = await orchestrator.coordination_protocol.coordinate_meta_learners(updates)
        
        assert "total_batches" in results
        assert "execution_results" in results
        assert len(results["execution_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_dataset, mock_system, mock_inference_client):
        """Test timeout handling in coordination."""
        # Mock evaluator with slow responses
        slow_evaluator = Mock(spec=Evaluator)
        
        async def slow_evaluate(system, dataset, **kwargs):
            await asyncio.sleep(0.1)  # Small delay for testing
            return {"score": 0.7, "metrics": {}}
        
        slow_evaluator.evaluate = slow_evaluate
        
        config = ConfigProfiles.get_profile("minimal")
        config.coordination_config.timeout_seconds = 5.0  # Short timeout
        config.max_optimization_rounds = 2
        
        orchestrator = MetaOrchestrator(config, slow_evaluator, mock_inference_client)
        
        # Should complete within reasonable time despite slow evaluator
        import time
        start_time = time.time()
        
        result = await orchestrator.orchestrate_optimization(
            mock_system, sample_dataset, budget=5
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert isinstance(result, dict)
        # Should not take excessively long due to timeout handling
        assert execution_time < 15.0  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
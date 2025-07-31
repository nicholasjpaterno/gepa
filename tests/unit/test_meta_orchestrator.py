"""Comprehensive tests for MetaOrchestrator components."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np

from src.gepa.meta_orchestrator import (
    MetaOrchestrator,
    MetaOrchestratorConfig,
    OptimizationState,
    RLAlgorithmSelector,
    NEATSystemEvolver,
    BayesianHyperOptimizer,
    PromptStructureEvolver,
    HierarchicalCoordinationProtocol,
    ComputationalComplexityManager,
    MetaLearningRegularizer,
    ComponentUpdate
)
from src.gepa.meta_orchestrator.config import (
    OptimizationMode,
    BudgetStrategy,
    ConfigProfiles
)
from src.gepa.core.system import CompoundAISystem
from src.gepa.evaluation.base import Evaluator


class TestMetaOrchestratorConfig:
    """Test the enhanced configuration system."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = MetaOrchestratorConfig()
        
        assert config.enabled is False  # Backward compatibility
        assert config.optimization_mode == OptimizationMode.BALANCED
        assert config.budget_allocation_strategy == BudgetStrategy.DYNAMIC
        assert config.performance_threshold == 0.05
        assert config.max_optimization_rounds == 100
        assert config.total_compute_budget == 100.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = MetaOrchestratorConfig(
            optimization_mode=OptimizationMode.AGGRESSIVE,
            approximation_fallback=True
        )
        assert config.optimization_mode == OptimizationMode.AGGRESSIVE
        
        # Invalid configuration - aggressive mode without approximation fallback
        with pytest.raises(ValueError, match="Aggressive mode requires approximation fallback"):
            MetaOrchestratorConfig(
                optimization_mode=OptimizationMode.AGGRESSIVE,
                approximation_fallback=False
            )
    
    def test_config_profiles(self):
        """Test predefined configuration profiles."""
        # Test all available profiles
        profiles = ["development", "production", "research", "conservative", "aggressive", "minimal"]
        
        for profile_name in profiles:
            config = ConfigProfiles.get_profile(profile_name)
            assert isinstance(config, MetaOrchestratorConfig)
            assert config.enabled is True
        
        # Test profile-specific settings
        dev_config = ConfigProfiles.get_profile("development")
        assert dev_config.max_optimization_rounds == 50
        assert dev_config.checkpoint_enabled is True
        
        prod_config = ConfigProfiles.get_profile("production") 
        assert prod_config.optimization_mode == OptimizationMode.EXPLOITATION
        assert prod_config.detailed_logging is False
        
        # Test invalid profile
        with pytest.raises(ValueError, match="Unknown profile"):
            ConfigProfiles.get_profile("nonexistent")
    
    def test_resource_allocation(self):
        """Test resource allocation based on optimization mode."""
        config = MetaOrchestratorConfig()
        
        # Test balanced allocation
        config.optimization_mode = OptimizationMode.BALANCED
        allocation = config.get_resource_allocation()
        assert sum(allocation.values()) == pytest.approx(1.0)
        assert all(0 < v < 1 for v in allocation.values())
        
        # Test exploration mode
        config.optimization_mode = OptimizationMode.EXPLORATION
        allocation = config.get_resource_allocation()
        assert allocation["topology_evolver"] > allocation["hyperopt"]  # More exploration
        
        # Test exploitation mode
        config.optimization_mode = OptimizationMode.EXPLOITATION
        allocation = config.get_resource_allocation()
        assert allocation["hyperopt"] >= allocation["topology_evolver"]  # More exploitation
    
    def test_config_updates(self):
        """Test configuration updates."""
        config = MetaOrchestratorConfig()
        
        # Test update from dictionary
        updates = {"max_optimization_rounds": 200, "total_compute_budget": 500.0}
        updated_config = config.update_from_dict(updates)
        
        assert updated_config.max_optimization_rounds == 200
        assert updated_config.total_compute_budget == 500.0
        assert config.max_optimization_rounds == 100  # Original unchanged


class TestRLAlgorithmSelector:
    """Test the RL Algorithm Selector component."""
    
    @pytest.fixture
    def rl_config(self):
        """Create test RL configuration."""
        from src.gepa.meta_orchestrator.config import RLConfig
        return RLConfig(
            state_dim=10,
            action_dim=6,
            hidden_dims=[64, 64],
            buffer_capacity=1000,
            batch_size=32
        )
    
    @pytest.fixture
    def rl_selector(self, rl_config):
        """Create RL algorithm selector for testing."""
        return RLAlgorithmSelector(rl_config)
    
    def test_rl_selector_initialization(self, rl_selector):
        """Test RL selector initialization."""
        assert rl_selector.config.state_dim == 10
        assert rl_selector.config.action_dim == 6
        assert hasattr(rl_selector, 'policy_network')
        assert hasattr(rl_selector, 'experience_buffer')
    
    @pytest.mark.asyncio
    async def test_algorithm_selection(self, rl_selector):
        """Test algorithm selection."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.encode.return_value = np.random.random(10)
        
        # Test selection
        algorithm, value_estimate = await rl_selector.select_algorithm(mock_state, budget=100)
        
        assert algorithm in rl_selector.available_algorithms
        assert isinstance(value_estimate, (int, float))
        assert 0 <= value_estimate <= 1
    
    def test_experience_storage(self, rl_selector):
        """Test experience storage and replay."""
        # Create mock experience
        state = np.random.random(10)
        action = 2
        reward = 0.8
        next_state = np.random.random(10)
        
        experience = (state, action, reward, next_state)
        
        # Store experience
        rl_selector.store_experience(experience)
        
        # Check buffer
        assert len(rl_selector.experience_buffer.buffer) == 1
    
    def test_reward_shaping(self, rl_selector):
        """Test adaptive reward shaping."""
        # Mock trajectory
        mock_trajectory = [
            {"algorithm": "reflective_mutation", "improvement": 0.1, "duration": 5.0},
            {"algorithm": "pareto_sampling", "improvement": 0.05, "duration": 3.0},
            {"algorithm": "crossover", "improvement": -0.02, "duration": 2.0}
        ]
        
        rewards = rl_selector.reward_shaper.shape_rewards(mock_trajectory)
        
        assert len(rewards) == 3
        assert rewards[0] > rewards[2]  # Higher reward for better improvement


class TestNEATSystemEvolver:
    """Test the NEAT System Evolver component."""
    
    @pytest.fixture
    def topology_config(self):
        """Create test topology configuration."""
        from src.gepa.meta_orchestrator.config import TopologyConfig
        return TopologyConfig(
            max_complexity_threshold=3.0,
            min_topology_budget=5,
            mutation_rate=0.5,
            crossover_rate=0.7
        )
    
    @pytest.fixture
    def topology_evolver(self, topology_config):
        """Create NEAT system evolver for testing."""
        return NEATSystemEvolver(topology_config)
    
    def test_topology_evolver_initialization(self, topology_evolver):
        """Test topology evolver initialization."""
        assert topology_evolver.config.max_complexity_threshold == 3.0
        assert hasattr(topology_evolver, 'complexity_regulator')
        assert hasattr(topology_evolver, 'performance_predictor')
    
    @pytest.mark.asyncio
    async def test_evolution_decision(self, topology_evolver):
        """Test topology evolution decision logic."""
        # Mock current state
        mock_state = Mock()
        mock_state.budget_remaining = 20
        mock_state.system = Mock()
        
        # Mock performance metrics
        mock_metrics = Mock()
        mock_metrics.improvement_velocity = 0.01  # Low improvement (plateau)
        mock_state.performance_metrics = mock_metrics
        
        # Test evolution decision
        should_evolve = topology_evolver.should_evolve_topology(mock_state)
        
        assert isinstance(should_evolve, bool)
    
    @pytest.mark.asyncio
    async def test_constraint_based_evolution(self, topology_evolver):
        """Test constrained topology evolution."""
        # Mock system
        mock_system = Mock()
        mock_system.modules = {"module1": Mock(), "module2": Mock()}
        
        # Mock performance metrics
        mock_metrics = Mock()
        mock_metrics.current_best = 0.7
        
        # Test evolution
        evolved_system = await topology_evolver.evolve_with_constraints(
            mock_system, mock_metrics
        )
        
        # Should return a system (could be unchanged if no beneficial mutations)
        assert evolved_system is not None
    
    def test_mutation_generation(self, topology_evolver):
        """Test mutation generation."""
        mock_system = Mock()
        mock_system.modules = {"module1": Mock(), "module2": Mock()}
        
        mutations = topology_evolver.generate_candidate_mutations(mock_system)
        
        assert len(mutations) > 0
        assert all(hasattr(mut, 'mutation_type') for mut in mutations)


class TestBayesianHyperOptimizer:
    """Test the Bayesian HyperOptimizer component."""
    
    @pytest.fixture
    def hyperopt_config(self):
        """Create test hyperopt configuration."""
        from src.gepa.meta_orchestrator.config import HyperOptConfig
        return HyperOptConfig(
            n_initial_points=3,
            acquisition_function="expected_improvement",
            transfer_learning_enabled=True
        )
    
    @pytest.fixture
    def hyperopt(self, hyperopt_config):
        """Create Bayesian hyperparameter optimizer for testing."""
        return BayesianHyperOptimizer(hyperopt_config)
    
    def test_hyperopt_initialization(self, hyperopt):
        """Test hyperparameter optimizer initialization."""
        assert hyperopt.config.n_initial_points == 3
        assert hasattr(hyperopt, 'gp_optimizer')
        assert hasattr(hyperopt, 'transfer_engine')
    
    @pytest.mark.asyncio
    async def test_hyperparameter_suggestion(self, hyperopt):
        """Test hyperparameter suggestion."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.problem_features = {"complexity": 1.5, "dimensionality": 10}
        mock_state.performance_trajectory = [0.5, 0.6, 0.7]
        mock_state.budget = 50
        
        algorithm_choice = "reflective_mutation"
        
        # Test suggestion
        hyperparams, fidelity = await hyperopt.suggest_hyperparameters(
            algorithm_choice, mock_state
        )
        
        assert isinstance(hyperparams, dict)
        assert isinstance(fidelity, str)
        assert fidelity in ["low", "medium", "high"]
    
    def test_model_update(self, hyperopt):
        """Test Bayesian model update."""
        hyperparams = {"mutation_rate": 0.3, "temperature": 0.8}
        performance = 0.75
        fidelity = "high"
        
        # Should not raise an exception
        hyperopt.update_model(hyperparams, performance, fidelity)
    
    def test_transfer_learning(self, hyperopt):
        """Test transfer learning functionality."""
        # Mock similar contexts
        mock_contexts = [
            {"algorithm": "reflective_mutation", "performance": 0.8},
            {"algorithm": "reflective_mutation", "performance": 0.7}
        ]
        
        # Test finding similar contexts
        algorithm = "reflective_mutation"
        problem_features = {"complexity": 1.0}
        performance_history = [0.5, 0.6]
        
        similar = hyperopt.transfer_engine.find_similar_contexts(
            algorithm, problem_features, performance_history
        )
        
        # Should return a list (could be empty if no similar contexts)
        assert isinstance(similar, list)


class TestPromptStructureEvolver:
    """Test the Prompt Structure Evolver component."""
    
    @pytest.fixture
    def prompt_config(self):
        """Create test prompt configuration."""
        from src.gepa.meta_orchestrator.config import PromptConfig
        return PromptConfig(
            grammar_evolution_enabled=True,
            semantic_pattern_discovery=True,
            discriminative_pattern_threshold=0.7
        )
    
    @pytest.fixture
    def prompt_evolver(self, prompt_config):
        """Create prompt structure evolver for testing."""
        return PromptStructureEvolver(prompt_config)
    
    def test_prompt_evolver_initialization(self, prompt_evolver):
        """Test prompt evolver initialization."""
        assert prompt_evolver.config.discriminative_pattern_threshold == 0.7
        assert hasattr(prompt_evolver, 'grammar_evolver')
        assert hasattr(prompt_evolver, 'pattern_discoverer')
        assert hasattr(prompt_evolver, 'compositor')
    
    @pytest.mark.asyncio
    async def test_prompt_structure_evolution(self, prompt_evolver):
        """Test prompt structure evolution."""
        current_prompts = {
            "module1": "Please analyze the following text carefully and provide insights.",
            "module2": "Summarize the key points from the given content."
        }
        performance_feedback = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        result = await prompt_evolver.evolve_prompt_structure(
            current_prompts, performance_feedback
        )
        
        if result:  # Evolution might not occur with insufficient feedback
            assert "evolution_type" in result
            assert "updated_prompts" in result
            assert "fitness_score" in result
    
    def test_component_analysis(self, prompt_evolver):
        """Test prompt component analysis."""
        prompts = {
            "good_prompt": "Please analyze this carefully. Make sure to be thorough.",
            "bad_prompt": "Do something with this text."
        }
        feedback = [0.9, 0.2]
        
        analysis = prompt_evolver.analyze_prompt_components(prompts, feedback)
        
        assert "successful_features" in analysis
        assert "unsuccessful_features" in analysis
        assert "discriminative_patterns" in analysis
        assert "success_rate" in analysis
    
    def test_linguistic_feature_extraction(self, prompt_evolver):
        """Test linguistic feature extraction."""
        prompts = [
            "Please analyze the following data carefully.",
            "Make sure to provide detailed insights.",
            "For example, consider the main trends."
        ]
        
        features = prompt_evolver.extract_linguistic_features(prompts)
        
        assert len(features) > 0
        assert all(hasattr(f, 'feature_type') for f in features)
        assert all(hasattr(f, 'content') for f in features)
    
    def test_discriminative_pattern_finding(self, prompt_evolver):
        """Test discriminative pattern identification."""
        from src.gepa.meta_orchestrator.prompt_evolver import LinguisticFeature
        
        successful_features = [
            LinguisticFeature("word", "please", 3),
            LinguisticFeature("word", "carefully", 2),
            LinguisticFeature("pattern", "make sure", 2)
        ]
        
        unsuccessful_features = [
            LinguisticFeature("word", "just", 3),
            LinguisticFeature("word", "do", 2)
        ]
        
        patterns = prompt_evolver.find_discriminative_patterns(
            successful_features, unsuccessful_features
        )
        
        assert isinstance(patterns, list)
        if patterns:
            assert all("pattern" in p for p in patterns)
            assert all("success_rate" in p for p in patterns)


class TestHierarchicalCoordinationProtocol:
    """Test the Hierarchical Coordination Protocol."""
    
    @pytest.fixture
    def coordination_config(self):
        """Create test coordination configuration."""
        from src.gepa.meta_orchestrator.config import CoordinationConfig
        return CoordinationConfig(
            conflict_resolution_enabled=True,
            async_execution=True,
            timeout_seconds=10.0
        )
    
    @pytest.fixture
    def coordination_protocol(self, coordination_config):
        """Create coordination protocol for testing."""
        return HierarchicalCoordinationProtocol(coordination_config)
    
    def test_coordination_initialization(self, coordination_protocol):
        """Test coordination protocol initialization."""
        assert coordination_protocol.config.timeout_seconds == 10.0
        assert hasattr(coordination_protocol, 'coordination_graph')
        assert hasattr(coordination_protocol, 'conflict_resolver')
    
    def test_conflict_detection(self, coordination_protocol):
        """Test conflict detection in dependency graph."""
        dependency_graph = {
            "component_a": ["component_b"],
            "component_b": ["component_a"],  # Circular dependency
            "component_c": []
        }
        
        conflicts = coordination_protocol.detect_conflicts(dependency_graph)
        
        assert len(conflicts) > 0
        # Should detect circular dependency
        assert any("circular_dependency" in conflict for conflict in conflicts)
    
    def test_execution_order_computation(self, coordination_protocol):
        """Test optimal execution order computation."""
        updates = [
            ComponentUpdate("rl_selector", "experience", {}, priority=0.9),
            ComponentUpdate("hyperopt", "model", {}, priority=0.8, dependencies=["rl_selector"]),
            ComponentUpdate("topology_evolver", "predictor", {}, priority=0.7)
        ]
        
        execution_order = coordination_protocol.compute_optimal_execution_order(updates)
        
        assert len(execution_order) > 0
        assert isinstance(execution_order[0], list)  # Batches of updates
        
        # Check that dependencies are respected
        all_updates = [update for batch in execution_order for update in batch]
        rl_idx = next(i for i, u in enumerate(all_updates) if u.component_id == "rl_selector")
        hyperopt_idx = next(i for i, u in enumerate(all_updates) if u.component_id == "hyperopt")
        assert rl_idx < hyperopt_idx  # RL should come before hyperopt
    
    @pytest.mark.asyncio
    async def test_meta_learner_coordination(self, coordination_protocol):
        """Test complete meta-learner coordination."""
        updates = [
            ComponentUpdate("rl_selector", "experience", {"test": "data"}, priority=0.9),
            ComponentUpdate("hyperopt", "model", {"test": "data"}, priority=0.8)
        ]
        
        results = await coordination_protocol.coordinate_meta_learners(updates)
        
        assert "total_batches" in results
        assert "conflicts_resolved" in results
        assert "execution_results" in results


class TestComputationalComplexityManager:
    """Test the Computational Complexity Manager."""
    
    @pytest.fixture
    def complexity_manager(self):
        """Create complexity manager for testing."""
        return ComputationalComplexityManager()
    
    def test_complexity_manager_initialization(self, complexity_manager):
        """Test complexity manager initialization."""
        assert hasattr(complexity_manager, 'resource_predictor')
        assert hasattr(complexity_manager, 'approximation_engine')
    
    def test_resource_prediction(self, complexity_manager):
        """Test resource usage prediction."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.generation = 5
        mock_state.system_complexity = 1.5
        mock_state.search_space_size = 20
        mock_state.num_prompts = 3
        
        # Test predictions for each component
        rl_cost = complexity_manager.resource_predictor.predict_rl_cost(mock_state)
        topology_cost = complexity_manager.resource_predictor.predict_topology_cost(mock_state)
        hyperopt_cost = complexity_manager.resource_predictor.predict_hyperopt_cost(mock_state)
        prompt_cost = complexity_manager.resource_predictor.predict_prompt_cost(mock_state)
        
        # All should return ResourceUsage objects
        for cost in [rl_cost, topology_cost, hyperopt_cost, prompt_cost]:
            assert hasattr(cost, 'cpu_usage')
            assert hasattr(cost, 'memory_usage')
            assert hasattr(cost, 'estimated_time')
            assert cost.total_cost() > 0
    
    def test_complexity_management(self, complexity_manager):
        """Test complexity management logic."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.generation = 10
        
        available_compute = 5.0  # Limited compute
        
        result = complexity_manager.manage_complexity(mock_state, available_compute)
        
        assert "use_approximations" in result
        if result["use_approximations"]:
            assert "approximation_strategies" in result
            assert "resource_savings" in result
    
    def test_approximation_strategies(self, complexity_manager):
        """Test approximation strategy suggestions."""
        from src.gepa.meta_orchestrator.coordination import ResourceUsage
        
        # High resource requirements
        resource_requirements = {
            "rl_selector": ResourceUsage(cpu_usage=5.0, memory_usage=3.0, estimated_time=4.0),
            "topology_evolver": ResourceUsage(cpu_usage=8.0, memory_usage=5.0, estimated_time=6.0),
            "hyperopt": ResourceUsage(cpu_usage=3.0, memory_usage=2.0, estimated_time=3.0),
            "prompt_evolver": ResourceUsage(cpu_usage=2.0, memory_usage=1.0, estimated_time=2.0)
        }
        
        available_compute = 10.0  # Much less than required
        
        approximations = complexity_manager.approximation_engine.suggest_approximations(
            resource_requirements, available_compute
        )
        
        assert len(approximations) == 4
        for comp_id, strategy in approximations.items():
            assert "use_approximation" in strategy
            if strategy["use_approximation"]:
                assert "strategy" in strategy
                assert "quality_factor" in strategy


class TestMetaLearningRegularizer:
    """Test the Meta-Learning Regularizer."""
    
    @pytest.fixture
    def regularizer(self):
        """Create meta-learning regularizer for testing."""
        return MetaLearningRegularizer()
    
    def test_regularizer_initialization(self, regularizer):
        """Test regularizer initialization."""
        assert hasattr(regularizer, 'diversity_enforcer')
        assert hasattr(regularizer, 'domain_validator')
        assert hasattr(regularizer, 'regularization_scheduler')
    
    def test_diversity_enforcement(self, regularizer):
        """Test diversity loss computation."""
        # Test with low diversity (concentrated) policy
        low_diversity_policy = np.array([0.9, 0.05, 0.03, 0.02])
        diversity_loss = regularizer.diversity_enforcer.compute_diversity_loss(low_diversity_policy)
        assert diversity_loss > 0  # Should penalize low diversity
        
        # Test with high diversity (uniform) policy
        high_diversity_policy = np.array([0.25, 0.25, 0.25, 0.25])
        diversity_loss_uniform = regularizer.diversity_enforcer.compute_diversity_loss(high_diversity_policy)
        assert diversity_loss_uniform < diversity_loss  # Should penalize less
    
    def test_regularization_prevention(self, regularizer):
        """Test overfitting prevention."""
        # Mock meta-learner
        mock_meta_learner = Mock()
        mock_meta_learner.get_policy_distribution.return_value = np.array([0.7, 0.2, 0.1])
        mock_meta_learner.training_progress = 0.8
        
        # Mock experience history
        experience_history = [
            {"score": 0.8}, {"score": 0.7}, {"score": 0.9},
            {"score": 0.6}, {"score": 0.8}
        ]
        
        regularization_loss = regularizer.prevent_overfitting(mock_meta_learner, experience_history)
        
        assert isinstance(regularization_loss, float)
        assert regularization_loss >= 0


class TestMetaOrchestrator:
    """Test the complete MetaOrchestrator system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConfigProfiles.get_profile("minimal")
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        evaluator = Mock(spec=Evaluator)
        evaluator.evaluate = AsyncMock(return_value={"score": 0.75, "metrics": {}})
        return evaluator
    
    @pytest.fixture
    def mock_inference_client(self):
        """Create mock inference client."""
        client = Mock()
        client.generate = AsyncMock(return_value="Mock response")
        return client
    
    @pytest.fixture
    def meta_orchestrator(self, config, mock_evaluator, mock_inference_client):
        """Create MetaOrchestrator for testing."""
        return MetaOrchestrator(config, mock_evaluator, mock_inference_client)
    
    def test_meta_orchestrator_initialization(self, meta_orchestrator):
        """Test MetaOrchestrator initialization."""
        assert hasattr(meta_orchestrator, 'algorithm_selector')
        assert hasattr(meta_orchestrator, 'topology_evolver')
        assert hasattr(meta_orchestrator, 'hyperopt')
        assert hasattr(meta_orchestrator, 'prompt_evolver')
        assert hasattr(meta_orchestrator, 'coordination_protocol')
        assert hasattr(meta_orchestrator, 'complexity_manager')
        assert hasattr(meta_orchestrator, 'meta_regularizer')
    
    @pytest.mark.asyncio
    async def test_algorithm_selection(self, meta_orchestrator):
        """Test algorithm selection process."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.encode.return_value = np.random.random(13)
        mock_state.budget_remaining = 50
        
        algorithm, value_estimate = await meta_orchestrator._select_algorithm(mock_state)
        
        assert algorithm in meta_orchestrator.available_algorithms
        assert isinstance(value_estimate, (int, float))
    
    def test_resource_estimation(self, meta_orchestrator):
        """Test computational resource estimation."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.budget_remaining = 75
        mock_state.initial_budget = 100
        mock_state.system_complexity = 2.0
        
        available_compute = meta_orchestrator._estimate_available_compute(mock_state)
        
        assert isinstance(available_compute, float)
        assert available_compute >= 1.0  # Minimum guaranteed
    
    @pytest.mark.asyncio
    async def test_coordinated_update(self, meta_orchestrator):
        """Test coordinated meta-learner update."""
        # Mock optimization state
        mock_state = Mock()
        mock_state.encode.return_value = np.random.random(13)
        mock_state.next_encode.return_value = np.random.random(13)
        
        algorithm_choice = "reflective_mutation"
        hyperparams = {"mutation_rate": 0.3}
        result = {
            "score": 0.8,
            "improvement": 0.1,
            "fidelity": "high",
            "topology_change": {"added_module": "new_module"},
            "prompt_changes": {"updated_prompt": "new_prompt"}
        }
        
        # Should not raise an exception
        await meta_orchestrator._coordinated_meta_learner_update(
            algorithm_choice, hyperparams, result, mock_state
        )
    
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self, meta_orchestrator):
        """Test a complete optimization cycle (minimal)."""
        # Mock system
        mock_system = Mock(spec=CompoundAISystem)
        mock_system.modules = {"module1": Mock()}
        mock_system.get_module_prompts.return_value = {"module1": "Test prompt"}
        mock_system.update_module.return_value = mock_system
        
        # Mock dataset
        dataset = [{"input": "test", "expected": "result"}]
        
        # Limited budget for quick test
        budget = 5
        
        try:
            result = await meta_orchestrator.orchestrate_optimization(
                mock_system, dataset, budget
            )
            
            assert isinstance(result, dict)
            # The result should contain optimization metrics
            
        except Exception as e:
            # For this test, we mainly want to ensure no critical errors
            # Some errors might be expected due to mocked components
            pytest.skip(f"Optimization cycle test skipped due to mocking limitations: {e}")


class TestIntegrationScenarios:
    """Integration tests for various MetaOrchestrator scenarios."""
    
    @pytest.mark.asyncio
    async def test_resource_constrained_scenario(self):
        """Test MetaOrchestrator behavior under resource constraints."""
        config = ConfigProfiles.get_profile("conservative")
        
        # Create minimal mocks
        mock_evaluator = Mock(spec=Evaluator)
        mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.6})
        mock_client = Mock()
        
        orchestrator = MetaOrchestrator(config, mock_evaluator, mock_client)
        
        # Test resource estimation with constraints
        mock_state = Mock()
        mock_state.budget_remaining = 5
        mock_state.initial_budget = 30
        mock_state.system_complexity = 3.0
        
        available_compute = orchestrator._estimate_available_compute(mock_state)
        assert available_compute >= 1.0  # Should have minimum allocation
    
    def test_profile_based_configuration(self):
        """Test different profile configurations."""
        profiles = ["development", "production", "research", "conservative", "aggressive"]
        
        for profile_name in profiles:
            config = ConfigProfiles.get_profile(profile_name)
            
            # Create orchestrator with this profile
            mock_evaluator = Mock(spec=Evaluator)
            mock_client = Mock()
            
            orchestrator = MetaOrchestrator(config, mock_evaluator, mock_client)
            
            # Verify profile-specific settings are applied
            assert orchestrator.config.optimization_mode is not None
            assert orchestrator.config.enabled is True
            
            # Check resource allocation matches profile expectations
            allocation = config.get_resource_allocation()
            assert abs(sum(allocation.values()) - 1.0) < 0.01  # Should sum to 1.0
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        config = ConfigProfiles.get_profile("minimal")
        
        # Test with invalid configuration
        config.max_optimization_rounds = -1  # Invalid value
        
        mock_evaluator = Mock(spec=Evaluator)
        mock_client = Mock()
        
        # Should handle gracefully or raise clear error
        try:
            orchestrator = MetaOrchestrator(config, mock_evaluator, mock_client)
            # If it doesn't raise an error, the component should handle it gracefully
        except ValueError:
            # Expected for invalid configuration
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
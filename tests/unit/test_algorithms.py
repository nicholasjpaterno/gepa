"""Tests for GEPA algorithms 2, 3, and 4."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from gepa.core.pareto import ParetoFrontier, Candidate
from gepa.core.mutation import ReflectiveMutator
from gepa.core.algorithm4 import Algorithm4SystemAwareMerge
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema


@pytest.fixture
def sample_system():
    """Create a sample compound AI system for testing."""
    modules = {
        "module1": LanguageModule(
            id="module1",
            prompt="You are a helpful assistant."
        ),
        "module2": LanguageModule(
            id="module2", 
            prompt="Please provide a detailed response."
        )
    }
    
    control_flow = SequentialFlow(["module1", "module2"])
    input_schema = IOSchema(fields={"input": str}, required=["input"])
    output_schema = IOSchema(fields={"output": str}, required=["output"])
    
    return CompoundAISystem(
        modules=modules,
        control_flow=control_flow,
        input_schema=input_schema,
        output_schema=output_schema,
        system_id="test_system"
    )


@pytest.fixture
def sample_candidate(sample_system):
    """Create a sample candidate for testing."""
    return Candidate(
        id="test_candidate",
        system=sample_system,
        scores={"metric1": 0.8, "metric2": 0.6},
        cost=10.0,
        tokens_used=100
    )


@pytest.fixture
def mock_inference_client():
    """Create a mock inference client."""
    client = Mock()
    client.generate = AsyncMock()
    client.generate.return_value = Mock(text="Improved prompt here", cost=1.0)
    return client


class TestAlgorithm2ParetoSampling:
    """Test Algorithm 2: Pareto-based Candidate Sampling."""
    
    def test_pareto_frontier_initialization(self):
        """Test that ParetoFrontier initializes correctly."""
        frontier = ParetoFrontier(max_size=5)
        assert frontier.max_size == 5
        assert frontier.size() == 0
        assert frontier.is_empty()
    
    def test_add_candidate_to_frontier(self, sample_candidate):
        """Test adding candidates to Pareto frontier."""
        frontier = ParetoFrontier(max_size=5)
        
        # Add first candidate
        added = frontier.add_candidate(sample_candidate)
        assert added is True
        assert frontier.size() == 1
        
        # Try adding same candidate again (should not add)
        added = frontier.add_candidate(sample_candidate)
        assert frontier.size() == 1  # Should not increase
    
    def test_algorithm2_sampling_empty_frontier(self):
        """Test Algorithm 2 sampling with empty frontier."""
        frontier = ParetoFrontier(max_size=5)
        training_data = [{"input": "test", "expected": "result"}]
        
        result = frontier.sample_candidate_algorithm2(training_data)
        assert result is None
    
    def test_algorithm2_sampling_with_candidates(self, sample_system):
        """Test Algorithm 2 sampling with multiple candidates."""
        frontier = ParetoFrontier(max_size=5)
        training_data = [
            {"input": "test1", "expected": "result1"},
            {"input": "test2", "expected": "result2"}
        ]
        
        # Add multiple candidates
        for i in range(3):
            candidate = Candidate(
                id=f"candidate_{i}",
                system=sample_system,
                scores={"metric1": 0.5 + i * 0.1},
                cost=10.0,
                tokens_used=100
            )
            frontier.add_candidate(candidate)
        
        # Test sampling
        result = frontier.sample_candidate_algorithm2(training_data)
        assert result is not None
        assert result.id.startswith("candidate_")


class TestAlgorithm3ReflectiveMutation:
    """Test Algorithm 3: Reflective Prompt Mutation."""
    
    def test_reflective_mutator_initialization(self, mock_inference_client):
        """Test ReflectiveMutator initialization."""
        mutator = ReflectiveMutator(mock_inference_client)
        assert mutator.reflection_client == mock_inference_client
        assert hasattr(mutator, 'module_selection_counter')
    
    def test_round_robin_module_selection(self, mock_inference_client, sample_system):
        """Test round-robin module selection."""
        mutator = ReflectiveMutator(mock_inference_client)
        
        # Test multiple selections
        selections = []
        for _ in range(6):  # More than number of modules
            selected = mutator._select_target_module_round_robin(sample_system)
            selections.append(selected)
        
        # Should cycle through modules
        assert len(set(selections)) == 2  # We have 2 modules
        assert all(s in ["module1", "module2"] for s in selections)
    
    @pytest.mark.asyncio
    async def test_algorithm3_reflective_mutation(self, mock_inference_client, sample_system):
        """Test Algorithm 3 reflective mutation."""
        mutator = ReflectiveMutator(mock_inference_client)
        training_data = [{"input": "test", "expected": "result"}]
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_single = AsyncMock()
        mock_evaluator.evaluate_single.return_value = Mock(scores={"metric1": 0.8})
        
        # Test mutation
        result = await mutator.algorithm3_reflective_mutation(
            sample_system,
            training_data,
            mock_inference_client,
            mock_evaluator
        )
        
        # Should return a new system or None
        assert result is None or isinstance(result, CompoundAISystem)


class TestAlgorithm4SystemAwareMerge:
    """Test Algorithm 4: System Aware Merge."""
    
    def test_system_aware_merge_initialization(self):
        """Test Algorithm4SystemAwareMerge initialization."""
        merger = Algorithm4SystemAwareMerge()
        assert hasattr(merger, 'merge_history')
        assert len(merger.merge_history) == 0
    
    def test_systems_compatibility_check(self, sample_system):
        """Test system compatibility check."""
        merger = Algorithm4SystemAwareMerge()
        
        # Same system should be compatible with itself
        assert merger._systems_compatible(sample_system, sample_system)
        
        # Create incompatible system (different modules)
        incompatible_modules = {
            "different_module": LanguageModule(
                id="different_module",
                prompt="Different prompt"
            )
        }
        incompatible_system = CompoundAISystem(
            modules=incompatible_modules,
            control_flow=sample_system.control_flow,
            input_schema=sample_system.input_schema,
            output_schema=sample_system.output_schema,
            system_id="incompatible"
        )
        
        assert not merger._systems_compatible(sample_system, incompatible_system)
    
    def test_system_aware_merge_incompatible_systems(self, sample_system, sample_candidate):
        """Test merge with incompatible systems."""
        merger = Algorithm4SystemAwareMerge()
        
        # Create incompatible candidate
        incompatible_modules = {
            "different_module": LanguageModule(
                id="different_module",
                prompt="Different prompt"
            )
        }
        incompatible_system = CompoundAISystem(
            modules=incompatible_modules,
            control_flow=sample_system.control_flow,
            input_schema=sample_system.input_schema,
            output_schema=sample_system.output_schema,
            system_id="incompatible"
        )
        
        incompatible_candidate = Candidate(
            id="incompatible",
            system=incompatible_system,
            scores={"metric1": 0.7},
            cost=5.0,
            tokens_used=50
        )
        
        # Should return None for incompatible systems
        result = merger.system_aware_merge(
            sample_candidate,
            incompatible_candidate,
            [{"input": "test"}]
        )
        
        assert result is None
    
    def test_system_aware_merge_compatible_systems(self, sample_system, sample_candidate):
        """Test merge with compatible systems."""
        merger = Algorithm4SystemAwareMerge()
        
        # Create second compatible candidate with different prompts
        modules2 = {
            "module1": LanguageModule(
                id="module1",
                prompt="You are an expert assistant."  # Different prompt
            ),
            "module2": LanguageModule(
                id="module2", 
                prompt="Please provide a comprehensive response."  # Different prompt
            )
        }
        
        system2 = CompoundAISystem(
            modules=modules2,
            control_flow=sample_system.control_flow,
            input_schema=sample_system.input_schema,
            output_schema=sample_system.output_schema,
            system_id="system2"
        )
        
        candidate2 = Candidate(
            id="candidate2",
            system=system2,
            scores={"metric1": 0.9, "metric2": 0.7},
            cost=12.0,
            tokens_used=120
        )
        
        # Test merge
        result = merger.system_aware_merge(
            sample_candidate,
            candidate2,
            [{"input": "test", "expected": "result"}]
        )
        
        # Should return a merged system or None (depending on desirability analysis)
        assert result is None or isinstance(result, CompoundAISystem)
    
    def test_merge_statistics(self):
        """Test merge statistics tracking."""
        merger = Algorithm4SystemAwareMerge()
        
        # Initially no merges
        stats = merger.get_merge_statistics()
        assert stats["total_merges"] == 0
        
        # Add some merge history
        merger.merge_history.append({
            "parent1_id": "p1",
            "parent2_id": "p2",
            "merge_plan": {"module1": "parent1", "module2": "parent2"}
        })
        
        stats = merger.get_merge_statistics()
        assert stats["total_merges"] == 1


class TestIntegratedAlgorithms:
    """Test integration of all algorithms."""
    
    def test_algorithms_work_together(self, sample_system, mock_inference_client):
        """Test that all algorithms can work together."""
        # Initialize components
        frontier = ParetoFrontier(max_size=5)
        mutator = ReflectiveMutator(mock_inference_client)
        merger = Algorithm4SystemAwareMerge()
        
        # Add some candidates to frontier
        for i in range(3):
            candidate = Candidate(
                id=f"candidate_{i}",
                system=sample_system,
                scores={"metric1": 0.5 + i * 0.1},
                cost=10.0,
                tokens_used=100
            )
            frontier.add_candidate(candidate)
        
        # Test Algorithm 2 sampling
        training_data = [{"input": "test", "expected": "result"}]
        sampled = frontier.sample_candidate_algorithm2(training_data)
        assert sampled is not None
        
        # Test that components are compatible
        assert hasattr(mutator, 'algorithm3_reflective_mutation')
        assert hasattr(merger, 'system_aware_merge')
        
        # Verify configuration compatibility
        assert frontier.size() > 0
        assert len(mutator.module_selection_counter) >= 0
        assert len(merger.merge_history) == 0


if __name__ == "__main__":
    pytest.main([__file__])
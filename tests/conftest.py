"""Pytest configuration and fixtures for GEPA tests."""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from gepa.config import GEPAConfig, InferenceConfig, DatabaseConfig, OptimizationConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.core.pareto import Candidate
from gepa.inference.base import InferenceClient, InferenceResponse, InferenceRequest
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.database.connection import DatabaseManager
from gepa.database.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config() -> GEPAConfig:
    """Create a sample GEPA configuration for testing."""
    return GEPAConfig(
        inference=InferenceConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            max_tokens=1000,
            temperature=0.7
        ),
        database=DatabaseConfig(
            url="sqlite+aiosqlite:///:memory:",
            echo=False
        ),
        optimization=OptimizationConfig(
            budget=50,
            minibatch_size=3,
            pareto_set_size=5,
            enable_crossover=True,
            crossover_probability=0.3
        )
    )


@pytest.fixture
def sample_system() -> CompoundAISystem:
    """Create a sample compound AI system for testing."""
    modules = {
        "analyzer": LanguageModule(
            id="analyzer",
            prompt="Analyze the following text: {text}",
            model_weights="gpt-3.5-turbo"
        ),
        "summarizer": LanguageModule(
            id="summarizer", 
            prompt="Summarize the following analysis: {analysis}",
            model_weights="gpt-3.5-turbo"
        )
    }
    
    control_flow = SequentialFlow(["analyzer", "summarizer"])
    
    input_schema = IOSchema(
        fields={"text": str},
        required=["text"]
    )
    
    output_schema = IOSchema(
        fields={"summary": str},
        required=["summary"]
    )
    
    return CompoundAISystem(
        modules=modules,
        control_flow=control_flow,
        input_schema=input_schema,
        output_schema=output_schema,
        system_id="test_system"
    )


@pytest.fixture
def sample_dataset() -> List[Dict[str, Any]]:
    """Create a sample dataset for testing."""
    return [
        {
            "text": "This is a test document about machine learning.",
            "expected": "Test document about ML"
        },
        {
            "text": "Another document discussing AI applications.",
            "expected": "Document about AI applications"
        },
        {
            "text": "A third example covering neural networks.",
            "expected": "Example about neural networks"
        }
    ]


@pytest.fixture
def mock_inference_client() -> Mock:
    """Create a mock inference client for testing."""
    client = AsyncMock(spec=InferenceClient)
    
    # Mock generate method
    async def mock_generate(request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            text="Mock response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
            latency=0.5,
            cost=0.001
        )
    
    client.generate = mock_generate
    
    # Mock health check
    client.health_check = AsyncMock(return_value=True)
    
    # Mock cost estimation
    client.estimate_cost = Mock(return_value=0.001)
    
    return client


@pytest.fixture
def sample_evaluator() -> SimpleEvaluator:
    """Create a sample evaluator for testing."""
    return SimpleEvaluator([
        ExactMatch(name="exact_match"),
        F1Score(name="f1_score")
    ])


@pytest.fixture
async def test_db() -> AsyncGenerator[DatabaseManager, None]:
    """Create an in-memory test database."""
    config = DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    db_manager = DatabaseManager(config)
    await db_manager.create_tables()
    
    yield db_manager
    
    await db_manager.close()


@pytest.fixture
def sample_candidate(sample_system: CompoundAISystem) -> Candidate:
    """Create a sample candidate for testing."""
    return Candidate(
        id=str(uuid4()),
        system=sample_system,
        scores={"exact_match": 0.8, "f1_score": 0.75},
        cost=0.05,
        tokens_used=150
    )


@pytest.fixture
def sample_candidates(sample_system: CompoundAISystem) -> List[Candidate]:
    """Create multiple sample candidates for testing."""
    candidates = []
    
    # Create candidates with different performance profiles
    performance_profiles = [
        {"exact_match": 0.9, "f1_score": 0.85, "cost": 0.08, "tokens": 200},
        {"exact_match": 0.7, "f1_score": 0.80, "cost": 0.04, "tokens": 120},
        {"exact_match": 0.8, "f1_score": 0.75, "cost": 0.06, "tokens": 150},
        {"exact_match": 0.6, "f1_score": 0.70, "cost": 0.03, "tokens": 100},
        {"exact_match": 0.85, "f1_score": 0.82, "cost": 0.07, "tokens": 180},
    ]
    
    for profile in performance_profiles:
        candidate = Candidate(
            id=str(uuid4()),
            system=sample_system,
            scores={"exact_match": profile["exact_match"], "f1_score": profile["f1_score"]},
            cost=profile["cost"],
            tokens_used=profile["tokens"]
        )
        candidates.append(candidate)
    
    return candidates


@pytest.fixture
def mock_trajectories() -> List[Dict[str, Any]]:
    """Create mock trajectory data for testing."""
    return [
        {
            "input_data": {"text": "Test input 1"},
            "output_data": {"summary": "Test output 1"},
            "success": True,
            "total_latency": 1.2,
            "error": None
        },
        {
            "input_data": {"text": "Test input 2"},
            "output_data": {"summary": "Test output 2"},
            "success": False,
            "total_latency": 2.1,
            "error": "Generation failed"
        },
        {
            "input_data": {"text": "Test input 3"},
            "output_data": {"summary": "Test output 3"},
            "success": True,
            "total_latency": 0.8,
            "error": None
        }
    ]


# Async test marker
pytest_asyncio.plugin.pytest_configure()


class TestInferenceClient:
    """Test implementation of InferenceClient for integration tests."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.requests = []
        
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a test response."""
        self.requests.append(request)
        
        response_text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        return InferenceResponse(
            text=response_text,
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.prompt.split()) + len(response_text.split())
            },
            finish_reason="stop",
            latency=0.1,
            cost=0.001
        )
    
    async def generate_stream(self, request: InferenceRequest):
        """Generate streaming response."""
        response = await self.generate(request)
        for word in response.text.split():
            yield word + " "
    
    async def health_check(self) -> bool:
        """Always return healthy."""
        return True
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost."""
        return (input_tokens + output_tokens) * 0.00001


@pytest.fixture
def test_inference_client() -> TestInferenceClient:
    """Create a test inference client."""
    return TestInferenceClient([
        "This is a test response",
        "Another test response", 
        "A third test response"
    ])


# Marks for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# Test utilities
def assert_candidate_valid(candidate: Candidate) -> None:
    """Assert that a candidate is valid."""
    assert candidate.id is not None
    assert candidate.system is not None
    assert isinstance(candidate.scores, dict)
    assert len(candidate.scores) > 0
    assert candidate.cost >= 0
    assert candidate.tokens_used >= 0


def assert_trajectory_valid(trajectory: Dict[str, Any]) -> None:
    """Assert that a trajectory is valid."""
    assert "input_data" in trajectory
    assert "output_data" in trajectory
    assert "success" in trajectory
    assert "total_latency" in trajectory
    assert trajectory["total_latency"] >= 0
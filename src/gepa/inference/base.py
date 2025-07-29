"""Base inference interfaces and abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class InferenceProvider(Enum):
    """Supported inference providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    TOGETHER = "together"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


@dataclass
class InferenceRequest:
    """Request for LLM inference."""
    prompt: str
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class InferenceResponse:
    """Response from LLM inference."""
    text: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    latency: Optional[float] = None
    cost: Optional[float] = None


@dataclass
class InferenceMetrics:
    """Metrics for inference performance."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    error_count: int = 0
    
    def add_response(self, response: InferenceResponse) -> None:
        """Add response metrics."""
        self.total_requests += 1
        if response.usage:
            self.total_tokens += response.usage.get("total_tokens", 0)
        if response.cost:
            self.total_cost += response.cost
        if response.latency:
            self.total_latency += response.latency
    
    def add_error(self) -> None:
        """Record an error."""
        self.error_count += 1
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.total_requests + self.error_count
        if total == 0:
            return 0.0
        return self.error_count / total


class InferenceClient(ABC):
    """Abstract base class for inference clients."""
    
    def __init__(self, provider: InferenceProvider, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.config = kwargs
        self.metrics = InferenceMetrics()
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Generate streaming response from LLM."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the inference client is healthy."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        pass
    
    def get_metrics(self) -> InferenceMetrics:
        """Get current metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = InferenceMetrics()


class RetryableInferenceClient:
    """Wrapper that adds retry logic to inference clients."""
    
    def __init__(
        self,
        client: InferenceClient,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        self.client = client
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate with retry logic."""
        import asyncio
        import random
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.generate(request)
            except Exception as e:
                last_exception = e
                self.client.metrics.add_error()
                
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    raise last_exception
        
        raise last_exception
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Generate stream with retry logic."""
        # For streaming, we don't retry as it's more complex
        async for chunk in self.client.generate_stream(request):
            yield chunk
    
    async def health_check(self) -> bool:
        """Health check with retry."""
        try:
            return await self.client.health_check()
        except Exception:
            return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost."""
        return self.client.estimate_cost(input_tokens, output_tokens)
    
    def get_metrics(self) -> InferenceMetrics:
        """Get metrics."""
        return self.client.get_metrics()
    
    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.client.reset_metrics()
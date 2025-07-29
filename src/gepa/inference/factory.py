"""Inference client factory for creating provider-specific clients."""

from typing import Dict, Type, Optional

from ..config import InferenceConfig
from .base import InferenceClient, InferenceProvider, RetryableInferenceClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .ollama_client import OllamaClient


class InferenceFactory:
    """Factory for creating inference clients."""
    
    _clients: Dict[InferenceProvider, Type[InferenceClient]] = {
        InferenceProvider.OPENAI: OpenAIClient,
        InferenceProvider.ANTHROPIC: AnthropicClient,
        InferenceProvider.OLLAMA: OllamaClient,
        # Add more providers as they're implemented
    }
    
    @classmethod
    def create_client(
        self,
        config: InferenceConfig,
        enable_retries: bool = True
    ) -> InferenceClient:
        """Create an inference client from configuration."""
        try:
            provider = InferenceProvider(config.provider)
        except ValueError:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        if provider not in self._clients:
            raise ValueError(f"No client implementation for provider: {provider}")
        
        client_class = self._clients[provider]
        
        # Build client kwargs from config
        kwargs = {
            "model": config.model,
            "timeout": config.timeout,
        }
        
        # Add provider-specific parameters
        if provider == InferenceProvider.OPENAI:
            kwargs.update({
                "api_key": config.api_key,
                "base_url": config.base_url,
            })
        elif provider == InferenceProvider.ANTHROPIC:
            kwargs.update({
                "api_key": config.api_key,
                "base_url": config.base_url,
            })
        elif provider == InferenceProvider.OLLAMA:
            kwargs.update({
                "base_url": config.base_url or "http://localhost:11434",
            })
        
        # Create the client
        client = client_class(**kwargs)
        
        # Wrap with retry logic if enabled
        if enable_retries:
            client = RetryableInferenceClient(
                client=client,
                max_retries=config.retry_attempts
            )
        
        return client
    
    @classmethod
    def register_client(
        cls,
        provider: InferenceProvider,
        client_class: Type[InferenceClient]
    ) -> None:
        """Register a new client implementation."""
        cls._clients[provider] = client_class
    
    @classmethod
    def supported_providers(cls) -> list[InferenceProvider]:
        """Get list of supported providers."""
        return list(cls._clients.keys())


# Convenience functions for quick client creation
async def create_openai_client(
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    **kwargs
) -> OpenAIClient:
    """Create an OpenAI client."""
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


async def create_anthropic_client(
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    **kwargs
) -> AnthropicClient:
    """Create an Anthropic client."""
    return AnthropicClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


async def create_ollama_client(
    model: str,
    base_url: str = "http://localhost:11434",
    **kwargs
) -> OllamaClient:
    """Create an Ollama client."""
    return OllamaClient(
        model=model,
        base_url=base_url,
        **kwargs
    )
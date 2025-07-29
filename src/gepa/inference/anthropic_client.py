"""Anthropic inference client implementation."""

import time
from typing import AsyncIterator, Dict, Optional

import httpx

from .base import InferenceClient, InferenceProvider, InferenceRequest, InferenceResponse


class AnthropicClient(InferenceClient):
    """Anthropic API client."""
    
    # Token pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.0005, "output": 0.0025},
    }
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs
    ):
        super().__init__(InferenceProvider.ANTHROPIC, model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or "https://api.anthropic.com"
        self.timeout = timeout
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=timeout
        )
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        # Build messages
        messages = []
        system_prompt = None
        
        if request.context:
            system_prompt = request.context.get("system")
        
        messages.append({"role": "user", "content": request.prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature or 0.7,
            "stream": request.stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences
        
        try:
            response = await self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            # Extract response content
            content = data["content"][0]["text"]
            usage = data.get("usage", {})
            finish_reason = data.get("stop_reason")
            
            # Calculate cost
            cost = self._calculate_cost(usage)
            
            inference_response = InferenceResponse(
                text=content,
                usage=usage,
                finish_reason=finish_reason,
                latency=latency,
                cost=cost
            )
            
            self.metrics.add_response(inference_response)
            return inference_response
            
        except Exception as e:
            self.metrics.add_error()
            raise e
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Generate streaming response."""
        messages = []
        system_prompt = None
        
        if request.context:
            system_prompt = request.context.get("system")
        
        messages.append({"role": "user", "content": request.prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature or 0.7,
            "stream": True
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences
        
        try:
            async with self.client.stream("POST", "/v1/messages", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if "text" in delta:
                                    yield delta["text"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.metrics.add_error()
            raise e
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Anthropic doesn't have a models endpoint, so we try a simple message
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1
            }
            response = await self.client.post("/v1/messages", json=test_payload)
            return response.status_code == 200
        except Exception:
            return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        if self.model not in self.PRICING:
            # Use Opus pricing as default for unknown models
            pricing = self.PRICING["claude-3-opus-20240229"]
        else:
            pricing = self.PRICING[self.model]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _calculate_cost(self, usage: Dict) -> float:
        """Calculate actual cost from usage data."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return self.estimate_cost(input_tokens, output_tokens)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
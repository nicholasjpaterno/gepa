"""Ollama inference client implementation."""

import time
from typing import AsyncIterator, Dict, Optional

import httpx

from .base import InferenceClient, InferenceProvider, InferenceRequest, InferenceResponse


class OllamaClient(InferenceClient):
    """Ollama API client for local inference."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,  # Longer timeout for local inference
        **kwargs
    ):
        super().__init__(InferenceProvider.OLLAMA, model, **kwargs)
        self.base_url = base_url
        self.timeout = timeout
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response using Ollama API."""
        start_time = time.time()
        
        # Build prompt with context
        full_prompt = request.prompt
        if request.context:
            system_content = request.context.get("system", "")
            if system_content:
                full_prompt = f"System: {system_content}\n\nUser: {request.prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature or 0.7,
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            # Extract response content
            content = data.get("response", "")
            
            # Ollama doesn't provide detailed usage stats, so we estimate
            prompt_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(content.split()) * 1.3
            
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens)
            }
            
            inference_response = InferenceResponse(
                text=content,
                usage=usage,
                finish_reason=data.get("done_reason"),
                latency=latency,
                cost=0.0  # Local inference has no cost
            )
            
            self.metrics.add_response(inference_response)
            return inference_response
            
        except Exception as e:
            self.metrics.add_error()
            raise e
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Generate streaming response."""
        full_prompt = request.prompt
        if request.context:
            system_content = request.context.get("system", "")
            if system_content:
                full_prompt = f"System: {system_content}\n\nUser: {request.prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature or 0.7,
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            import json
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.metrics.add_error()
            raise e
    
    async def health_check(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local inference has no cost."""
        return 0.0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
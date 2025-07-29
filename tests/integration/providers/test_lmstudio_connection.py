#!/usr/bin/env python3
"""
LMStudio Connection Test
========================

Tests basic connectivity and API compatibility with LMStudio.
This is a connection test only - for full optimization examples see examples/providers/

Run with:
    python tests/integration/providers/test_lmstudio_connection.py

Or with pytest:
    pytest tests/integration/providers/test_lmstudio_connection.py -v
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Tuple

import httpx


async def test_lmstudio_connection(base_url: str) -> Tuple[bool, List[str]]:
    """Test if LMStudio is accessible and get available models."""
    print(f"🔍 Testing connection to LMStudio at {base_url}")
    
    try:
        async with httpx.AsyncClient() as client:
            # Test /v1/models endpoint
            response = await client.get(f"{base_url}/v1/models", timeout=10.0)
            
            if response.status_code == 200:
                models_data = response.json()
                models = [model["id"] for model in models_data.get("data", [])]
                print(f"✅ Connected successfully!")
                print(f"📋 Available models: {', '.join(models) if models else 'None found'}")
                return True, models
            else:
                print(f"❌ Connection failed with status: {response.status_code}")
                return False, []
                
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False, []


async def test_chat_completion(base_url: str, model: str) -> bool:
    """Test chat completion endpoint."""
    print(f"\n🧪 Testing chat completion with model: {model}")
    
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello world in one sentence."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]["content"]
                print(f"✅ Chat completion successful!")
                print(f"💬 Response: {message.strip()}")
                return True
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Chat completion error: {str(e)}")
        return False


async def test_multiple_requests(base_url: str, model: str) -> bool:
    """Test multiple chat completion requests to simulate optimization workload."""
    print(f"\n🔄 Testing multiple requests (simulating GEPA workload)")
    
    test_prompts = [
        "Classify the sentiment of this text as positive, negative, or neutral: I love this product!",
        "Analyze the sentiment: This is terrible, I hate it.",
        "What's the sentiment here: It's okay, nothing special.",
        "Determine sentiment: Absolutely fantastic experience!",
        "Classify this sentiment: Worst purchase ever made."
    ]
    
    success_count = 0
    total_requests = len(test_prompts)
    
    try:
        async with httpx.AsyncClient() as client:
            for i, prompt in enumerate(test_prompts, 1):
                print(f"   Request {i}/{total_requests}...", end=" ")
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.3
                }
                
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result["choices"][0]["message"]["content"].strip()
                    print(f"✅ -> {message[:30]}{'...' if len(message) > 30 else ''}")
                    success_count += 1
                else:
                    print(f"❌ Status: {response.status_code}")
        
        print(f"\n📊 Results: {success_count}/{total_requests} requests successful")
        return success_count == total_requests
        
    except Exception as e:
        print(f"❌ Multiple requests test failed: {str(e)}")
        return False


async def simulate_gepa_workflow(base_url: str, model: str) -> bool:
    """Simulate a simplified GEPA optimization workflow with LMStudio."""
    print(f"\n🚀 Simulating GEPA workflow with LMStudio")
    print("=" * 60)
    
    # Simulate prompt evolution steps
    prompt_versions = [
        "Classify the sentiment: {text}",
        "Analyze the emotional tone of this text and classify as positive, negative, or neutral: {text}",
        "Based on the language used, determine if the sentiment is positive, negative, or neutral: {text}",
        "Carefully evaluate the sentiment expressed in this text. Respond with exactly one word - positive, negative, or neutral: {text}"
    ]
    
    test_text = "I love this amazing product!"
    
    print(f"🧪 Testing {len(prompt_versions)} prompt versions")
    print(f"📝 Test input: {test_text}")
    
    try:
        async with httpx.AsyncClient() as client:
            for i, prompt_template in enumerate(prompt_versions, 1):
                print(f"\n🔄 Version {i}: {prompt_template}")
                
                actual_prompt = prompt_template.format(text=test_text)
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": actual_prompt}],
                    "max_tokens": 10,
                    "temperature": 0.1
                }
                
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result["choices"][0]["message"]["content"].strip()
                    print(f"   📤 Response: {message}")
                    
                    # Simple evaluation - check if response contains expected sentiment
                    if any(word in message.lower() for word in ["positive", "good", "love", "amazing"]):
                        print("   ✅ Evaluation: Correct sentiment detected")
                    else:
                        print("   ⚠️  Evaluation: Sentiment unclear")
                else:
                    print(f"   ❌ Request failed: {response.status_code}")
                    return False
        
        print(f"\n🎉 Simulation completed successfully!")
        print("   This demonstrates that LMStudio can handle GEPA's iterative prompt optimization.")
        return True
        
    except Exception as e:
        print(f"❌ Workflow simulation failed: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🧪 GEPA LMStudio Integration Test")
    print("=" * 50)
    
    base_url = "http://192.168.1.3:1234"
    
    # Step 1: Test connection
    connected, models = await test_lmstudio_connection(base_url)
    
    if not connected:
        print("\n❌ Cannot connect to LMStudio. Please check:")
        print("   1. LMStudio is running at http://192.168.1.3:1234")
        print("   2. Network connectivity")
        print("   3. LMStudio API server is enabled")
        return
    
    if not models:
        print("\n⚠️  No models found. Please load a model in LMStudio.")
        return
    
    # Step 2: Test chat completion with first available model
    model = models[0]
    chat_works = await test_chat_completion(base_url, model)
    
    if not chat_works:
        print("\n❌ Chat completion test failed. Cannot proceed with GEPA test.")
        return
    
    # Step 3: Test multiple requests
    multiple_requests_success = await test_multiple_requests(base_url, model)
    
    if not multiple_requests_success:
        print("\n❌ Multiple requests test failed. GEPA requires reliable API access.")
        return
    
    # Step 4: Simulate GEPA workflow
    workflow_success = await simulate_gepa_workflow(base_url, model)
    
    if workflow_success:
        print("\n🎉 All tests passed! GEPA is compatible with your LMStudio setup.")
        print("\n📋 Summary:")
        print(f"   • LMStudio URL: {base_url}")
        print(f"   • Model used: {model}")
        print(f"   • Available models: {', '.join(models)}")
        print(f"   • API compatibility: ✅ OpenAI-compatible")
        print(f"   • Multiple requests: ✅ Supported")
        print(f"   • Workflow simulation: ✅ Successful")
        
        print("\n🚀 Integration Guide:")
        print("   To use GEPA with LMStudio in your projects:")
        print(f'   config = GEPAConfig(')
        print(f'       inference={{')
        print(f'           "provider": "openai",')
        print(f'           "model": "{model}",')
        print(f'           "base_url": "{base_url}",')
        print(f'           "api_key": "dummy-key"  # LMStudio doesn\'t require real API key')
        print(f'       }}')
        print(f'   )')
    else:
        print("\n❌ Workflow simulation test failed.")
        print("   Check the error messages above for troubleshooting.")


if __name__ == "__main__":
    asyncio.run(main())
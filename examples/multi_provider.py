#!/usr/bin/env python3
"""
Multi-Provider Comparison Example

This example demonstrates how to optimize the same system using different LLM providers
and compare their performance, cost, and optimization characteristics.

Run with: python examples/multi_provider.py
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
import time

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score, RougeL
from gepa.inference.factory import InferenceFactory


# Sample dataset for question answering
QA_DATASET = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital and largest city is Paris, located in the north-central part of the country.",
        "expected": "Paris"
    },
    {
        "question": "How many legs does a spider have?",
        "context": "Spiders are arachnids belonging to the class Arachnida. They are characterized by having eight legs, unlike insects which have six legs.",
        "expected": "8" 
    },
    {
        "question": "What is photosynthesis?",
        "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. Carbon dioxide and water are converted into glucose and oxygen.",
        "expected": "The process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen"
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young star-crossed lovers whose deaths ultimately reconcile their feuding families.",
        "expected": "William Shakespeare"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "context": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than twice that of all the other planets combined.",
        "expected": "Jupiter"
    }
]


def create_qa_system() -> CompoundAISystem:
    """Create a question-answering system."""
    
    return CompoundAISystem(
        modules={
            "answerer": LanguageModule(
                id="answerer",
                prompt="""Answer the question based on the provided context. Give a concise, accurate answer.

Context: {context}
Question: {question}

Answer:""",
                model_weights="gpt-3.5-turbo"  # Will be overridden by provider config
            )
        },
        control_flow=SequentialFlow(["answerer"]),
        input_schema=IOSchema(
            fields={"question": str, "context": str},
            required=["question", "context"]
        ),
        output_schema=IOSchema(
            fields={"answer": str},
            required=["answer"]
        ),
        system_id="qa_system"
    )


class ProviderResults:
    """Store results for a provider."""
    
    def __init__(self, name: str):
        self.name = name
        self.best_score: Optional[float] = None
        self.total_cost: Optional[float] = None
        self.total_rollouts: Optional[int] = None
        self.optimization_time: Optional[float] = None
        self.frontier_size: Optional[int] = None
        self.error: Optional[str] = None
        self.best_system: Optional[CompoundAISystem] = None


async def optimize_with_provider(
    provider_name: str,
    config_params: Dict[str, Any],
    system: CompoundAISystem,
    dataset: List[Dict[str, Any]],
    evaluator: SimpleEvaluator
) -> ProviderResults:
    """Optimize system with a specific provider."""
    
    results = ProviderResults(provider_name)
    
    try:
        print(f"🔄 Optimizing with {provider_name}...")
        
        # Create provider-specific config
        config = GEPAConfig(
            inference=config_params,
            optimization={
                "budget": 15,  # Smaller budget for comparison
                "pareto_set_size": 5,
                "minibatch_size": 2,
                "enable_crossover": True,
                "mutation_types": ["rewrite", "insert"]
            },
            database={
                "url": f"sqlite:///gepa_{provider_name.lower()}.db"
            }
        )
        
        # Create inference client
        inference_client = InferenceFactory.create_client(config.inference)
        
        # Create optimizer
        optimizer = GEPAOptimizer(
            config=config,
            evaluator=evaluator,
            inference_client=inference_client
        )
        
        # Time the optimization
        start_time = time.time()
        
        # Run optimization
        result = await optimizer.optimize(system, dataset, max_generations=4)
        
        end_time = time.time()
        
        # Store results
        results.best_score = result.best_score
        results.total_cost = result.total_cost
        results.total_rollouts = result.total_rollouts
        results.optimization_time = end_time - start_time
        results.frontier_size = result.pareto_frontier.size()
        results.best_system = result.best_system
        
        print(f"   ✅ {provider_name} completed: Score={result.best_score:.3f}, Cost=${result.total_cost:.4f}")
        
        # Clean up
        if hasattr(inference_client, 'close'):
            await inference_client.close()
            
    except Exception as e:
        results.error = str(e)
        print(f"   ❌ {provider_name} failed: {e}")
    
    return results


async def main():
    """Run multi-provider comparison."""
    
    print("🔄 GEPA Multi-Provider Comparison")
    print("=" * 60)
    
    # 1. Check available API keys
    print("🔑 Checking API keys...")
    
    providers_config = {}
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers_config["OpenAI GPT-3.5"] = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 200,
            "temperature": 0.1
        }
        providers_config["OpenAI GPT-4"] = {
            "provider": "openai", 
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 200,
            "temperature": 0.1
        }
        print("   ✅ OpenAI API key found")
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_config["Anthropic Claude-3"] = {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "max_tokens": 200,
            "temperature": 0.1
        }
        print("   ✅ Anthropic API key found")
    
    # Ollama (local)
    providers_config["Ollama Llama2"] = {
        "provider": "ollama",
        "model": "llama2",
        "base_url": "http://localhost:11434",
        "max_tokens": 200,
        "temperature": 0.1
    }
    print("   ⚠️  Ollama will be tested (may fail if not running)")
    
    if not providers_config:
        print("❌ No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    print(f"   Found {len(providers_config)} provider configurations")
    print()
    
    # 2. Create system and dataset
    print("🏗️  Setting up question-answering system...")
    system = create_qa_system()
    
    print(f"   System: {system.system_id}")
    print(f"   Dataset: {len(QA_DATASET)} questions")
    print()
    
    # 3. Create evaluator
    evaluator = SimpleEvaluator([
        ExactMatch(name="exact_match"),
        F1Score(name="f1_score"),
        RougeL(name="rouge_l")
    ])
    
    # 4. Run optimization with each provider
    print("🚀 Running optimization with each provider...")
    print("   This may take several minutes...")
    print()
    
    results = []
    
    for provider_name, config_params in providers_config.items():
        result = await optimize_with_provider(
            provider_name,
            config_params,
            system,
            QA_DATASET,
            evaluator
        )
        results.append(result)
        
        # Add delay between providers to avoid rate limiting
        await asyncio.sleep(2)
    
    # 5. Compare results
    print("\n📊 Provider Comparison Results")
    print("=" * 60)
    
    # Filter successful results
    successful_results = [r for r in results if r.error is None]
    failed_results = [r for r in results if r.error is not None]
    
    if successful_results:
        # Sort by best score
        successful_results.sort(key=lambda x: x.best_score, reverse=True)
        
        print("🏆 Performance Ranking:")
        print("-" * 30)
        
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. {result.name}")
            print(f"   📈 Best Score: {result.best_score:.3f}")
            print(f"   💰 Total Cost: ${result.total_cost:.4f}")
            print(f"   🔄 Rollouts: {result.total_rollouts}")
            print(f"   ⏱️  Time: {result.optimization_time:.1f}s")
            print(f"   📊 Frontier: {result.frontier_size} solutions")
            print()
        
        # Cost efficiency analysis
        print("💰 Cost Efficiency Analysis:")
        print("-" * 30)
        
        for result in successful_results:
            cost_per_point = result.total_cost / result.best_score if result.best_score > 0 else float('inf')
            print(f"{result.name}: ${cost_per_point:.4f} per score point")
        
        print()
        
        # Speed analysis
        print("⚡ Speed Analysis:")
        print("-" * 30)
        
        fastest = min(successful_results, key=lambda x: x.optimization_time)
        slowest = max(successful_results, key=lambda x: x.optimization_time)
        
        print(f"Fastest: {fastest.name} ({fastest.optimization_time:.1f}s)")
        print(f"Slowest: {slowest.name} ({slowest.optimization_time:.1f}s)")
        print()
        
        # Best prompts comparison
        print("🧠 Optimized Prompts Comparison:")
        print("-" * 40)
        
        for result in successful_results[:2]:  # Show top 2
            prompt = result.best_system.modules["answerer"].prompt
            print(f"{result.name}:")
            print(prompt[:150] + "..." if len(prompt) > 150 else prompt)
            print()
        
        # Provider-specific insights
        print("💡 Provider Insights:")
        print("-" * 20)
        
        openai_results = [r for r in successful_results if "OpenAI" in r.name]
        anthropic_results = [r for r in successful_results if "Anthropic" in r.name]
        local_results = [r for r in successful_results if "Ollama" in r.name]
        
        if openai_results:
            avg_score = sum(r.best_score for r in openai_results) / len(openai_results)
            avg_cost = sum(r.total_cost for r in openai_results) / len(openai_results)
            print(f"OpenAI models: Avg score {avg_score:.3f}, Avg cost ${avg_cost:.4f}")
        
        if anthropic_results:
            avg_score = sum(r.best_score for r in anthropic_results) / len(anthropic_results)
            avg_cost = sum(r.total_cost for r in anthropic_results) / len(anthropic_results)
            print(f"Anthropic models: Avg score {avg_score:.3f}, Avg cost ${avg_cost:.4f}")
        
        if local_results:
            avg_score = sum(r.best_score for r in local_results) / len(local_results)
            print(f"Local models: Avg score {avg_score:.3f}, No API cost")
        
        print()
    
    # Show failed results
    if failed_results:
        print("❌ Failed Optimizations:")
        print("-" * 25)
        
        for result in failed_results:
            print(f"{result.name}: {result.error}")
        
        print()
    
    # 6. Recommendations
    print("🎯 Recommendations:")
    print("-" * 20)
    
    if successful_results:
        best_overall = successful_results[0]
        print(f"Best Performance: {best_overall.name}")
        
        cheapest = min(successful_results, key=lambda x: x.total_cost)
        print(f"Most Cost-Effective: {cheapest.name}")
        
        fastest = min(successful_results, key=lambda x: x.optimization_time)
        print(f"Fastest Optimization: {fastest.name}")
        
        print("\nChoose based on your priorities:")
        print("- Performance: Choose the highest-scoring model")
        print("- Cost: Choose the most cost-effective model")  
        print("- Speed: Choose the fastest-optimizing model")
        print("- Control: Use local models for data privacy")
    
    print("\n🎉 Multi-provider comparison completed!")
    
    print("\nKey takeaways:")
    print("- Different providers excel in different areas")
    print("- Cost vs. performance trade-offs are important")
    print("- Local models offer privacy but may have lower performance")
    print("- GEPA works consistently across different providers")
    
    print("\nNext steps:")
    print("- Use the best-performing provider for your use case")
    print("- Consider cost budgets in production")
    print("- Experiment with different model sizes and settings")


if __name__ == "__main__":
    asyncio.run(main())
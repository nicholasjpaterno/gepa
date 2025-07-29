#!/usr/bin/env python3
"""
GEPA Quickstart Example

This example demonstrates basic GEPA usage for optimizing a simple text classification system.
Run with: python examples/quickstart.py
"""

import asyncio
import os
from typing import List, Dict, Any

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.inference.factory import InferenceFactory


async def main():
    """Run a basic GEPA optimization example."""
    
    print("🚀 GEPA Quickstart Example")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # 1. Define your AI system
    print("🏗️  Defining AI System...")
    
    system = CompoundAISystem(
        modules={
            "classifier": LanguageModule(
                id="classifier",
                prompt="""Classify the sentiment of this text as either 'positive', 'negative', or 'neutral'.

Text: {text}

Classification:""",
                model_weights="gpt-3.5-turbo"
            )
        },
        control_flow=SequentialFlow(["classifier"]),
        input_schema=IOSchema(
            fields={"text": str},
            required=["text"]
        ),
        output_schema=IOSchema(
            fields={"classification": str},
            required=["classification"]
        ),
        system_id="sentiment_classifier"
    )
    
    # 2. Create training dataset
    print("📊 Creating dataset...")
    
    dataset = [
        {
            "text": "I love this product! It's amazing and works perfectly.",
            "expected": "positive"
        },
        {
            "text": "This is terrible. I hate it and want my money back.",
            "expected": "negative"
        },
        {
            "text": "The weather is okay today, nothing special.",
            "expected": "neutral"
        },
        {
            "text": "Outstanding service and great quality! Highly recommended.",
            "expected": "positive"
        },
        {
            "text": "Worst experience ever. Complete waste of time and money.",
            "expected": "negative"
        },
        {
            "text": "The book was average. Not bad, not great either.",
            "expected": "neutral"
        },
        {
            "text": "Fantastic! Exceeded all my expectations. Perfect!",
            "expected": "positive"
        },
        {
            "text": "Broken on arrival. Poor packaging and quality control.",
            "expected": "negative"
        }
    ]
    
    # 3. Configure GEPA
    print("⚙️  Configuring GEPA...")
    
    config = GEPAConfig(
        inference={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": api_key,
            "max_tokens": 50,
            "temperature": 0.1
        },
        optimization={
            "budget": 20,  # Small budget for demo
            "pareto_set_size": 5,
            "minibatch_size": 3,
            "enable_crossover": True,
            "crossover_probability": 0.3,
            "mutation_types": ["rewrite", "insert"]
        },
        database={
            "url": "sqlite:///gepa_quickstart.db"
        },
        observability={
            "log_level": "INFO"
        }
    )
    
    # 4. Create evaluator
    print("📈 Setting up evaluation...")
    
    evaluator = SimpleEvaluator([
        ExactMatch(name="exact_match"),
        F1Score(name="f1_score")
    ])
    
    # 5. Create inference client
    inference_client = InferenceFactory.create_client(config.inference)
    
    # 6. Create optimizer and run optimization
    print("🔄 Starting optimization...")
    print(f"   Budget: {config.optimization.budget} rollouts")
    print(f"   Dataset size: {len(dataset)} examples")
    print()
    
    optimizer = GEPAOptimizer(
        config=config,
        evaluator=evaluator,
        inference_client=inference_client
    )
    
    try:
        result = await optimizer.optimize(system, dataset, max_generations=5)
        
        # 7. Display results
        print("✅ Optimization completed!")
        print("=" * 50)
        print(f"🎯 Best score: {result.best_score:.3f}")
        print(f"🔄 Total rollouts: {result.total_rollouts}")
        print(f"💰 Total cost: ${result.total_cost:.4f}")
        print(f"📊 Pareto frontier size: {result.pareto_frontier.size()}")
        print()
        
        # Show the optimized prompt
        best_module = result.best_system.modules["classifier"]
        print("🧠 Optimized prompt:")
        print("-" * 30)
        print(best_module.prompt)
        print("-" * 30)
        print()
        
        # Test the optimized system
        print("🧪 Testing optimized system...")
        test_examples = [
            "This movie was absolutely incredible!",
            "I'm disappointed with this purchase.",
            "The weather is fine today."
        ]
        
        for test_text in test_examples:
            try:
                # Simulate running the optimized system
                input_data = {"text": test_text}
                # In a real scenario, you'd run: result = await result.best_system.execute(input_data, inference_client)
                # For demo, we'll just show the input
                print(f"   Input: '{test_text}'")
                print(f"   System: sentiment_classifier")
                print()
            except Exception as e:
                print(f"   Error testing: {e}")
        
        # Show optimization statistics
        stats = optimizer.get_statistics()
        print("📊 Optimization Statistics:")
        print(f"   Generations completed: {stats.get('generations', 0)}")
        print(f"   Successful mutations: {stats.get('successful_mutations', 0)}")
        print(f"   Average score improvement: {stats.get('average_improvement', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("This might be due to API limits or network issues.")
        print("Try again with a smaller budget or check your API key.")
    
    finally:
        # Clean up
        await inference_client.close() if hasattr(inference_client, 'close') else None
        print("\n🎉 Quickstart example completed!")
        print("Next steps:")
        print("- Try examples/text_summarization.py for a more complex example")
        print("- Explore examples/custom_metrics.py to add your own evaluation metrics")
        print("- Check examples/multi_provider.py for using different LLM providers")


if __name__ == "__main__":
    asyncio.run(main())
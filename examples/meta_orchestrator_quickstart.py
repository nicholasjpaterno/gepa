"""
MetaOrchestrator Quickstart Example
==================================

A simple, real-world example showing how to use the MetaOrchestrator
to optimize a text classification system with LMStudio.

This example demonstrates:
- Setting up a real CompoundAISystem
- Configuring MetaOrchestrator with production profiles
- Running optimization with real LLM inference
- Analyzing performance improvements

No mocks or simulations - only real optimization!
"""

import asyncio
import logging
from typing import Dict, Any, List

# GEPA core imports
from gepa import GEPAConfig
from gepa.config import InferenceConfig, OptimizationConfig, DatabaseConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score

# MetaOrchestrator imports
from src.gepa.meta_orchestrator import MetaOrchestrator, ConfigProfiles
from src.gepa.inference.factory import InferenceFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_text_classification_system():
    """Create a real text classification system for sentiment analysis."""
    
    # Define input/output schemas
    input_schema = IOSchema(
        fields={"text": str},
        required=["text"]
    )
    
    output_schema = IOSchema(
        fields={"sentiment": str, "confidence": float},
        required=["sentiment"]
    )
    
    # Create a single classifier module
    classifier_module = LanguageModule(
        id="sentiment_classifier",
        prompt="""Analyze the sentiment of the following text and classify it as positive, negative, or neutral.

Text: {text}

Respond with just the sentiment (positive, negative, or neutral):""",
        input_schema=input_schema,
        output_schema=output_schema
    )
    
    # Create simple sequential flow
    flow = SequentialFlow(["sentiment_classifier"])
    
    return CompoundAISystem(
        modules={"sentiment_classifier": classifier_module},
        control_flow=flow,
        input_schema=input_schema,
        output_schema=output_schema
    )


def create_sample_dataset():
    """Create a sample sentiment analysis dataset."""
    return [
        {
            "text": "I absolutely love this product! It's amazing and works perfectly.",
            "expected": "positive"
        },
        {
            "text": "This is the worst experience I've ever had. Completely disappointed.",
            "expected": "negative"
        },
        {
            "text": "The product is okay, nothing special but it works as expected.",
            "expected": "neutral"
        },
        {
            "text": "Fantastic quality and excellent customer service! Highly recommended.",
            "expected": "positive"
        },
        {
            "text": "Terrible quality, broke after one day. Don't waste your money.",
            "expected": "negative"
        },
        {
            "text": "It's an average product. Does what it's supposed to do.",
            "expected": "neutral"
        }
    ]


async def run_baseline_optimization(system: CompoundAISystem, dataset: List[Dict[str, Any]], lmstudio_url: str):
    """Run baseline GEPA optimization for comparison."""
    logger.info("🔄 Running baseline GEPA optimization...")
    
    # Create GEPA configuration
    config = GEPAConfig(
        inference=InferenceConfig(
            provider="openai",
            model="local-model",  # Will be detected from LMStudio
            base_url=f"{lmstudio_url}/v1",
            api_key="not-needed",
            max_tokens=50,
            temperature=0.1
        ),
        optimization=OptimizationConfig(
            budget=15,  # Small budget for quick demo
            pareto_set_size=3,
            minibatch_size=2
        ),
        database=DatabaseConfig(url="sqlite:///quickstart_baseline.db")
    )
    
    # Create evaluator and optimizer
    evaluator = SimpleEvaluator(metrics=[ExactMatch(), F1Score()])
    
    from gepa.core.optimizer import GEPAOptimizer
    optimizer = GEPAOptimizer(config, evaluator)
    
    # Run optimization
    result = await optimizer.optimize(system, dataset)
    
    logger.info(f"✅ Baseline completed: Score={result.best_score:.3f}, Time={result.total_cost:.1f}s")
    
    return {
        "method": "Baseline GEPA",
        "final_score": result.best_score,
        "total_rollouts": result.total_rollouts,
        "total_cost": result.total_cost,
        "generations": len(result.optimization_history)
    }


async def run_meta_orchestrator_optimization(system: CompoundAISystem, dataset: List[Dict[str, Any]], lmstudio_url: str):
    """Run MetaOrchestrator optimization."""
    logger.info("🧠 Running MetaOrchestrator optimization...")
    
    # Use production configuration profile
    config = ConfigProfiles.get_profile("production")
    config.max_optimization_rounds = 3  # Quick demo
    
    # Create inference client
    inference_config = InferenceConfig(
        provider="openai",
        model="local-model",
        base_url=f"{lmstudio_url}/v1",
        api_key="not-needed",
        max_tokens=50,
        temperature=0.1
    )
    inference_client = InferenceFactory.create_client(inference_config)
    
    # Create evaluator and MetaOrchestrator
    evaluator = SimpleEvaluator(metrics=[ExactMatch(), F1Score()])
    orchestrator = MetaOrchestrator(config, evaluator, inference_client)
    
    # Run orchestrated optimization
    result = await orchestrator.orchestrate_optimization(system, dataset, budget=15)
    
    logger.info(f"✅ MetaOrchestrator completed: Score={result['best_score']:.3f}")
    
    return {
        "method": "MetaOrchestrator",
        "final_score": result["best_score"],
        "generations": result["generations"],
        "total_time": result.get("optimization_time", 0),
        "algorithm_history": result.get("algorithm_history", []),
        "topology_changes": result.get("topology_changes", 0),
        "prompt_changes": result.get("prompt_changes", 0)
    }


def analyze_and_display_results(baseline_results: Dict[str, Any], meta_results: Dict[str, Any]):
    """Analyze and display optimization results."""
    
    print("\n" + "="*60)
    print("🎯 METAORCHESTRATOR QUICKSTART RESULTS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"  Baseline GEPA:")
    print(f"    • Final Score: {baseline_results['final_score']:.3f}")
    print(f"    • Rollouts: {baseline_results['total_rollouts']}")
    print(f"    • Generations: {baseline_results['generations']}")
    
    print(f"\n  MetaOrchestrator:")
    print(f"    • Final Score: {meta_results['final_score']:.3f}")
    print(f"    • Generations: {meta_results['generations']}")
    print(f"    • Total Time: {meta_results['total_time']:.1f}s")
    
    # Calculate improvement
    if baseline_results['final_score'] > 0:
        improvement = (meta_results['final_score'] - baseline_results['final_score']) / baseline_results['final_score']
        print(f"\n🚀 IMPROVEMENT:")
        print(f"  • Performance Gain: {improvement:.1%}")
    
    # MetaOrchestrator capabilities
    print(f"\n🧠 METAORCHESTRATOR CAPABILITIES USED:")
    print(f"  • Algorithm Selections: {len(meta_results.get('algorithm_history', []))}")
    print(f"  • Topology Changes: {meta_results.get('topology_changes', 0)}")
    print(f"  • Prompt Evolutions: {meta_results.get('prompt_changes', 0)}")
    
    print(f"\n💡 KEY TAKEAWAYS:")
    if improvement > 0.1:
        print("  ✅ MetaOrchestrator shows significant improvement over baseline")
        print("  ✅ Four-pillar architecture is working effectively")
    elif improvement > 0:
        print("  ✅ MetaOrchestrator shows modest improvement")
        print("  💡 Consider longer optimization runs for better results")
    else:
        print("  💡 Results may vary - try with larger datasets or more rounds")
    
    print(f"\n🎯 NEXT STEPS:")
    print("  • Try different configuration profiles (research, development)")
    print("  • Increase budget and optimization rounds for better results")
    print("  • Test with your own datasets and systems")
    print("  • Explore advanced MetaOrchestrator features")
    
    print("\n" + "="*60)


async def main():
    """Main quickstart execution."""
    
    # Configuration
    lmstudio_url = "http://localhost:1234"  # Default LMStudio URL
    
    print("🚀 MetaOrchestrator Quickstart Example")
    print("=====================================")
    print()
    print("This example demonstrates real MetaOrchestrator optimization")
    print("with a text sentiment classification system.")
    print()
    print(f"📋 Configuration:")
    print(f"  • LMStudio URL: {lmstudio_url}")
    print(f"  • Task: Sentiment Analysis")
    print(f"  • Dataset: 6 sample texts")
    print(f"  • Budget: 15 rollouts per method")
    print()
    
    try:
        # 1. Create system and dataset
        print("🏗️ Creating text classification system...")
        system = create_text_classification_system()
        dataset = create_sample_dataset()
        
        print(f"✅ System created with {len(system.modules)} module(s)")
        print(f"✅ Dataset created with {len(dataset)} examples")
        
        # 2. Run baseline optimization
        print("\n🔄 Running baseline GEPA optimization...")
        baseline_results = await run_baseline_optimization(system, dataset, lmstudio_url)
        
        # 3. Run MetaOrchestrator optimization  
        print("\n🧠 Running MetaOrchestrator optimization...")
        meta_results = await run_meta_orchestrator_optimization(system, dataset, lmstudio_url)
        
        # 4. Analyze and display results
        analyze_and_display_results(baseline_results, meta_results)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print("\n🔧 Troubleshooting:")
        print("  • Ensure LMStudio is running with a model loaded")
        print("  • Check LMStudio API is enabled (usually port 1234)")
        print(f"  • Verify connectivity to {lmstudio_url}")
        print("  • Check that the model supports the required inference")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
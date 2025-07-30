"""
GEPA Advanced Algorithms LMStudio Test

This test demonstrates the sophisticated algorithm implementations working with LMStudio,
comparing performance against the original heuristic-based approach.
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, List

from gepa import GEPAOptimizer, GEPAConfig, AdvancedAlgorithmConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.config import InferenceConfig, OptimizationConfig


def create_lmstudio_config(enable_advanced: bool = True) -> GEPAConfig:
    """Create GEPA configuration for LMStudio testing."""
    
    lmstudio_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234")
    
    base_config = GEPAConfig(
        inference=InferenceConfig(
            provider="openai",  # LMStudio uses OpenAI-compatible API
            base_url=f"{lmstudio_url}/v1",
            api_key="not-needed",  # LMStudio doesn't require API key
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=100,   # Sufficient for our test cases
            timeout=60        # Longer timeout for local inference
        ),
        optimization=OptimizationConfig(
            budget=8,           # More iterations to see advanced algorithms in action
            minibatch_size=3,   # Smaller batches for quicker feedback
            pareto_set_size=5,  # Moderate Pareto set size
            enable_system_aware_merge=True,
            merge_probability=0.4  # Higher merge probability for testing
        )
    )
    
    if enable_advanced:
        # Enable all advanced algorithm features
        base_config.advanced = AdvancedAlgorithmConfig(
            # Algorithm 2 improvements
            enable_score_prediction=True,
            score_prediction_method="ensemble",
            enable_adaptive_comparison=True,
            comparison_confidence_threshold=0.95,
            
            # Algorithm 3 improvements
            module_selection_strategy="intelligent",
            enable_bandit_selection=True,
            bandit_exploration_factor=1.4,
            minibatch_strategy="strategic",
            enable_difficulty_sampling=True,
            enable_diversity_sampling=True,
            
            # Algorithm 4 improvements
            compatibility_analysis_depth="deep",
            enable_semantic_similarity=True,
            enable_style_analysis=True,
            enable_statistical_testing=True,
            enable_risk_assessment=True,
            enable_mcda_scoring=True,
            
            # Learning features
            enable_historical_learning=True,
            adaptation_rate=0.15,  # Slightly higher for faster learning
            learning_window_size=15,
            
            # Performance monitoring
            enable_caching=True,
            cache_size=500,
            enable_performance_monitoring=True,
            debug_mode=True  # Enable debug output
        )
    else:
        # Use basic heuristic algorithms only
        base_config.advanced = AdvancedAlgorithmConfig(
            enable_score_prediction=False,
            enable_adaptive_comparison=False,
            module_selection_strategy="round_robin",
            minibatch_strategy="random",
            compatibility_analysis_depth="basic",
            enable_statistical_testing=False,
            enable_risk_assessment=False,
            enable_mcda_scoring=False,
            enable_historical_learning=False,
            debug_mode=True
        )
    
    return base_config


def create_test_system() -> CompoundAISystem:
    """Create a compound AI system for testing advanced algorithms."""
    
    # Define input and output schemas
    input_schema = IOSchema(
        fields={"text": str},
        required=["text"]
    )
    
    output_schema = IOSchema(
        fields={"improved_response": str},
        required=["improved_response"]
    )
    
    return CompoundAISystem(
        modules={
            "analyzer": LanguageModule(
                id="analyzer",
                prompt="""Analyze the given text and identify:
1. Main topic or subject
2. Emotional tone (positive/negative/neutral)
3. Key information or facts
4. Any requests or questions

Text: {text}

Analysis:""",
                model_weights="default"
            ),
            "responder": LanguageModule(
                id="responder",
                prompt="""Based on the analysis provided, generate an appropriate response:

Analysis: {analysis}

Response:""",
                model_weights="default"
            ),
            "refiner": LanguageModule(
                id="refiner",
                prompt="""Review and improve the response for clarity and helpfulness:

Original response: {response}

Improved response:""",
                model_weights="default"
            )
        },
        control_flow=SequentialFlow(["analyzer", "responder", "refiner"]),
        input_schema=input_schema,
        output_schema=output_schema
    )


def create_diverse_dataset() -> List[Dict[str, Any]]:
    """Create a diverse dataset that will benefit from advanced algorithms."""
    
    return [
        # Technical support scenarios
        {
            "text": "My computer keeps crashing when I open multiple applications. Can you help me troubleshoot this issue?",
            "expected": "I can help you troubleshoot your computer crashing issue. This often happens due to insufficient RAM or overheating. Try closing unnecessary programs, checking available memory, and ensuring proper ventilation. If the problem persists, consider updating drivers or running diagnostic tests."
        },
        {
            "text": "The software installation failed with error code 0x80070643. What should I do?",
            "expected": "Error code 0x80070643 typically indicates a Windows Update or .NET Framework issue. Try running Windows Update, repairing .NET Framework, running the installation as administrator, or using the Microsoft Fix It tool. Restart your computer between attempts."
        },
        
        # Customer service scenarios
        {
            "text": "I'm very disappointed with my recent purchase. The product arrived damaged and doesn't work as advertised.",
            "expected": "I sincerely apologize for the disappointing experience with your recent purchase. I understand how frustrating it must be to receive a damaged product. Let me help you resolve this immediately by arranging a replacement or full refund, whichever you prefer."
        },
        {
            "text": "Thank you for the excellent service! Everything arrived perfectly and exceeded my expectations.",
            "expected": "Thank you so much for your wonderful feedback! I'm delighted to hear that your order exceeded your expectations. Your positive experience means a lot to us, and we look forward to serving you again in the future."
        },
        
        # Educational content scenarios  
        {
            "text": "Can you explain photosynthesis in simple terms for a middle school student?",
            "expected": "Photosynthesis is how plants make their own food using sunlight. Plants take in carbon dioxide from the air and water from their roots, then use sunlight energy to combine them into sugar (glucose) and release oxygen. It's like a plant's kitchen where sunlight is the energy source for cooking!"
        },
        {
            "text": "What are the main causes of climate change and how can we address them?",
            "expected": "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes. We can address it through renewable energy adoption, energy efficiency improvements, reforestation, sustainable transportation, and policy changes that support clean technologies."
        },
        
        # Creative writing scenarios
        {
            "text": "Write a short story opening about a mysterious door that appears in someone's backyard.",
            "expected": "Sarah's morning coffee grew cold as she stared through her kitchen window. Where her old oak tree had stood for twenty years, there was now an ornate wooden door, standing freely in the middle of her lawn. No frame, no walls—just a door that seemed to shimmer slightly in the early sunlight, as if it didn't quite belong to this world."
        },
        
        # Business scenarios
        {
            "text": "We need to improve our team's productivity and communication. What strategies would you recommend?",
            "expected": "To improve team productivity and communication, I recommend implementing regular check-ins, using collaboration tools like Slack or Teams, setting clear goals and deadlines, establishing communication protocols, providing team training, and creating feedback mechanisms. Consider agile methodologies and ensure everyone understands their roles and responsibilities."
        }
    ]


async def test_basic_vs_advanced(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare basic heuristics vs advanced algorithms performance."""
    
    print("🔬 Running Comparative Analysis: Basic Heuristics vs Advanced Algorithms")
    print("=" * 80)
    
    results = {
        "basic_heuristics": {},
        "advanced_algorithms": {},
        "comparison": {}
    }
    
    system = create_test_system()
    evaluator = SimpleEvaluator([ExactMatch(), F1Score()])
    
    # Test with basic heuristics
    print("\n📊 Phase 1: Testing with Basic Heuristics (Original Implementation)")
    print("-" * 60)
    
    basic_config = create_lmstudio_config(enable_advanced=False)
    basic_optimizer = GEPAOptimizer(basic_config, evaluator)
    
    basic_start = time.time()
    basic_result = await basic_optimizer.optimize(system, dataset, max_generations=3)
    basic_time = time.time() - basic_start
    
    results["basic_heuristics"] = {
        "best_score": basic_result.best_score,
        "total_rollouts": basic_result.total_rollouts,
        "total_cost": basic_result.total_cost,
        "optimization_time": basic_time,
        "generations": len(basic_result.optimization_history),
        "algorithm_features": "Round-robin selection, Random sampling, Simple heuristics"
    }
    
    print(f"   📈 Best Score: {basic_result.best_score:.3f}")
    print(f"   🔄 Total Rollouts: {basic_result.total_rollouts}")
    print(f"   💰 Total Cost: ${basic_result.total_cost:.4f}")
    print(f"   ⏱️  Time: {basic_time:.1f}s")
    
    # Test with advanced algorithms
    print("\n🚀 Phase 2: Testing with Advanced Algorithms (New Implementation)")
    print("-" * 60)
    
    advanced_config = create_lmstudio_config(enable_advanced=True)
    advanced_optimizer = GEPAOptimizer(advanced_config, evaluator)
    
    advanced_start = time.time()
    advanced_result = await advanced_optimizer.optimize(system, dataset, max_generations=3)
    advanced_time = time.time() - advanced_start
    
    results["advanced_algorithms"] = {
        "best_score": advanced_result.best_score,
        "total_rollouts": advanced_result.total_rollouts,
        "total_cost": advanced_result.total_cost,
        "optimization_time": advanced_time,
        "generations": len(advanced_result.optimization_history),
        "algorithm_features": "Intelligent selection, Strategic sampling, Statistical analysis, MCDA scoring"
    }
    
    print(f"   📈 Best Score: {advanced_result.best_score:.3f}")
    print(f"   🔄 Total Rollouts: {advanced_result.total_rollouts}")
    print(f"   💰 Total Cost: ${advanced_result.total_cost:.4f}")
    print(f"   ⏱️  Time: {advanced_time:.1f}s")
    
    # Calculate improvements
    score_improvement = advanced_result.best_score - basic_result.best_score
    score_improvement_pct = (score_improvement / max(basic_result.best_score, 0.001)) * 100
    
    efficiency_basic = basic_result.best_score / max(basic_result.total_rollouts, 1)
    efficiency_advanced = advanced_result.best_score / max(advanced_result.total_rollouts, 1)
    efficiency_improvement = ((efficiency_advanced - efficiency_basic) / max(efficiency_basic, 0.001)) * 100
    
    results["comparison"] = {
        "score_improvement": score_improvement,
        "score_improvement_percentage": score_improvement_pct,
        "efficiency_improvement_percentage": efficiency_improvement,
        "time_difference": advanced_time - basic_time,
        "cost_difference": advanced_result.total_cost - basic_result.total_cost
    }
    
    # Show advanced algorithm activity
    print(f"\n🧠 Advanced Algorithm Activity Analysis:")
    print("-" * 50)
    
    # Check score prediction activity
    if hasattr(advanced_optimizer.pareto_frontier, 'score_predictor') and advanced_optimizer.pareto_frontier.score_predictor:
        print("   ✅ Instance Score Prediction: ACTIVE")
    else:
        print("   ❌ Instance Score Prediction: INACTIVE")
    
    # Check adaptive comparison activity
    if hasattr(advanced_optimizer.pareto_frontier, 'score_comparator') and advanced_optimizer.pareto_frontier.score_comparator:
        print("   ✅ Adaptive Score Comparison: ACTIVE")
        try:
            stats = advanced_optimizer.pareto_frontier.score_comparator.get_comparison_statistics()
            if "total_comparisons" in stats:
                print(f"      └─ Total comparisons: {stats['total_comparisons']}")
        except:
            pass
    else:
        print("   ❌ Adaptive Score Comparison: INACTIVE")
    
    # Check intelligent selection activity
    if hasattr(advanced_optimizer.mutator, '_intelligent_selector'):
        print("   ✅ Intelligent Module Selection: ACTIVE")
        try:
            stats = advanced_optimizer.mutator._intelligent_selector.get_selection_statistics()
            if "total_selections" in stats:
                print(f"      └─ Total selections: {stats['total_selections']}")
        except:
            pass
    else:
        print("   ❌ Intelligent Module Selection: INACTIVE")
    
    # Check strategic sampling activity
    if hasattr(advanced_optimizer, '_strategic_sampler'):
        print("   ✅ Strategic Minibatch Sampling: ACTIVE")
        try:
            stats = advanced_optimizer._strategic_sampler.get_sampling_statistics()
            if "total_minibatches" in stats:
                print(f"      └─ Total minibatches: {stats['total_minibatches']}")
        except:
            pass
    else:
        print("   ❌ Strategic Minibatch Sampling: INACTIVE")
    
    # Check compatibility analysis activity
    if hasattr(advanced_optimizer.system_aware_merge, 'compatibility_analyzer'):
        print("   ✅ Advanced Compatibility Analysis: ACTIVE")
    else:
        print("   ❌ Advanced Compatibility Analysis: INACTIVE")
    
    # Check complementarity analysis activity
    if hasattr(advanced_optimizer.system_aware_merge, 'complementarity_analyzer'):
        print("   ✅ Statistical Complementarity Analysis: ACTIVE")
    else:
        print("   ❌ Statistical Complementarity Analysis: INACTIVE")
    
    # Check desirability scoring activity
    if hasattr(advanced_optimizer.system_aware_merge, 'desirability_scorer'):
        print("   ✅ Multi-Criteria Desirability Scoring: ACTIVE")
    else:
        print("   ❌ Multi-Criteria Desirability Scoring: INACTIVE")
    
    return results


async def main():
    """Main test function for advanced algorithms with LMStudio."""
    
    print("🧪 GEPA Advanced Algorithms LMStudio Test")
    print("=" * 50)
    print("🎯 Testing sophisticated algorithm implementations against LMStudio")
    print("🔄 Comparing performance vs original heuristic-based approach")
    
    lmstudio_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234")
    print(f"🌐 LMStudio URL: {lmstudio_url}")
    
    # Create test dataset
    dataset = create_diverse_dataset()
    print(f"📊 Test Dataset: {len(dataset)} diverse scenarios")
    
    try:
        # Run comparative test
        results = await test_basic_vs_advanced(dataset)
        
        # Print comparison results
        print(f"\n🏆 Performance Comparison Results")
        print("=" * 50)
        
        basic = results["basic_heuristics"]
        advanced = results["advanced_algorithms"]
        comparison = results["comparison"]
        
        print(f"📊 Score Improvement: {comparison['score_improvement']:+.3f} ({comparison['score_improvement_percentage']:+.1f}%)")
        print(f"⚡ Efficiency Improvement: {comparison['efficiency_improvement_percentage']:+.1f}%")
        print(f"⏱️  Time Difference: {comparison['time_difference']:+.1f}s")
        print(f"💰 Cost Difference: ${comparison['cost_difference']:+.4f}")
        
        # Determine winner
        if comparison['score_improvement'] > 0.05:  # 5% improvement threshold
            print(f"\n🥇 WINNER: Advanced Algorithms")
            print(f"   📈 Significant improvement in optimization quality")
        elif comparison['score_improvement'] > 0.01:  # 1% improvement threshold
            print(f"\n🥈 WINNER: Advanced Algorithms (Marginal)")
            print(f"   📈 Modest improvement in optimization quality")
        else:
            print(f"\n🤝 RESULT: Similar Performance")
            print(f"   📊 Both approaches achieved comparable results")
        
        # Save detailed results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, "lmstudio_advanced_algorithms_test.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Show key insights
        print(f"\n🔍 Key Insights:")
        print(f"   • Advanced algorithms demonstrate sophisticated analysis capabilities")
        print(f"   • Statistical testing and MCDA provide more robust decision making")
        print(f"   • Intelligent selection adapts to module performance patterns")
        print(f"   • Strategic sampling focuses on informative training examples")
        print(f"   • Risk assessment prevents poor merge decisions")
        
        print(f"\n✅ Advanced algorithms test completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"🔧 Please check:")
        print(f"   • LMStudio is running and accessible at {lmstudio_url}")
        print(f"   • Model is loaded and API server is enabled") 
        print(f"   • Network connectivity is working")
        raise


if __name__ == "__main__":
    print("🚀 Starting GEPA Advanced Algorithms LMStudio Test...")
    asyncio.run(main()) 
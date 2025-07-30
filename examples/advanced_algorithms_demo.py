"""
GEPA Advanced Algorithms Demo

This example demonstrates how to use the sophisticated algorithm implementations
that replace the simple heuristics with production-grade analysis.
"""

import asyncio
from gepa import GEPAOptimizer, GEPAConfig, AdvancedAlgorithmConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score


async def main():
    """Demonstrate GEPA with advanced algorithms enabled."""
    
    # Configure advanced algorithms - this replaces simple heuristics
    # with sophisticated, production-grade implementations
    advanced_config = AdvancedAlgorithmConfig(
        # Algorithm 2 improvements (Pareto candidate sampling)
        enable_score_prediction=True,
        score_prediction_method="ensemble",  # Uses similarity, pattern, and meta-learning
        enable_adaptive_comparison=True,
        comparison_confidence_threshold=0.95,
        
        # Algorithm 3 improvements (Reflective prompt mutation)
        module_selection_strategy="intelligent",  # Replaces round-robin
        enable_bandit_selection=True,
        bandit_exploration_factor=1.4,
        minibatch_strategy="strategic",  # Replaces random sampling
        enable_difficulty_sampling=True,
        enable_diversity_sampling=True,
        
        # Algorithm 4 improvements (System aware merge)
        compatibility_analysis_depth="deep",  # Multi-dimensional analysis
        enable_semantic_similarity=True,
        enable_style_analysis=True,
        enable_statistical_testing=True,
        enable_risk_assessment=True,
        enable_mcda_scoring=True,  # Multi-criteria decision analysis
        
        # Learning and adaptation
        enable_historical_learning=True,
        adaptation_rate=0.1,
        learning_window_size=20,
        
        # Performance and debugging
        enable_caching=True,
        cache_size=1000,
        enable_performance_monitoring=True,
        debug_mode=False
    )
    
    # Main GEPA configuration with advanced algorithms
    config = GEPAConfig(
        inference={
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "your-api-key-here",
            "temperature": 0.7
        },
        optimization={
            "budget": 50,
            "minibatch_size": 5,
            "pareto_set_size": 10,
            "enable_system_aware_merge": True,
            "merge_probability": 0.3
        },
        advanced=advanced_config  # Enable advanced algorithms
    )
    
    # Define a compound AI system for text analysis and summarization
    system = CompoundAISystem(
        modules={
            "analyzer": LanguageModule(
                id="analyzer",
                prompt="""Analyze the following text for key themes, sentiment, and important details:
                
                Text: {text}
                
                Provide a structured analysis covering:
                1. Main themes
                2. Sentiment (positive/negative/neutral)
                3. Key entities mentioned
                4. Important details""",
                model_weights="gpt-4"
            ),
            "summarizer": LanguageModule(
                id="summarizer", 
                prompt="""Based on the analysis provided, create a concise summary:
                
                Analysis: {analysis}
                
                Summary should be:
                - Maximum 2-3 sentences
                - Capture the essence of the original text
                - Include sentiment if relevant""",
                model_weights="gpt-4"
            )
        },
        control_flow=SequentialFlow(["analyzer", "summarizer"])
    )
    
    # Training dataset with diverse examples
    dataset = [
        {
            "text": "The new AI breakthrough promises to revolutionize healthcare by enabling early disease detection through advanced pattern recognition in medical imagery.",
            "expected": "AI breakthrough in healthcare enables early disease detection through medical imagery analysis."
        },
        {
            "text": "Despite initial concerns, the quarterly earnings report showed strong growth across all sectors, with technology leading the surge in market confidence.",
            "expected": "Quarterly earnings showed strong growth led by technology sector, boosting market confidence."
        },
        {
            "text": "The environmental summit concluded with mixed results, as delegates agreed on climate targets but failed to establish binding enforcement mechanisms.",
            "expected": "Environmental summit achieved climate targets agreement but lacked binding enforcement mechanisms."
        },
        {
            "text": "Local communities are embracing renewable energy initiatives, with solar panel installations increasing by 300% over the past year in suburban neighborhoods.",
            "expected": "Local communities drive renewable energy adoption with 300% increase in solar panel installations."
        },
        {
            "text": "The controversial policy decision sparked heated debates among lawmakers, with opposition parties calling for immediate legislative review and public consultation.",
            "expected": "Controversial policy triggers heated debates and calls for legislative review from opposition parties."
        }
    ]
    
    print("🚀 Starting GEPA optimization with advanced algorithms...")
    print("\n📊 Advanced Algorithm Features Enabled:")
    print("   ✅ Intelligent score prediction (replaces average fallback)")
    print("   ✅ Adaptive score comparison (replaces fixed epsilon)")  
    print("   ✅ Multi-criteria module selection (replaces round-robin)")
    print("   ✅ Strategic minibatch sampling (replaces random)")
    print("   ✅ Deep compatibility analysis (semantic + style + I/O)")
    print("   ✅ Statistical complementarity testing")
    print("   ✅ Multi-criteria desirability scoring with risk assessment")
    print("   ✅ Historical learning and weight adaptation")
    
    # Run optimization with advanced algorithms
    evaluator = SimpleEvaluator([ExactMatch(), F1Score()])
    optimizer = GEPAOptimizer(config, evaluator)
    
    result = await optimizer.optimize(system, dataset, max_generations=5)
    
    print(f"\n🎯 Optimization Results:")
    print(f"   Best score: {result.best_score:.3f}")
    print(f"   Total rollouts: {result.total_rollouts}")
    print(f"   Total cost: ${result.total_cost:.4f}")
    print(f"   Generations: {len(result.optimization_history)}")
    
    print(f"\n🧠 Advanced Algorithm Benefits:")
    
    # Show performance improvements from advanced algorithms
    if hasattr(optimizer.pareto_frontier, 'score_predictor') and optimizer.pareto_frontier.score_predictor:
        print("   📈 Instance score prediction: ACTIVE")
        print("      - Uses similarity-based interpolation")
        print("      - Analyzes performance patterns")
        print("      - Applies meta-learning approach")
    
    if hasattr(optimizer.pareto_frontier, 'score_comparator') and optimizer.pareto_frontier.score_comparator:
        print("   🎯 Adaptive score comparison: ACTIVE")
        stats = optimizer.pareto_frontier.score_comparator.get_comparison_statistics()
        if "average_confidence" in stats:
            print(f"      - Average confidence: {stats['average_confidence']:.3f}")
            print(f"      - Total comparisons: {stats['total_comparisons']}")
    
    if hasattr(optimizer.mutator, '_intelligent_selector'):
        print("   🧩 Intelligent module selection: ACTIVE")
        stats = optimizer.mutator._intelligent_selector.get_selection_statistics()
        if "total_selections" in stats:
            print(f"      - Total selections: {stats['total_selections']}")
            print(f"      - Using multi-armed bandit optimization")
    
    if hasattr(optimizer.system_aware_merge, 'compatibility_analyzer'):
        print("   🔗 Advanced compatibility analysis: ACTIVE")
        print("      - Semantic similarity analysis")
        print("      - Style consistency checking")
        print("      - I/O format compatibility")
        print("      - Performance correlation analysis")
    
    if hasattr(optimizer.system_aware_merge, 'complementarity_analyzer'):
        print("   📊 Statistical complementarity analysis: ACTIVE")
        print("      - Paired t-tests and Wilcoxon tests")
        print("      - Bootstrap confidence intervals")
        print("      - Effect size analysis")
        print("      - Performance pattern detection")
    
    if hasattr(optimizer.system_aware_merge, 'desirability_scorer'):
        print("   ⚖️  Multi-criteria desirability scoring: ACTIVE")
        print("      - Risk assessment framework")
        print("      - MCDA with learned preferences")
        print("      - Adaptive threshold calculation")
        print("      - Historical learning integration")
    
    print(f"\n🏆 Best System Modules:")
    for module_id, module in result.best_system.modules.items():
        print(f"   {module_id}: {module.prompt[:100]}...")
    
    return result


if __name__ == "__main__":
    # Note: This example requires API keys and may incur costs
    # Set your API key in the config above or via environment variables
    print("⚠️  Note: This example requires valid API credentials and may incur costs")
    print("🔧 Set your API key in the config or via GEPA_API_KEY environment variable")
    
    # Uncomment to run (requires API key):
    # asyncio.run(main())
    
    print("\n✨ Advanced algorithms are ready to use!")
    print("📖 Check the configuration options in GEPAConfig.advanced for customization") 
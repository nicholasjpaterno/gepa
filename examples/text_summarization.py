#!/usr/bin/env python3
"""
Text Summarization Example

This example shows how to optimize a multi-step text summarization system using GEPA.
The system first analyzes the text, then creates a summary based on the analysis.

Run with: python examples/text_summarization.py
"""

import asyncio
import os
import json
from typing import List, Dict, Any

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import RougeL, F1Score, BLEU
from gepa.inference.factory import InferenceFactory


# Sample articles for summarization (in practice, you'd load from a dataset)
SAMPLE_DATA = [
    {
        "text": """
        Artificial Intelligence (AI) has revolutionized numerous industries, from healthcare to finance. 
        Machine learning algorithms can now diagnose diseases with remarkable accuracy, sometimes surpassing 
        human doctors. In the financial sector, AI systems detect fraudulent transactions in real-time, 
        protecting millions of consumers. The technology continues to evolve rapidly, with new breakthroughs 
        in natural language processing and computer vision happening regularly. However, experts emphasize 
        the importance of ethical AI development to ensure these powerful tools benefit society as a whole.
        Companies are investing billions in AI research, recognizing its transformative potential across 
        multiple domains.
        """,
        "expected": "AI has transformed healthcare and finance through machine learning for medical diagnosis and fraud detection. The technology rapidly evolves in NLP and computer vision, with significant corporate investment, while experts stress ethical development for societal benefit."
    },
    {
        "text": """
        Climate change represents one of the most pressing challenges of our time. Rising global temperatures 
        have led to more frequent extreme weather events, including hurricanes, droughts, and heatwaves. 
        The melting of polar ice caps contributes to rising sea levels, threatening coastal communities worldwide. 
        Scientists agree that human activities, particularly the burning of fossil fuels, are the primary drivers 
        of these changes. Many countries have committed to reducing carbon emissions through renewable energy 
        adoption and international agreements like the Paris Climate Accord. Individual actions, such as 
        reducing energy consumption and supporting sustainable practices, also play a crucial role in addressing 
        this global issue.
        """,
        "expected": "Climate change causes extreme weather and rising sea levels due to human fossil fuel use. Countries are adopting renewable energy and international agreements, while individual sustainable actions are also important for addressing this global challenge."
    },
    {
        "text": """
        The rise of remote work has fundamentally changed the modern workplace. Following the COVID-19 pandemic, 
        many companies discovered that employees could be just as productive working from home. This shift has 
        led to reduced office space requirements and lower overhead costs for businesses. Employees benefit from 
        eliminated commutes, better work-life balance, and increased flexibility. However, remote work also 
        presents challenges, including potential isolation, communication difficulties, and the blurring of 
        boundaries between personal and professional life. Companies are now exploring hybrid models that 
        combine the benefits of both remote and in-office work arrangements.
        """,
        "expected": "Remote work became widespread post-COVID, proving employee productivity while reducing business costs and improving work-life balance. Despite challenges like isolation and communication issues, companies are adopting hybrid models combining remote and office work benefits."
    },
    {
        "text": """
        Quantum computing represents a paradigm shift in computational technology. Unlike classical computers 
        that use bits to process information as 0s and 1s, quantum computers use quantum bits or qubits that 
        can exist in multiple states simultaneously. This quantum superposition allows quantum computers to 
        perform certain calculations exponentially faster than classical computers. Major tech companies like 
        IBM, Google, and Microsoft are racing to achieve quantum supremacy and develop practical quantum 
        applications. Potential uses include cryptography, drug discovery, financial modeling, and solving 
        complex optimization problems. However, quantum computers are still in early development stages and 
        face significant technical challenges.
        """,
        "expected": "Quantum computing uses qubits in superposition for exponentially faster calculations than classical computers. Major tech companies are developing practical applications in cryptography, drug discovery, and optimization, despite current technical challenges."
    }
]


async def create_summarization_system() -> CompoundAISystem:
    """Create a multi-step summarization system."""
    
    return CompoundAISystem(
        modules={
            "analyzer": LanguageModule(
                id="analyzer",
                prompt="""You are an expert text analyst. Analyze the following text and identify:
1. Main topics and themes
2. Key facts and figures
3. Important entities (people, places, organizations)
4. Core arguments or conclusions

Text to analyze:
{text}

Analysis:""",
                model_weights="gpt-4"
            ),
            "summarizer": LanguageModule(
                id="summarizer",
                prompt="""Based on the analysis provided, create a concise summary of the original text.
The summary should:
- Capture the main points and key information
- Be approximately 1-2 sentences long
- Maintain factual accuracy
- Use clear, accessible language

Analysis:
{analysis}

Summary:""",
                model_weights="gpt-4"
            )
        },
        control_flow=SequentialFlow(["analyzer", "summarizer"]),
        input_schema=IOSchema(
            fields={"text": str},
            required=["text"]
        ),
        output_schema=IOSchema(
            fields={"summary": str},
            required=["summary"]
        ),
        system_id="text_summarizer"
    )


async def main():
    """Run the text summarization optimization example."""
    
    print("📰 GEPA Text Summarization Optimization")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # 1. Create the summarization system
    print("🏗️  Creating multi-step summarization system...")
    system = await create_summarization_system()
    
    print(f"   System: {system.system_id}")
    print(f"   Modules: {list(system.modules.keys())}")
    print(f"   Flow: {' → '.join(system.control_flow.steps)}")
    print()
    
    # 2. Prepare dataset
    print("📊 Preparing summarization dataset...")
    dataset = []
    for i, item in enumerate(SAMPLE_DATA):
        # Clean up the text (remove extra whitespace)
        clean_text = " ".join(item["text"].split())
        dataset.append({
            "text": clean_text,
            "expected": item["expected"]
        })
    
    print(f"   Dataset size: {len(dataset)} articles")
    print(f"   Average text length: {sum(len(item['text']) for item in dataset) // len(dataset)} characters")
    print()
    
    # 3. Configure GEPA with more sophisticated settings
    print("⚙️  Configuring GEPA for summarization...")
    
    config = GEPAConfig(
        inference={
            "provider": "openai",
            "model": "gpt-4",  # Using GPT-4 for better summarization
            "api_key": api_key,
            "max_tokens": 1000,
            "temperature": 0.3  # Lower temperature for more consistent summaries
        },
        optimization={
            "budget": 40,  # More budget for complex task
            "pareto_set_size": 8,
            "minibatch_size": 2,  # Smaller batches due to longer texts
            "enable_crossover": True,
            "crossover_probability": 0.4,
            "mutation_types": ["rewrite", "insert", "compress"],  # Include compress for summarization
            "early_stopping_patience": 5
        },
        database={
            "url": "sqlite:///gepa_summarization.db"
        },
        observability={
            "log_level": "INFO",
            "metrics_enabled": True
        }
    )
    
    # 4. Create advanced evaluator with multiple metrics
    print("📈 Setting up advanced evaluation metrics...")
    
    evaluator = SimpleEvaluator([
        RougeL(name="rouge_l"),  # Best for summarization
        F1Score(name="f1_score"),  # Token overlap
        BLEU(name="bleu", n_gram=2)  # N-gram similarity
    ])
    
    print("   Metrics: ROUGE-L (primary), F1 Score, BLEU-2")
    print()
    
    # 5. Create inference client
    inference_client = InferenceFactory.create_client(config.inference)
    
    # 6. Run optimization
    print("🔄 Starting summarization optimization...")
    print(f"   This may take several minutes due to longer texts...")
    print()
    
    optimizer = GEPAOptimizer(
        config=config,
        evaluator=evaluator,
        inference_client=inference_client
    )
    
    try:
        # Test original system first
        print("🧪 Testing original system...")
        original_result = await evaluator.evaluate_system(
            system, dataset[:1], inference_client
        )
        print(f"   Original ROUGE-L: {original_result.get('rouge_l', 0):.3f}")
        print()
        
        # Run optimization
        result = await optimizer.optimize(system, dataset, max_generations=8)
        
        # 7. Display detailed results
        print("✅ Summarization optimization completed!")
        print("=" * 60)
        print(f"🎯 Best ROUGE-L score: {result.best_score:.3f}")
        print(f"📈 Improvement: {((result.best_score - original_result.get('rouge_l', 0)) / original_result.get('rouge_l', 1) * 100):+.1f}%")
        print(f"🔄 Total rollouts: {result.total_rollouts}")
        print(f"💰 Total cost: ${result.total_cost:.4f}")
        print()
        
        # Show optimized prompts
        print("🧠 Optimized System:")
        print("-" * 40)
        
        analyzer_prompt = result.best_system.modules["analyzer"].prompt
        summarizer_prompt = result.best_system.modules["summarizer"].prompt
        
        print("📊 Analyzer Module:")
        print(analyzer_prompt[:200] + "..." if len(analyzer_prompt) > 200 else analyzer_prompt)
        print()
        
        print("📝 Summarizer Module:")
        print(summarizer_prompt[:200] + "..." if len(summarizer_prompt) > 200 else summarizer_prompt)
        print()
        
        # 8. Test optimized system
        print("🧪 Testing optimized system on new text...")
        
        test_text = """
        Space exploration has entered a new era with private companies joining government agencies 
        in pursuing ambitious missions. SpaceX has revolutionized rocket technology with reusable 
        launch vehicles, significantly reducing costs. NASA's Artemis program aims to return humans 
        to the Moon by 2026, while private companies plan Mars colonization missions. The James Webb 
        Space Telescope has provided unprecedented views of distant galaxies, advancing our understanding 
        of the universe's origins. International collaboration remains crucial, with the International 
        Space Station serving as a platform for scientific research and diplomacy.
        """
        
        print("Input text:")
        print(test_text[:150] + "...")
        print()
        
        # Simulate system execution (in practice, you'd call: await result.best_system.execute(...))
        print("🚀 Generated summary:")
        print("(In practice, this would show the actual optimized system output)")
        print()
        
        # Show Pareto frontier analysis
        print("📊 Pareto Frontier Analysis:")
        print(f"   Solutions in frontier: {result.pareto_frontier.size()}")
        
        # Get diverse samples from frontier
        diverse_candidates = result.pareto_frontier.get_diverse_sample(3)
        for i, candidate in enumerate(diverse_candidates, 1):
            rouge_score = candidate.scores.get('rouge_l', 0)
            f1_score = candidate.scores.get('f1_score', 0)
            cost = candidate.cost
            print(f"   Candidate {i}: ROUGE-L={rouge_score:.3f}, F1={f1_score:.3f}, Cost=${cost:.4f}")
        
        print()
        
        # Show optimization statistics
        stats = optimizer.get_statistics()
        print("📈 Optimization Statistics:")
        print(f"   Generations: {stats.get('generations', 0)}")
        print(f"   Successful mutations: {stats.get('successful_mutations', 0)}")
        print(f"   Best generation: {stats.get('best_generation', 0)}")
        print(f"   Convergence: {'Yes' if stats.get('converged', False) else 'No'}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if hasattr(inference_client, 'close'):
            await inference_client.close()
        
        print("\n🎉 Text summarization example completed!")
        print("\nKey takeaways:")
        print("- Multi-step systems can be optimized effectively with GEPA")
        print("- ROUGE-L is particularly effective for summarization tasks")
        print("- The compress mutation type helps reduce prompt verbosity")
        print("- Pareto optimization balances quality vs. cost trade-offs")
        
        print("\nNext steps:")
        print("- Try examples/code_generation.py for code synthesis optimization")
        print("- Explore examples/custom_system.py for complex multi-agent workflows")


if __name__ == "__main__":
    asyncio.run(main())
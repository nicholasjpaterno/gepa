#!/usr/bin/env python3
"""
Code Generation Example

This example demonstrates optimizing a code generation system using GEPA.
The system generates Python functions based on natural language descriptions
and evaluates them using code execution metrics.

Run with: python examples/code_generation.py
"""

import asyncio
import os
from typing import List, Dict, Any

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import CodeExecutionMetric, ExactMatch, F1Score
from gepa.inference.factory import InferenceFactory


# Sample coding problems (inspired by HumanEval dataset)
CODING_PROBLEMS = [
    {
        "description": "Write a function that takes a list of numbers and returns the sum of all even numbers.",
        "function_name": "sum_even_numbers",
        "test_cases": [
            {"input": [1, 2, 3, 4, 5, 6], "expected_output": 12},
            {"input": [1, 3, 5], "expected_output": 0},
            {"input": [2, 4, 6, 8], "expected_output": 20},
            {"input": [], "expected_output": 0}
        ]
    },
    {
        "description": "Write a function that checks if a string is a palindrome (ignoring case and spaces).",
        "function_name": "is_palindrome",
        "test_cases": [
            {"input": "racecar", "expected_output": True},
            {"input": "A man a plan a canal Panama", "expected_output": True},
            {"input": "hello", "expected_output": False},
            {"input": "Madam", "expected_output": True}
        ]
    },
    {
        "description": "Write a function that finds the maximum element in a list of numbers.",
        "function_name": "find_maximum",
        "test_cases": [
            {"input": [1, 5, 3, 9, 2], "expected_output": 9},
            {"input": [-1, -5, -3], "expected_output": -1},
            {"input": [42], "expected_output": 42},
            {"input": [0, 0, 0], "expected_output": 0}
        ]
    },
    {
        "description": "Write a function that counts the number of vowels in a string (case insensitive).",
        "function_name": "count_vowels",
        "test_cases": [
            {"input": "hello", "expected_output": 2},
            {"input": "AEIOU", "expected_output": 5},
            {"input": "xyz", "expected_output": 0},
            {"input": "Programming", "expected_output": 3}
        ]
    },
    {
        "description": "Write a function that returns the factorial of a non-negative integer.",
        "function_name": "factorial",
        "test_cases": [
            {"input": 5, "expected_output": 120},
            {"input": 0, "expected_output": 1},
            {"input": 1, "expected_output": 1},
            {"input": 4, "expected_output": 24}
        ]
    }
]


def create_dataset_from_problems(problems: List[Dict]) -> List[Dict[str, Any]]:
    """Convert coding problems to GEPA dataset format."""
    dataset = []
    
    for problem in problems:
        # Create test execution context
        test_context = {
            "function_name": problem["function_name"],
            "test_cases": problem["test_cases"]
        }
        
        dataset.append({
            "description": problem["description"],
            "function_name": problem["function_name"],
            "expected": test_context  # CodeExecutionMetric will use this
        })
    
    return dataset


async def create_code_generation_system() -> CompoundAISystem:
    """Create a code generation system with planning and implementation steps."""
    
    return CompoundAISystem(
        modules={
            "planner": LanguageModule(
                id="planner",
                prompt="""You are an expert programmer. Analyze the following problem and create a plan:

Problem: {description}
Function name: {function_name}

Create a step-by-step plan for implementing this function:
1. Understand the requirements
2. Identify edge cases
3. Plan the algorithm approach
4. Consider implementation details

Plan:""",
                model_weights="gpt-4"
            ),
            "coder": LanguageModule(
                id="coder",
                prompt="""Based on the plan provided, implement the Python function.

Requirements:
- Write clean, efficient Python code
- Handle edge cases appropriately
- Include proper error handling if needed
- Follow Python naming conventions
- Only return the function code, no explanations

Plan: {plan}
Function name: {function_name}

Python code:
```python
def {function_name}():
    # Your implementation here
    pass
```

Implementation:""",
                model_weights="gpt-4"
            )
        },
        control_flow=SequentialFlow(["planner", "coder"]),
        input_schema=IOSchema(
            fields={"description": str, "function_name": str},
            required=["description", "function_name"]
        ),
        output_schema=IOSchema(
            fields={"code": str},
            required=["code"]
        ),
        system_id="code_generator"
    )


async def main():
    """Run the code generation optimization example."""
    
    print("💻 GEPA Code Generation Optimization")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # 1. Create the code generation system
    print("🏗️  Creating code generation system...")
    system = await create_code_generation_system()
    
    print(f"   System: {system.system_id}")
    print(f"   Modules: {list(system.modules.keys())}")
    print(f"   Two-step process: Planning → Implementation")
    print()
    
    # 2. Prepare coding dataset
    print("📊 Preparing coding problems dataset...")
    dataset = create_dataset_from_problems(CODING_PROBLEMS)
    
    print(f"   Problems: {len(dataset)}")
    print("   Sample problems:")
    for i, problem in enumerate(dataset[:3]):
        print(f"     {i+1}. {problem['function_name']}: {problem['description'][:50]}...")
    print()
    
    # 3. Configure GEPA for code generation
    print("⚙️  Configuring GEPA for code generation...")
    
    config = GEPAConfig(
        inference={
            "provider": "openai",
            "model": "gpt-4",  # GPT-4 is better for code generation
            "api_key": api_key,
            "max_tokens": 1500,  # More tokens for code
            "temperature": 0.1   # Low temperature for consistent code
        },
        optimization={
            "budget": 30,  # Moderate budget for code tasks
            "pareto_set_size": 6,
            "minibatch_size": 2,  # Small batches due to execution time
            "enable_crossover": True,
            "crossover_probability": 0.3,
            "mutation_types": ["rewrite", "insert"],  # Good for code prompts
            "early_stopping_patience": 4
        },
        database={
            "url": "sqlite:///gepa_code_generation.db"
        },
        observability={
            "log_level": "INFO"
        }
    )
    
    # 4. Create code-specific evaluator
    print("📈 Setting up code execution evaluation...")
    
    evaluator = SimpleEvaluator([
        CodeExecutionMetric(
            name="code_execution",
            timeout=5.0,  # 5 second timeout for code execution
            safe_mode=True  # Enable sandboxed execution
        ),
        ExactMatch(name="exact_syntax"),  # For exact code matches
        F1Score(name="code_similarity")   # For code token similarity
    ])
    
    print("   Primary metric: Code Execution (pass/fail)")
    print("   Secondary metrics: Exact match, Code similarity")
    print("   Safety: Sandboxed execution with 5s timeout")
    print()
    
    # 5. Create inference client
    inference_client = InferenceFactory.create_client(config.inference)
    
    # 6. Run optimization
    print("🔄 Starting code generation optimization...")
    print("   Note: This involves executing generated code safely")
    print()
    
    optimizer = GEPAOptimizer(
        config=config,
        evaluator=evaluator,
        inference_client=inference_client
    )
    
    try:
        # Test original system on one problem
        print("🧪 Testing original system...")
        sample_problem = dataset[0]
        print(f"   Problem: {sample_problem['description']}")
        
        # In practice, you'd run the system here
        print("   (Original system baseline would be measured here)")
        print()
        
        # Run optimization
        result = await optimizer.optimize(system, dataset, max_generations=6)
        
        # 7. Display results
        print("✅ Code generation optimization completed!")
        print("=" * 60)
        print(f"🎯 Best execution score: {result.best_score:.3f}")
        print(f"📊 Success rate: {result.best_score * 100:.1f}% of tests passed")
        print(f"🔄 Total rollouts: {result.total_rollouts}")
        print(f"💰 Total cost: ${result.total_cost:.4f}")
        print()
        
        # Show optimized prompts
        print("🧠 Optimized System Prompts:")
        print("-" * 40)
        
        planner_prompt = result.best_system.modules["planner"].prompt
        coder_prompt = result.best_system.modules["coder"].prompt
        
        print("📋 Planning Module:")
        print(planner_prompt[:300] + "..." if len(planner_prompt) > 300 else planner_prompt)
        print()
        
        print("⌨️  Coding Module:")
        print(coder_prompt[:300] + "..." if len(coder_prompt) > 300 else coder_prompt)
        print()
        
        # 8. Test optimized system on a new problem
        print("🧪 Testing optimized system on new problem...")
        
        new_problem = {
            "description": "Write a function that reverses a string without using built-in reverse methods.",
            "function_name": "reverse_string",
            "test_cases": [
                {"input": "hello", "expected_output": "olleh"},
                {"input": "Python", "expected_output": "nohtyP"},
                {"input": "", "expected_output": ""},
                {"input": "a", "expected_output": "a"}
            ]
        }
        
        print(f"   New problem: {new_problem['description']}")
        print("   Expected function: reverse_string")
        print()
        print("   Generated solution:")
        print("   (In practice, the optimized system would generate code here)")
        
        # Example of what might be generated
        example_code = '''
def reverse_string(s):
    """Reverse a string without using built-in reverse methods."""
    result = ""
    for char in s:
        result = char + result
    return result
        '''
        print(example_code)
        
        # 9. Show Pareto frontier analysis
        print("📊 Pareto Frontier Analysis:")
        print(f"   Solutions in frontier: {result.pareto_frontier.size()}")
        
        diverse_candidates = result.pareto_frontier.get_diverse_sample(3)
        for i, candidate in enumerate(diverse_candidates, 1):
            exec_score = candidate.scores.get('code_execution', 0)
            similarity = candidate.scores.get('code_similarity', 0)
            cost = candidate.cost
            print(f"   Solution {i}: Execution={exec_score:.3f}, Similarity={similarity:.3f}, Cost=${cost:.4f}")
        
        print()
        
        # 10. Optimization insights
        stats = optimizer.get_statistics()
        print("📈 Optimization Insights:")
        print(f"   Generations completed: {stats.get('generations', 0)}")
        print(f"   Successful mutations: {stats.get('successful_mutations', 0)}")
        print(f"   Code execution improvements: {stats.get('execution_improvements', 0)}")
        print(f"   Average tokens per solution: {stats.get('avg_tokens', 0)}")
        
        # Code generation specific insights
        print("\n💡 Code Generation Insights:")
        print("   - Planning step helps with complex logic")
        print("   - Lower temperature reduces syntax errors")
        print("   - Code execution metric provides objective feedback")
        print("   - Mutation helps discover better algorithmic approaches")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if hasattr(inference_client, 'close'):
            await inference_client.close()
        
        print("\n🎉 Code generation example completed!")
        
        print("\nKey learnings:")
        print("- Code execution metrics provide objective evaluation")
        print("- Two-step planning improves code quality")
        print("- Low temperature settings reduce syntax errors")
        print("- GEPA can optimize both correctness and efficiency")
        
        print("\nNext examples to try:")
        print("- examples/multi_provider.py - Compare different LLM providers")
        print("- examples/custom_metrics.py - Create domain-specific metrics")
        print("- examples/advanced_system.py - Multi-agent code review system")


if __name__ == "__main__":
    asyncio.run(main())
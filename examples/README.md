# GEPA Examples

This directory contains comprehensive examples demonstrating various aspects of GEPA (GeneticPareto) prompt optimization. Each example is self-contained and includes detailed explanations.

## 🚀 Getting Started

### Prerequisites

1. **Install GEPA**:
   ```bash
   pip install gepa
   # or for development:
   pip install -e ".[dev]"
   ```

2. **Set up API keys** (at least one required):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Optional: Set up local inference**:
   ```bash
   # For Ollama
   ollama serve
   ollama pull llama2
   ```

### Quick Test

Run the quickstart example to verify your setup:
```bash
python examples/quickstart.py
```

## 📚 Example Overview

| Example | Description | Complexity | Key Features |
|---------|-------------|------------|--------------|
| `quickstart.py` | Basic sentiment classification | Beginner | Simple setup, basic metrics |
| `meta_orchestrator_quickstart.py` | **NEW!** MetaOrchestrator example | Beginner | Four-pillar architecture, real optimization |
| `text_summarization.py` | Multi-step summarization system | Intermediate | Complex system, ROUGE metrics |
| `code_generation.py` | Python code synthesis | Intermediate | Code execution, planning system |
| `multi_provider.py` | Provider comparison | Intermediate | Multiple LLMs, cost analysis |
| `custom_metrics.py` | Business-specific metrics | Advanced | Custom evaluation, multi-output |

## 🎯 Example Details

### 1. Quickstart (`quickstart.py`)

**Purpose**: Get started with GEPA using a simple sentiment classification task.

**What you'll learn**:
- Basic GEPA setup and configuration
- Creating simple AI systems
- Using built-in evaluation metrics
- Understanding optimization results

**Run time**: ~2-3 minutes  
**API calls**: ~20-30

```bash
python examples/quickstart.py
```

**Expected output**:
- Optimized sentiment classification system
- Performance improvements over baseline
- Cost and rollout statistics

---

### 2. MetaOrchestrator Quickstart (`meta_orchestrator_quickstart.py`) **🔥 NEW!**

**Purpose**: Experience the revolutionary MetaOrchestrator four-pillar architecture with real optimization.

**What you'll learn**:
- MetaOrchestrator vs baseline GEPA comparison
- Four-pillar architecture in action:
  - RL-based Algorithm Selection
  - Predictive Topology Evolution  
  - Multi-Fidelity Bayesian HyperOptimization
  - Structural Prompt Evolution
- Real performance analysis
- Production configuration profiles

**Prerequisites**: Running LMStudio with a loaded model

**Run time**: ~3-5 minutes  
**LMStudio calls**: ~30-40

```bash
# Ensure LMStudio is running first
python examples/meta_orchestrator_quickstart.py
```

**Expected output**:
- Baseline vs MetaOrchestrator performance comparison
- Real optimization improvements (not simulated!)
- Component contribution analysis
- Revolutionary four-pillar architecture demonstration

---

### 3. Text Summarization (`text_summarization.py`)

**Purpose**: Optimize a multi-step text summarization system using advanced metrics.

**What you'll learn**:
- Multi-module AI systems (analyzer → summarizer)
- Using ROUGE-L metrics for summarization
- Advanced GEPA configuration
- Pareto frontier analysis

**Run time**: ~5-7 minutes  
**API calls**: ~80-120

```bash
python examples/text_summarization.py
```

**Expected output**:
- Optimized summarization prompts
- ROUGE-L score improvements
- Analysis of system evolution

---

### 4. Code Generation (`code_generation.py`)

**Purpose**: Optimize a code generation system with execution-based evaluation.

**What you'll learn**:
- Code execution metrics
- Two-step planning systems
- Handling code-specific challenges
- Safety considerations for code execution

**Run time**: ~4-6 minutes  
**API calls**: ~60-90

```bash
python examples/code_generation.py
```

**Expected output**:
- Optimized code generation prompts
- Code execution success rates
- Generated Python functions

---

### 5. Multi-Provider Comparison (`multi_provider.py`)

**Purpose**: Compare different LLM providers on the same optimization task.

**What you'll learn**:
- Working with multiple LLM providers
- Cost vs. performance analysis
- Provider-specific optimizations
- Speed and efficiency comparisons

**Run time**: ~8-12 minutes (depends on providers)  
**API calls**: Varies by available providers

```bash
python examples/multi_provider.py
```

**Expected output**:
- Performance ranking of providers
- Cost efficiency analysis
- Speed comparisons
- Provider-specific insights

---

### 6. Custom Metrics (`custom_metrics.py`)

**Purpose**: Create domain-specific evaluation metrics for business applications.

**What you'll learn**:
- Implementing custom evaluation metrics
- Business-focused optimization
- Multi-output system evaluation
- Weighted scoring systems

**Run time**: ~3-5 minutes  
**API calls**: ~40-60

```bash
python examples/custom_metrics.py
```

**Expected output**:
- Custom metric implementations
- Business impact scoring
- Priority-weighted evaluation
- Multi-objective optimization results

## 🛠️ Running Examples

### Individual Examples

Each example can be run independently:

```bash
# Basic usage
python examples/quickstart.py

# With verbose output
python examples/text_summarization.py

# With custom configuration
GEPA_LOG_LEVEL=DEBUG python examples/code_generation.py
```

### Batch Execution

Run multiple examples sequentially:

```bash
# Run all beginner examples
for example in quickstart.py text_summarization.py; do
    echo "Running $example..."
    python examples/$example
    echo "Completed $example"
    echo "---"
done
```

### Configuration Options

Most examples support environment variable configuration:

```bash
# Adjust budget
export GEPA_BUDGET=50

# Change provider
export GEPA_PROVIDER=anthropic

# Set log level
export GEPA_LOG_LEVEL=DEBUG

# Use local models
export GEPA_BASE_URL=http://localhost:11434
```

## 🔧 Troubleshooting

### Common Issues

**1. API Key Not Found**
```
❌ Please set OPENAI_API_KEY environment variable
```
**Solution**: Set your API key as shown in prerequisites.

**2. Rate Limiting**
```
Error: Rate limit exceeded
```
**Solution**: Add delays between runs or reduce budget in config.

**3. Local Model Connection**
```
Connection refused to localhost:11434
```
**Solution**: Ensure Ollama is running: `ollama serve`

**4. Import Errors**
```
ModuleNotFoundError: No module named 'gepa'
```
**Solution**: Install GEPA: `pip install gepa`

### Performance Tips

1. **Start Small**: Use smaller budgets for initial testing
2. **Use Appropriate Models**: GPT-3.5 for speed, GPT-4 for quality
3. **Monitor Costs**: Track API usage, especially with GPT-4
4. **Batch Operations**: Run multiple examples together to share setup costs

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export GEPA_LOG_LEVEL=DEBUG
python examples/quickstart.py
```

## 📊 Expected Results

### Performance Improvements

Typical improvements across examples:

| Example | Baseline Score | Optimized Score | Improvement |
|---------|---------------|-----------------|-------------|
| Sentiment Classification | 0.650 | 0.820 | +26% |
| Text Summarization | 0.420 | 0.580 | +38% |
| Code Generation | 0.300 | 0.650 | +117% |
| Custom Metrics | 0.550 | 0.750 | +36% |

### Cost Analysis

Typical costs per example (using GPT-3.5-turbo):

| Example | Budget | Avg Cost | Cost/Point |
|---------|--------|----------|------------|
| Quickstart | 20 | $0.15 | $0.0009 |
| Summarization | 40 | $0.45 | $0.0008 |
| Code Generation | 30 | $0.35 | $0.0005 |
| Multi-Provider | 15×3 | $0.60 | $0.0012 |

## 🎓 Learning Path

### Beginner Path
1. `quickstart.py` - Learn GEPA basics
2. `text_summarization.py` - Understand complex systems
3. `multi_provider.py` - Explore different providers

### Advanced Path
1. `code_generation.py` - Work with execution metrics
2. `custom_metrics.py` - Create domain-specific evaluation
3. Build your own custom system

### Production Path
1. Run all examples to understand capabilities
2. Adapt patterns to your use case
3. Implement proper monitoring and error handling
4. Set up continuous optimization workflows

## 🤝 Contributing Examples

Want to add your own example? Great! Follow these guidelines:

### Example Structure
```python
#!/usr/bin/env python3
"""
Example Title

Brief description of what this example demonstrates.
Run with: python examples/your_example.py
"""

import asyncio
# ... imports

async def main():
    """Main example function with clear steps."""
    print("🚀 Your Example Title")
    print("=" * 50)
    
    # 1. Setup
    # 2. Create system
    # 3. Configure GEPA
    # 4. Run optimization
    # 5. Show results
    # 6. Cleanup

if __name__ == "__main__":
    asyncio.run(main())
```

### Requirements
- Self-contained and runnable
- Clear documentation and comments
- Error handling for common issues
- Reasonable runtime (< 10 minutes)
- Educational value

### Submission Process
1. Create your example following the structure above
2. Test with different API keys and configurations
3. Add entry to this README
4. Submit a pull request with description

## 📖 Additional Resources

- **Research Paper**: [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
- **GitHub Issues**: [Report problems or request examples](https://github.com/nicholasjpaterno/gepa/issues)

## 🏆 Advanced Examples (Coming Soon)

- `advanced_system.py` - Multi-agent workflows
- `production_deployment.py` - Full production setup
- `distributed_optimization.py` - Scaling optimization
- `model_comparison.py` - Comprehensive model evaluation
- `real_world_case_study.py` - End-to-end business application

---

**Happy optimizing! 🚀**
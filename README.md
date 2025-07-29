# GEPA: GeneticPareto Prompt Optimization

[![CI](https://github.com/nicholasjpaterno/gepa/workflows/CI/badge.svg)](https://github.com/nicholasjpaterno/gepa/actions)
[![PyPI version](https://badge.fury.io/py/gepa.svg)](https://badge.fury.io/py/gepa)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **🚀 Production-ready prompt optimization that outperforms reinforcement learning with 35x fewer rollouts**

GEPA (GeneticPareto) is a cutting-edge prompt optimization framework that uses evolutionary algorithms with Pareto frontier management to automatically improve your prompts. Based on the research paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457), GEPA achieves **10% better performance than GRPO with 35x fewer rollouts**.


## ✨ Key Features

- **🔧 Provider Agnostic**: Works with OpenAI, Anthropic, Ollama, LMStudio, and any local inference setup
- **⚡ High Performance**: 35x more efficient than traditional reinforcement learning approaches  
- **🧠 Reflective Evolution**: Uses natural language feedback for intelligent prompt mutations
- **📊 Multi-Objective**: Optimizes for multiple metrics simultaneously using Pareto frontier management
- **🏗️ Production Ready**: Built with async/await, comprehensive observability, and robust error handling
- **📈 Built-in Metrics**: Comprehensive evaluation with ExactMatch, F1Score, ROUGE-L, BLEU, and code execution
- **🐳 Cloud Native**: Docker support with monitoring stack (Prometheus, Grafana, Jaeger)
- **🔄 Easy Integration**: Simple Python API with extensive configuration options

## 🚀 Quick Start

### Installation

```bash
# coming soon
```

For development installation:
```bash
git clone https://github.com/nicholasjpaterno/gepa.git
cd gepa
pip install -e ".[dev]"
```

### Basic Usage

```python
import asyncio
from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score

# Define your AI system
system = CompoundAISystem(
    modules={
        "analyzer": LanguageModule(
            id="analyzer",
            prompt="Analyze this text carefully: {text}",
            model_weights="gpt-4"
        ),
        "summarizer": LanguageModule(
            id="summarizer", 
            prompt="Create a concise summary: {analysis}",
            model_weights="gpt-4"
        )
    },
    control_flow=SequentialFlow(["analyzer", "summarizer"])
)

# Configure GEPA
config = GEPAConfig(
    inference={"provider": "openai", "model": "gpt-4", "api_key": "your-key"},
    optimization={"budget": 50, "pareto_set_size": 10}
)

# Your training dataset  
dataset = [
    {"text": "Long article about AI...", "expected": "AI summary"},
    {"text": "Complex research paper...", "expected": "Research summary"},
    # ... more examples
]

# Run optimization
async def optimize_prompts():
    evaluator = SimpleEvaluator([ExactMatch(), F1Score()])
    optimizer = GEPAOptimizer(config, evaluator)
    
    result = await optimizer.optimize(system, dataset)
    
    print(f"🎯 Best score: {result.best_score:.3f}")
    print(f"🔄 Rollouts used: {result.total_rollouts}")
    print(f"💰 Total cost: ${result.total_cost:.4f}")
    
    return result.best_system

asyncio.run(optimize_prompts())
```

### CLI Usage

```bash
# Create example configuration
gepa init my-optimization

# Run optimization
gepa optimize --config config.yaml --dataset data.json --budget 100

# Monitor progress
gepa status <run-id>

# Export results
gepa export <run-id> --format json
```

### Quick Testing with LMStudio

```bash
# Install dependencies
make install

# Test with your LMStudio setup
make test-lmstudio

# Run full optimization example (see real 400% improvement!)
make test-lmstudio-optimize

# View all available commands
make help
```

## 🏗️ Architecture

GEPA is built with a modular, production-ready architecture:

```
gepa/
├── core/           # Core GEPA algorithm implementation
├── inference/      # Provider abstraction layer  
├── evaluation/     # Metrics and evaluation framework
├── database/       # Data persistence and ORM
├── observability/  # Metrics, tracing, and logging
├── config/         # Configuration management
└── cli/           # Command-line interface
```

### Core Algorithm (Algorithm 1 from paper)

1. **🎯 Pareto-based Candidate Sampling**: Maintains diversity while focusing on high-performing candidates
2. **🧠 Reflective Prompt Mutation**: Uses LLM-based reflection to analyze failures and improve prompts  
3. **🔄 System Aware Merge**: Optional crossover operations between complementary candidates
4. **📊 Budget-aware Optimization**: Tracks token usage and costs throughout optimization

### Key Components

- **GEPAOptimizer**: Main optimization engine implementing Algorithm 1 from the paper
- **ParetoFrontier**: Multi-objective optimization with diversity maintenance
- **InferenceClient**: Unified interface for all LLM providers
- **CompoundAISystem**: Framework for defining complex AI workflows  
- **DatabaseManager**: Persistent storage for optimization runs and results

## ⚙️ Configuration

### Using Different Providers

```python
# OpenAI
config = GEPAConfig(
    inference={"provider": "openai", "model": "gpt-4", "api_key": "sk-..."}
)

# Anthropic  
config = GEPAConfig(
    inference={"provider": "anthropic", "model": "claude-3-opus", "api_key": "sk-ant-..."}
)

# Local Ollama
config = GEPAConfig(
    inference={"provider": "ollama", "model": "llama2", "base_url": "http://localhost:11434"}
)
```

### Advanced Configuration

```python
from gepa.config import GEPAConfig, OptimizationConfig, DatabaseConfig

config = GEPAConfig(
    optimization=OptimizationConfig(
        budget=100,                    # Total evaluation budget
        pareto_set_size=15,           # Size of Pareto frontier
        enable_crossover=True,        # Enable crossover operations
        crossover_probability=0.3,    # Probability of crossover
        mutation_types=["rewrite", "insert", "delete"],
        minibatch_size=5             # Batch size for evaluation
    ),
    database=DatabaseConfig(
        url="postgresql://user:pass@localhost/gepa"
    ),
    observability={
        "metrics_enabled": True,
        "tracing_enabled": True, 
        "log_level": "INFO"
    }
)
```

### YAML Configuration

```yaml
inference:
  provider: "openai"  # openai, anthropic, ollama, etc.
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  max_tokens: 4096
  temperature: 0.7

optimization:
  budget: 100  # Maximum rollouts
  minibatch_size: 5
  pareto_set_size: 10
  enable_crossover: true
  crossover_probability: 0.3
  mutation_types: ["rewrite", "insert", "delete"]

database:
  url: "postgresql://gepa:gepa@localhost/gepa"

observability:
  metrics_enabled: true
  tracing_enabled: true
  log_level: "INFO"
```

## 🔌 Supported Providers

### Cloud Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
- **Anthropic**: Claude-3 (Opus, Sonnet, Haiku)
- **Together AI**: Various open source models
- **Hugging Face**: Inference endpoints

### Local Providers  
- **Ollama**: Local model serving
- **LM Studio**: Local model serving
- **vLLM**: High-performance inference server
- **llama.cpp**: CPU inference

## 📊 Evaluation Metrics

GEPA supports multiple built-in evaluation metrics:

```python
from gepa.evaluation.metrics import ExactMatch, F1Score, RougeL, BLEU, CodeExecutionMetric
from gepa.evaluation.base import SimpleEvaluator

evaluator = SimpleEvaluator([
    ExactMatch(name="exact_match"),
    F1Score(name="f1_score"),
    RougeL(name="rouge_l"),
    BLEU(name="bleu", n_gram=2),
    CodeExecutionMetric(name="code_exec", timeout=5.0)
])
```

Built-in metrics:
- **Exact Match**: Exact string matching
- **F1 Score**: Token-level F1 score
- **ROUGE-L**: Longest common subsequence
- **BLEU**: N-gram overlap score
- **Code Execution**: For code generation tasks
- **Semantic Similarity**: Using sentence transformers

Custom metrics can be easily added by extending the `Metric` base class.

## 🔄 Mutation Types

GEPA supports four types of reflective mutations:

1. **Rewrite**: Complete prompt rewriting based on failure analysis
2. **Insert**: Adding instructions, examples, or constraints
3. **Delete**: Removing unnecessary or harmful elements
4. **Compress**: Reducing token count while maintaining performance

## 🐳 Production Deployment

### Docker Compose (Recommended)

```bash
# Clone and start the full stack
git clone https://github.com/nicholasjpaterno/gepa.git
cd gepa
docker-compose up -d
```

This launches:
- GEPA optimization service
- PostgreSQL database
- Redis for caching
- Prometheus for metrics
- Grafana for dashboards
- Jaeger for distributed tracing

### Kubernetes

```yaml
# See examples/kubernetes/ for complete manifests
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gepa
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gepa
  template:
    metadata:
      labels:
        app: gepa
    spec:
      containers:
      - name: gepa
        image: gepa/gepa:latest
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres:5432/gepa"
```

## 📖 Examples

### Text Summarization

```python
from gepa.examples.summarization import optimize_summarization_system

# Optimize a summarization system
result = await optimize_summarization_system(
    dataset_path="data/espn_dailyresults.json",
    budget=100,
    provider="openai"
)
```

### Code Generation

```python
from gepa.examples.code_generation import optimize_code_system

# Optimize code generation prompts
result = await optimize_code_system(
    dataset_path="data/humaneval.json",
    budget=150,
    provider="anthropic"
)
```

### Question Answering

```python
from gepa.examples.qa import optimize_qa_system

# Optimize QA system
result = await optimize_qa_system(
    dataset_path="data/squad.json",
    budget=75,
    provider="ollama",
    model="llama2"
)
```

## 📊 Monitoring & Observability

GEPA includes comprehensive observability:

### Metrics
- Optimization progress and convergence
- Cost tracking across providers
- Performance metrics (latency, throughput)
- Error rates and success metrics

### Tracing
- End-to-end request tracing
- Individual module performance
- Database query optimization
- External API call monitoring

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response logging
- Error tracking with context

Access dashboards at:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686

## 📊 Research Results

From the original paper:
- **10% average improvement** over GRPO (Group Relative Policy Optimization)
- **Up to 20% improvement** on individual tasks
- **35x fewer rollouts** required compared to traditional RL
- **10%+ improvement** over MIPROv2 across two LLMs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- New inference providers
- Additional evaluation metrics
- Advanced mutation operators
- Visualization tools
- Documentation improvements

### Development Setup

```bash
# Clone the repository
git clone https://github.com/nicholasjpaterno/gepa.git
cd gepa

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/ tests/
black src/ tests/

# Type checking
mypy src/gepa
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires running services)
docker-compose up -d postgres redis
pytest tests/integration/

# All tests with coverage
pytest --cov=gepa --cov-report=html
```

## 📝 Research & Citation

GEPA is based on the research paper:

```bibtex
@article{gepa2025,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A. and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J. and Jiang, Meng and Potts, Christopher and Sen, Koushik and Dimakis, Alexandros G. and Stoica, Ion and Klein, Dan and Zaharia, Matei and Khattab, Omar},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025},
  month={July}
}
```

**Authors:**
- **Lakshya A. Agrawal** (UC Berkeley)
- **Shangyin Tan** (UC Berkeley) 
- **Dilara Soylu** (Stanford University)
- **Noah Ziems** (Notre Dame)
- **Rishi Khare** (UC Berkeley)
- **Krista Opsahl-Ong** (Databricks)
- **Arnav Singhvi** (Stanford University, Databricks)
- **Herumb Shandilya** (Stanford University)
- **Michael J. Ryan** (Stanford University)
- **Meng Jiang** (Notre Dame)
- **Christopher Potts** (Stanford University)
- **Koushik Sen** (UC Berkeley)
- **Alexandros G. Dimakis** (UC Berkeley, BespokeLabs.ai)
- **Ion Stoica** (UC Berkeley)
- **Dan Klein** (UC Berkeley)
- **Matei Zaharia** (UC Berkeley, Databricks)
- **Omar Khattab** (MIT)

## 🔒 Security

- Security policy: [SECURITY.md](SECURITY.md)
- Vulnerability reporting: Create a GitHub security advisory
- Automated security scanning in CI/CD
- Regular dependency updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 **Documentation**: See [README.md](README.md) and [examples/](examples/)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/nicholasjpaterno/gepa/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/nicholasjpaterno/gepa/issues)

## 🚀 Roadmap

- [ ] **v0.2**: Multi-agent optimization support
- [ ] **v0.3**: Advanced crossover strategies
- [ ] **v0.4**: Distributed optimization
- [ ] **v0.5**: AutoML integration
- [ ] **v1.0**: Production-grade stability

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nicholasjpaterno/gepa&type=Date)](https://star-history.com/#nicholasjpaterno/gepa&Date)

---

**Made with ❤️ by the GEPA community**

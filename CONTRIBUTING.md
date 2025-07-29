# Contributing to GEPA

Thank you for your interest in contributing to GEPA! We welcome contributions from the community and are excited to work with you to make GEPA even better.

## 🌟 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new features or improvements
- **📖 Documentation**: Improve our docs, examples, and tutorials
- **🔧 Code Contributions**: Add new features, fix bugs, or improve performance
- **🧪 Testing**: Add test cases and improve test coverage
- **🎨 Design**: Improve UI/UX for dashboards and visualizations
- **📊 Research**: Contribute to algorithmic improvements and benchmarking

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/gepa.git
cd gepa

# Add upstream remote
git remote add upstream https://github.com/nicholasjpaterno/gepa.git
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### 3. Set Up Local Services

```bash
# Start PostgreSQL and Redis for testing
docker-compose up -d postgres redis

# Run database migrations
python -m gepa.database.migrations upgrade head
```

### 4. Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/unit/

# Run linting
ruff check src/ tests/
black --check src/ tests/

# Type checking
mypy src/gepa
```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Keep your main branch up to date
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test files
pytest tests/unit/test_optimizer.py -v

# Run with coverage
pytest --cov=gepa --cov-report=html
```

### 4. Lint and Format

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Type checking
mypy src/gepa
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for new inference provider"

# Push to your fork
git push origin feature/your-feature-name
```

### 6. Submit Pull Request

1. Go to the GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template
5. Submit for review

## 📝 Code Style Guidelines

### Python Code Style

We use [Black](https://black.readthedocs.io/) for code formatting and [Ruff](https://docs.astral.sh/ruff/) for linting.

```python
# Good: Clear function names and docstrings
async def optimize_prompt_system(
    system: CompoundAISystem,
    dataset: List[Dict[str, Any]],
    config: OptimizationConfig
) -> OptimizationResult:
    """Optimize a compound AI system using GEPA algorithm.
    
    Args:
        system: The AI system to optimize
        dataset: Training dataset for evaluation
        config: Optimization configuration
        
    Returns:
        Optimization results including best system and metrics
    """
    # Implementation here...
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Document all public APIs
- Use type hints consistently

### Testing Style

```python
# Good: Descriptive test names and clear assertions
async def test_optimizer_respects_budget_limits():
    """Test that optimizer stops when budget is exhausted."""
    config = GEPAConfig(optimization={"budget": 10})
    optimizer = GEPAOptimizer(config, evaluator, client)
    
    result = await optimizer.optimize(system, dataset)
    
    assert result.total_rollouts <= 10
    assert result.best_system is not None
```

## 🏗️ Project Structure

```
gepa/
├── src/gepa/           # Main source code
│   ├── core/          # Core algorithm implementation
│   ├── inference/     # LLM provider integrations
│   ├── evaluation/    # Metrics and evaluation
│   ├── database/      # Data persistence
│   ├── observability/ # Monitoring and logging
│   ├── config/        # Configuration management
│   └── cli/          # Command-line interface
├── tests/             # Test suite
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── conftest.py   # Test configuration
├── examples/          # Example scripts and tutorials
├── docs/             # Documentation
└── .github/          # GitHub workflows and templates
```

## 🧪 Testing Guidelines

### Writing Tests

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_pareto_frontier_add_candidate():
    """Unit test for Pareto frontier."""
    # Test implementation

@pytest.mark.integration
async def test_optimizer_with_real_database():
    """Integration test with database."""
    # Test implementation

@pytest.mark.slow
async def test_full_optimization_workflow():
    """End-to-end test (takes longer to run)."""
    # Test implementation
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=gepa --cov-report=html

# Run specific test file
pytest tests/unit/test_optimizer.py -v
```

## 📋 Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Error messages or logs** if applicable

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### ✨ Feature Requests

For feature requests, please include:

- **Use case description** - why is this needed?
- **Proposed solution** - how should it work?
- **Alternatives considered** - other approaches you've thought of
- **Implementation ideas** - if you have technical suggestions

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### 🔧 Code Contributions

#### Adding New Inference Providers

1. Create a new client in `src/gepa/inference/providers/`
2. Inherit from `InferenceClient` base class
3. Implement required methods (`generate`, `health_check`, etc.)
4. Add provider to the factory in `src/gepa/inference/factory.py`
5. Add comprehensive tests
6. Update documentation

Example:
```python
from gepa.inference.base import InferenceClient

class NewProviderClient(InferenceClient):
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        # Implementation here
        pass
```

#### Adding New Evaluation Metrics

1. Create new metric class in `src/gepa/evaluation/metrics/`
2. Inherit from `Metric` base class
3. Implement `compute` and `batch_compute` methods
4. Add comprehensive tests
5. Update documentation

Example:
```python
from gepa.evaluation.base import Metric

class CustomMetric(Metric):
    def compute(self, predictions: List[str], references: List[str]) -> float:
        # Implementation here
        pass
```

#### Adding New Mutation Operators

1. Create new mutator in `src/gepa/core/mutation/`
2. Follow existing patterns from `ReflectiveMutator`
3. Add comprehensive tests
4. Update configuration options

## 📖 Documentation

### Documentation Structure

- **README.md**: Quick start and overview
- **docs/**: Comprehensive documentation
- **examples/**: Practical examples and tutorials
- **API Reference**: Auto-generated from docstrings

### Writing Documentation

- Use clear, beginner-friendly language
- Include practical examples
- Show both basic and advanced usage
- Keep examples up-to-date

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve locally
python -m http.server -d _build/html 8080
```

## 🔍 Code Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow our coding standards?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?

### Review Timeline

- **Initial Review**: Within 48 hours
- **Follow-up**: Within 24 hours of updates
- **Merge**: After approval from at least one maintainer

## 🏆 Recognition

Contributors are recognized in several ways:

- **README**: Listed in contributors section
- **Releases**: Mentioned in release notes
- **Social Media**: Highlighted on our channels
- **Special Recognition**: For significant contributions

## ❓ Getting Help

### Community Support

- **GitHub Discussions**: [Ask questions](https://github.com/nicholasjpaterno/gepa/discussions)
- **Issues**: [Report bugs or request features](https://github.com/nicholasjpaterno/gepa/issues)

### Maintainer Contact

- **General Questions**: Create a discussion
- **Security Issues**: Create a GitHub security advisory
- **Urgent Issues**: Mention @maintainers in GitHub

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

### Our Standards

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** not competitive

## 🎯 Contribution Ideas

Looking for ways to contribute? Here are some ideas:

### Beginner-Friendly
- Fix typos in documentation
- Add more examples to the examples/ directory
- Improve error messages
- Add more test cases

### Intermediate
- Add support for new LLM providers
- Implement new evaluation metrics
- Improve CLI user experience
- Add monitoring dashboards

### Advanced
- Optimize core algorithm performance
- Add distributed optimization support
- Implement advanced crossover strategies
- Contribute to research benchmarking

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Schedule

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly feature releases
- **Major releases**: Quarterly breaking changes

## 📊 Metrics and Success

We track contribution health through:

- **Response Time**: How quickly we respond to issues/PRs
- **Merge Rate**: Percentage of PRs successfully merged
- **Test Coverage**: Code coverage percentage
- **Community Growth**: Number of active contributors

---

Thank you for contributing to GEPA! Together, we're making prompt optimization more accessible and powerful for everyone. 🚀
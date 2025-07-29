# GEPA Testing Infrastructure

This directory contains organized testing infrastructure for various GEPA integrations and components.

## 📁 Directory Structure

```
testing/
├── README.md              # This file
└── lmstudio/              # LMStudio integration testing
    ├── README.md          # Comprehensive LMStudio testing guide
    ├── test_lmstudio.sh   # Main testing script
    ├── Dockerfile.test    # Docker container for testing
    └── docker-compose.test.yml  # Docker compose configuration
```

## 🎯 Available Test Suites

### LMStudio Integration (`testing/lmstudio/`)

**Purpose**: Test GEPA integration with LMStudio local inference server

**Features**:
- ✅ Connection and API compatibility testing
- ✅ Real GEPA optimization with tangible results (400% improvement achieved)
- ✅ Support for 18+ models
- ✅ Docker containerized testing
- ✅ Comprehensive result tracking and JSON export

**Quick Start**:
```bash
# From project root - Basic connection test
make test-lmstudio

# Full optimization example with results  
make test-lmstudio-optimize

# Direct access to testing suite
./testing/lmstudio/test_lmstudio.sh help
```

**Documentation**: See [testing/lmstudio/README.md](lmstudio/README.md)

## 🚀 Makefile Integration

For convenience, Make targets are provided in the project root:

- `make test-lmstudio` - Run LMStudio connection test
- `make test-lmstudio-optimize` - Run full optimization example
- `make help` - Show all available commands

The Makefile provides a clean interface to all testing infrastructure.

## 🏗️ Adding New Test Suites

To add a new testing suite (e.g., for OpenAI, Anthropic, etc.):

1. **Create directory**: `testing/[provider-name]/`
2. **Add test script**: `testing/[provider-name]/test_[provider].sh`
3. **Add Dockerfile**: `testing/[provider-name]/Dockerfile.test`
4. **Create README**: `testing/[provider-name]/README.md`
5. **Add Makefile targets**: Add `test-[provider]` and `test-[provider]-optimize` to root Makefile
6. **Update this README**: Add entry to "Available Test Suites"

### Template Structure
```
testing/[provider-name]/
├── README.md              # Testing guide
├── test_[provider].sh     # Main test script
├── Dockerfile.test        # Docker container
├── docker-compose.test.yml # Docker compose (if needed)
└── [provider]_config.yml  # Configuration (if needed)
```

## 🎯 Testing Philosophy

### Comprehensive Coverage
Each test suite should provide:
- **Connection Tests**: Basic API connectivity and compatibility
- **Integration Tests**: Real GEPA optimization with measurable results  
- **Performance Tests**: Cost, speed, and accuracy measurements
- **Documentation**: Clear setup and usage instructions

### Containerized Testing
- All tests should be containerized with Docker
- No local dependencies beyond Docker
- Consistent environments across machines
- Easy CI/CD integration

### Real Results
- Tests should demonstrate tangible GEPA improvements
- Provide concrete metrics (accuracy, cost, time)
- Export detailed results for analysis
- Show real-world applicable examples

## 📊 Testing Standards

### Result Format
All optimization tests should export results in this JSON format:
```json
{
  "optimization_summary": {
    "total_iterations": 6,
    "best_score": 0.667,
    "total_time": 88.7,
    "total_cost": 0.315,
    "improvement": 0.533
  },
  "detailed_results": [...]
}
```

### Documentation Requirements
Each test suite must include:
- Setup requirements and prerequisites
- Usage examples with expected outputs
- Troubleshooting guide
- Performance expectations
- Integration instructions

### Script Conventions
- Use consistent CLI interfaces (`test`, `optimize`, `help` modes)
- Support environment variable configuration
- Provide verbose debugging options
- Include error handling and user-friendly messages

## 🤝 Contributing

When adding new test suites:

1. Follow the directory structure conventions
2. Include comprehensive documentation  
3. Provide both basic and advanced testing modes
4. Ensure Docker containerization
5. Add appropriate launcher scripts
6. Update this README

## 📈 Current Status

### Completed ✅
- **LMStudio Integration**: Full testing suite with tangible results

### Planned 📋
- **OpenAI Integration**: API testing and optimization examples
- **Anthropic Integration**: Claude API testing
- **Ollama Integration**: Local model testing
- **Performance Benchmarks**: Cross-provider comparisons
- **CI/CD Integration**: Automated testing workflows

---

**Happy Testing! 🧪**
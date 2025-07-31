# MetaOrchestrator End-to-End Testing

## 🐳 Docker-First Testing (Zero Local Dependencies)

This directory contains comprehensive end-to-end testing infrastructure for the MetaOrchestrator framework with **complete containerization**. Following our [Docker Ruleset](../../DOCKER_RULESET.md), **no local Python dependencies are required**.

## 🧠 What's Being Tested

### MetaOrchestrator Four Pillars:
1. **RL-based Algorithm Selection** - Intelligent algorithm choice based on optimization state
2. **Predictive Topology Evolution** - Dynamic system architecture improvements  
3. **Multi-Fidelity Bayesian HyperOptimization** - Efficient parameter tuning with transfer learning
4. **Structural Prompt Evolution** - Advanced prompt structure optimization

### Real-World Scenario:
- **Multi-domain Content Generation System** optimization
- Domains: Marketing, Technical, Business, Lifestyle, Education, Communication
- Performance measurement across quality, relevance, and efficiency dimensions

## 🚀 Quick Start (Docker-Only)

### Option 1: With Your Running LMStudio (Recommended)

```bash
# From project root - connects to your LMStudio instance
./test-meta-orchestrator-real.sh
```

### Option 2: Manual Docker Compose

```bash
# Run validation + test with your LMStudio
docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-validate
docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-orchestrator-test

# Or start development environment
docker-compose -f docker-compose.lmstudio-real.yml up gepa-meta-dev
```

### Option 3: Use Real LMStudio (Recommended)

```bash
# Requires running LMStudio instance
./test-meta-orchestrator-real.sh
```

## 📋 Prerequisites

### For Real LMStudio Testing:
- ✅ **Docker** and **Docker Compose** installed
- ✅ **LMStudio running** with API server enabled (usually port 1234)
- ✅ **Model loaded** in LMStudio
- ✅ **Network connectivity** between Docker and LMStudio

### For Mock Testing:
- ✅ **Docker** and **Docker Compose** installed only
- ❌ **No LMStudio required** - mock service included

**🎯 Zero Local Dependencies:** No Python, pip, virtual environments, or package management needed on your host system!

## 📊 Expected Results

### Performance Targets:
- **25%+ improvement** over baseline GEPA optimization
- **Efficient resource utilization** (80%+ efficiency)
- **Demonstrated innovation** (topology evolution, prompt optimization)

### Success Criteria:
- ✅ **Performance Target**: Score improvement ≥ 25%
- ✅ **Efficiency Target**: Time efficiency ≥ 0.8x baseline
- ✅ **Innovation Target**: System/prompt evolution demonstrated

### Output Files:
- **Console output**: Real-time progress and results
- **JSON results**: `./results/meta_orchestrator_lmstudio_test_YYYYMMDD_HHMMSS.json`
- **Log files**: `./results/meta_orchestrator_lmstudio_test_YYYYMMDD_HHMMSS.log`

## 🔧 Configuration Options

### Docker Environment Variables:
```bash
# Set LMStudio URL (default: http://host.docker.internal:1234)
export LMSTUDIO_URL="http://localhost:1234"

# Run test with custom URL
LMSTUDIO_URL="http://your-lmstudio:1234" ./test-meta-orchestrator-real.sh
```

### Network Modes:
- **Host Network**: Direct access to host LMStudio (used by default)
- **Bridge Network**: Isolated container networking with external service access
- **Custom Networks**: Advanced Docker networking for complex setups

## 🎯 What Makes This Test Revolutionary

### Complete Containerization:
- **Zero Host Dependencies**: Everything runs in Docker containers
- **Reproducible Environment**: Same results across all systems
- **Easy Onboarding**: New developers productive immediately
- **Production Parity**: Test environment matches deployment

### Advanced Capabilities Demonstrated:
- **Adaptive Algorithm Selection**: RL agent learns optimal algorithm sequences
- **Intelligent Topology Evolution**: Predictive system architecture improvements
- **Transfer Learning**: Knowledge reuse across optimization runs  
- **Structural Optimization**: Prompt structure evolution beyond content changes

### Production Readiness:
- **Resource Management**: Intelligent compute allocation and approximation fallback
- **Error Handling**: Robust connectivity and execution error recovery
- **Monitoring**: Comprehensive metrics and performance tracking
- **Reproducibility**: Deterministic results with detailed logging

## 📈 Interpreting Results

### Performance Metrics:
- **Final Score**: Overall system performance (0.0-1.0)
- **Improvement**: Percentage gain over baseline
- **Efficiency**: Resource utilization effectiveness
- **Generations**: Optimization iterations completed

### Component Analysis:
- **Algorithm Selections**: RL agent decision history
- **Topology Changes**: System architecture evolutions  
- **HyperOpt Gains**: Parameter optimization improvements
- **Prompt Evolutions**: Structural prompt optimizations

### Success Indicators:
- **Score > 0.75**: Excellent content generation quality
- **Improvement > 25%**: Significant MetaOrchestrator advantage
- **Efficiency > 80%**: Optimal resource utilization
- **Evolutions > 0**: Innovation and adaptation demonstrated

## 🛠️ Troubleshooting

### Docker Issues:

**Container Build Failures:**
```bash
# Clean rebuild all containers
docker-compose -f docker-compose.lmstudio-real.yml build --no-cache

# Check Docker resources
docker system df
docker system prune  # Clean up if needed
```

**LMStudio Connection Issues:**
```bash
# Test connectivity from container
docker run --rm --network host curlimages/curl:latest \
  curl -v http://localhost:1234/v1/models

# Check Docker networking
docker network ls
docker network inspect bridge
```

**Permission Issues:**
```bash
# Fix results directory permissions
sudo chmod 777 ./results
mkdir -p ./results  # Create if missing
```

### Container Debugging:

**Interactive Shell Access:**
```bash
# Access running development container
docker-compose -f docker-compose.lmstudio-real.yml exec gepa-meta-dev bash

# Or start new debugging container
docker run -it --rm \
  --network host \
  -v $(pwd):/app \
  -e PYTHONPATH=/app/src \
  gepa-meta-orchestrator-lmstudio bash
```

**Log Analysis:**
```bash
# View all container logs
docker-compose -f docker-compose.lmstudio-real.yml logs

# Follow specific container logs
docker-compose -f docker-compose.lmstudio-real.yml logs -f gepa-meta-orchestrator-test
```

### Performance Optimization:

**Resource Constraints:**
- Allocate more Docker memory/CPU resources
- Use lighter MetaOrchestrator configuration profiles
- Enable approximation fallback for resource-constrained environments

**Slow Execution:**
- Check Docker resource allocation
- Optimize LMStudio model size and settings
- Use faster Docker storage driver if available

## 🎉 Advanced Usage

### Development Workflow:

```bash
# Start development environment
docker-compose -f docker-compose.lmstudio-real.yml up -d gepa-meta-dev

# Access Jupyter for interactive development  
open http://localhost:8888

# Run custom tests
docker-compose -f docker-compose.lmstudio-real.yml exec gepa-meta-dev \
  python your_custom_test.py
```

### Custom Test Scenarios:

1. **Modify Dataset**: Edit `examples/meta_orchestrator_lmstudio_test.py`
2. **Change Configuration**: Use different MetaOrchestrator profiles
3. **Add Metrics**: Extend evaluation metrics and analysis
4. **Custom Systems**: Test with your own AI system architectures

### Production Deployment:

```bash
# Build production containers
docker build -f docker/meta-orchestrator/Dockerfile -t your-registry/gepa-meta:latest .

# Deploy with production configuration
docker run --rm \
  -e LMSTUDIO_URL="http://production-lmstudio:1234" \
  -v /production/results:/app/results \
  your-registry/gepa-meta:latest
```

## 📚 Related Documentation

- [Docker Ruleset](../../DOCKER_RULESET.md) - Zero dependency development principles
- [MetaOrchestrator Architecture](../../docs/research/meta_orchestrator_hybrid_analysis.md)
- [Configuration Guide](../../examples/meta_orchestrator_config_examples.py)  
- [GEPA Core Documentation](../../README.md)

---

*This containerized testing framework demonstrates the future of AI system optimization - completely reproducible, dependency-free, and production-ready from day one.*
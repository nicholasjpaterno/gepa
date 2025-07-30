# Provider Integration Tests

This directory contains integration tests for different LLM providers to ensure GEPA compatibility.

## Test Structure

- `test_lmstudio_connection.py` - Tests basic LMStudio connectivity and API compatibility
- `test_openai_integration.py` - Tests OpenAI API integration (coming soon)
- `test_anthropic_integration.py` - Tests Anthropic API integration (coming soon)
- `test_ollama_integration.py` - Tests Ollama integration (coming soon)

## Running Tests

### LMStudio Tests

#### Prerequisites
1. LMStudio running with a model loaded
2. API server enabled in LMStudio
3. Network connectivity to LMStudio instance

#### Basic Connection Test
```bash
# Run connection test only
python tests/integration/providers/test_lmstudio_connection.py

# Or with pytest
pytest tests/integration/providers/test_lmstudio_connection.py -v

# With Docker
./test_lmstudio.sh test
```

#### Full Optimization Example
```bash
# Run real GEPA optimization with tangible results
./test_lmstudio.sh optimize

# Results will be saved to ./results/lmstudio_optimization_results.json
```

### Custom LMStudio URL
```bash
# For different LMStudio instances
LMSTUDIO_URL=http://your-ip:port ./test_lmstudio.sh test
LMSTUDIO_URL=http://localhost:1234 ./test_lmstudio.sh optimize
```

## Test Categories

### Connection Tests (`test` mode)
- ✅ API endpoint availability
- ✅ Model enumeration
- ✅ Basic chat completion
- ✅ Multiple concurrent requests
- ✅ Workflow simulation

### Optimization Examples (`optimize` mode)
- 🎯 Real GEPA optimization with measurable results
- 📊 Sentiment classification task with 15 training examples
- 🔄 6-stage prompt evolution (simulating GEPA mutations)
- 📈 Performance tracking and improvement metrics
- 💾 Detailed results saved to JSON

## Expected Results

### Connection Test Output
```
🧪 GEPA LMStudio Integration Test
==================================================
🔍 Testing connection to LMStudio at http://localhost:1234
✅ Connected successfully!
📋 Available models: model1, model2, model3...
🧪 Testing chat completion with model: model1
✅ Chat completion successful!
🔄 Testing multiple requests (simulating GEPA workload)
📊 Results: 5/5 requests successful
🚀 Simulating GEPA workflow with LMStudio
🎉 All tests passed! GEPA is compatible with your LMStudio setup.
```

### Optimization Example Output
```
🧪 GEPA LMStudio Optimization Example
==================================================
🔍 Detecting LMStudio setup at http://localhost:1234
✅ Found 18 models
🎯 Selected model: phi-4-mini-reasoning

📊 Dataset Information:
   • Training examples: 15
   • Test examples: 5
   • Model: phi-4-mini-reasoning
   • LMStudio URL: http://localhost:1234

🚀 GEPA Optimization Starting
============================================================
📊 Iteration  1 | Score: 0.600 | Time: 2.1s | Cost: $0.0010
   Prompt: Classify the sentiment of this text as positive, negative, or neutral: {text}

📊 Iteration  2 | Score: 0.733 | Time: 4.3s | Cost: $0.0020
   Prompt: Analyze the following text and classify its sentiment. Respond with exactly one...

📊 Iteration  3 | Score: 0.867 | Time: 6.8s | Cost: $0.0030
   Prompt: You are a sentiment analysis expert. Classify the sentiment...

🎯 Final Evaluation on Test Set
----------------------------------------
✅ Final test score: 0.800 (80.0% accuracy)

🎉 Optimization Summary:
   • Initial score: 0.600
   • Best score: 0.867
   • Improvement: +0.267 (+44.5%)
   • Total time: 15.2 seconds
   • Total cost: $0.0210
   • Iterations: 6

💾 Results saved to results/lmstudio_optimization_results.json
🚀 Success! GEPA optimization completed with tangible improvements.
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure LMStudio is running
   - Check API server is enabled in LMStudio settings
   - Verify IP address and port

2. **No Models Found**
   - Load a model in LMStudio interface
   - Wait for model to fully load before testing

3. **Docker Network Issues**
   - Use `--network host` for local LMStudio instances
   - Check firewall settings

4. **Permission Errors**
   - Ensure results directory is writable
   - Run with appropriate permissions

### Debugging

Enable verbose logging:
```bash
export GEPA_LOG_LEVEL=DEBUG
python tests/integration/providers/test_lmstudio_connection.py
```

## File Structure

```
tests/integration/providers/
├── README.md                    # This file
├── __init__.py                  # Package marker
├── test_lmstudio_connection.py  # Basic connectivity test
└── conftest.py                  # Pytest configuration (coming soon)

examples/providers/
├── __init__.py                  # Package marker
├── lmstudio_optimization.py     # Full optimization example
└── README.md                    # Examples documentation (coming soon)
```
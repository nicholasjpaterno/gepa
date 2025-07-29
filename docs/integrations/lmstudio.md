# LMStudio Integration

GEPA works seamlessly with LMStudio's OpenAI-compatible API. This guide shows how to set up and use GEPA with your local LMStudio instance.

## ✅ Compatibility Confirmed

**Test Results from http://192.168.1.3:1234:**
- ✅ Connection successful
- ✅ 18 models detected and accessible
- ✅ OpenAI-compatible API endpoints working
- ✅ Multiple concurrent requests supported
- ✅ GEPA workflow simulation successful

## 🚀 Quick Setup

### 1. Configure LMStudio

1. Start LMStudio with your preferred model loaded
2. Enable the API server (usually on port 1234)
3. Note your server's IP address and port

### 2. Configure GEPA

```python
from gepa import GEPAOptimizer, GEPAConfig

# Configure GEPA for LMStudio
config = GEPAConfig(
    inference={
        "provider": "openai",  # Use OpenAI provider for compatibility
        "model": "your-model-name",  # Use any model loaded in LMStudio
        "base_url": "http://192.168.1.3:1234",  # Your LMStudio URL
        "api_key": "dummy-key",  # LMStudio doesn't require real API key
        "max_tokens": 100,
        "temperature": 0.7
    },
    optimization={
        "budget": 50,
        "pareto_set_size": 10,
        "minibatch_size": 3
    }
)
```

### 3. Run Optimization

```python
import asyncio
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score

# Define your AI system
system = CompoundAISystem(
    modules={
        "classifier": LanguageModule(
            id="classifier",
            prompt="Classify the sentiment of this text as positive, negative, or neutral: {text}",
            model_weights="your-model-name"
        )
    },
    control_flow=SequentialFlow(["classifier"])
)

# Your dataset
dataset = [
    {"text": "I love this product!", "expected": "positive"},
    {"text": "This is terrible.", "expected": "negative"},
    {"text": "It's okay.", "expected": "neutral"},
]

async def main():
    evaluator = SimpleEvaluator([ExactMatch(), F1Score()])
    optimizer = GEPAOptimizer(config, evaluator)
    
    result = await optimizer.optimize(system, dataset)
    print(f"Best score: {result.best_score:.3f}")

asyncio.run(main())
```

## 🧪 Testing Your Setup

Use the included test script to verify compatibility:

```bash
# Build and run the test container
docker build -f Dockerfile.test -t gepa-lmstudio-test .
docker run --rm --network host gepa-lmstudio-test

# Or run the shell script
./test_lmstudio.sh

# Or customize the LMStudio URL
LMSTUDIO_URL=http://your-ip:port ./test_lmstudio.sh
```

## 📋 Available Models (Example from Test)

From the test run, LMStudio had these models available:
- ernie-4.5-21b-a3b-pt
- phi-4-mini-reasoning  
- qwen3-the-josiefied-omega-directive-22b-uncensored-abliterated
- qwen3-zero-coder-reasoning-0.8b
- llama-3_3-nemotron-super-49b-v1_5
- ...and 13 more models

GEPA can work with any model loaded in LMStudio.

## 🔧 Configuration Options

### Network Settings
```python
config = GEPAConfig(
    inference={
        "provider": "openai",
        "base_url": "http://192.168.1.3:1234",  # Local network
        # "base_url": "http://localhost:1234",   # Same machine
        # "base_url": "http://0.0.0.0:1234",     # All interfaces
    }
)
```

### Model Selection
```python
# List available models first
async with httpx.AsyncClient() as client:
    response = await client.get("http://192.168.1.3:1234/v1/models")
    models = [model["id"] for model in response.json()["data"]]
    print("Available models:", models)

# Then use any model in your config
config = GEPAConfig(
    inference={
        "model": "phi-4-mini-reasoning",  # Choose from available models
    }
)
```

### Performance Tuning
```python
config = GEPAConfig(
    inference={
        "max_tokens": 50,      # Shorter responses for faster optimization
        "temperature": 0.1,    # Lower temperature for consistency
    },
    optimization={
        "minibatch_size": 2,   # Smaller batches for local models
        "budget": 30,          # Adjust based on your time/compute constraints
    }
)
```

## 🚨 Troubleshooting

### Connection Issues
1. **"Connection refused"**: Check LMStudio is running and API server is enabled
2. **"No models found"**: Load a model in LMStudio interface
3. **Network errors**: Verify IP address and port are correct

### Performance Issues
1. **Slow responses**: Reduce `max_tokens` and `minibatch_size`
2. **Out of memory**: Use smaller models or reduce batch size
3. **Inconsistent results**: Lower `temperature` for more deterministic outputs

### API Compatibility
- LMStudio implements OpenAI's API spec
- Some advanced features may not be available
- API key is not required but can be set to any dummy value

## 🎯 Best Practices

1. **Start Small**: Use small budgets (20-50 rollouts) for initial testing
2. **Monitor Resources**: Watch GPU/CPU usage in LMStudio
3. **Choose Appropriate Models**: Balance model size with optimization speed
4. **Network Stability**: Use wired connections for reliable API access
5. **Backup Results**: Save optimization results before long runs

## 📈 Performance Expectations

Based on the test results:
- **Connection**: < 1 second
- **Single Request**: 1-3 seconds (depends on model size)
- **Batch Requests**: 5-15 seconds for 5 requests
- **Full Optimization**: 2-10 minutes (depends on budget and model)

## 🤝 Community Models

LMStudio supports many community models that work well with GEPA:
- **Qwen models**: Excellent for reasoning tasks
- **Phi models**: Fast and efficient for smaller tasks  
- **Llama models**: Good balance of performance and speed
- **Mistral models**: Great for creative and analytical tasks

Choose models based on your specific optimization task requirements.
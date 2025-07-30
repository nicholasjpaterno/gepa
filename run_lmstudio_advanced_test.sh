#!/bin/bash

# GEPA Advanced Algorithms LMStudio Test Runner
# ============================================

set -e

# Configuration
LMSTUDIO_URL=${LMSTUDIO_URL:-"http://localhost:1234"}
DOCKER_IMAGE="gepa-lmstudio-advanced-test"

echo "🧠 GEPA Advanced Algorithms LMStudio Test"
echo "========================================"
echo "🎯 Testing advanced algorithms vs basic heuristics with LMStudio"  
echo "🌐 LMStudio URL: $LMSTUDIO_URL"
echo

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "🏗️  Building Docker image with advanced algorithms..."
docker build -f testing/lmstudio/Dockerfile.test -t $DOCKER_IMAGE . --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image built successfully"
echo

# Create results directory
mkdir -p results

# Run the advanced algorithms test
echo "🚀 Running advanced algorithms test..."
echo "⏱️  This will compare basic heuristics vs advanced algorithms"
echo "📊 Expected to see sophisticated algorithm analysis in action"
echo

docker run --rm \
    --network host \
    -v "$(pwd)/results:/app/results" \
    -e LMSTUDIO_URL="$LMSTUDIO_URL" \
    -e PYTHONPATH="/app/src" \
    $DOCKER_IMAGE \
    python examples/providers/lmstudio_advanced_test.py

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Advanced algorithms test completed successfully!"
    echo
    echo "📊 Results Summary:"
    if [ -f "results/lmstudio_advanced_algorithms_test.json" ]; then
        echo "✅ Detailed results saved to: results/lmstudio_advanced_algorithms_test.json"
        
        # Try to show a brief summary if jq is available
        if command -v jq &> /dev/null; then
            echo
            echo "📈 Quick Results:"
            jq -r '.comparison | "Score Improvement: \(.score_improvement | tonumber | . * 100 | floor / 100) (\(.score_improvement_percentage | tonumber | floor)%)"' results/lmstudio_advanced_algorithms_test.json 2>/dev/null || true
            jq -r '.comparison | "Efficiency Change: \(.efficiency_improvement_percentage | tonumber | floor)%"' results/lmstudio_advanced_algorithms_test.json 2>/dev/null || true
        fi
    fi
    
    echo
    echo "🔍 Key Features Tested:"
    echo "   ✅ Instance Score Prediction (Algorithm 2)"
    echo "   ✅ Adaptive Score Comparison (Algorithm 2)" 
    echo "   ✅ Intelligent Module Selection (Algorithm 3)"
    echo "   ✅ Strategic Minibatch Sampling (Algorithm 3)"
    echo "   ✅ Deep Compatibility Analysis (Algorithm 4)"
    echo "   ✅ Statistical Complementarity Testing (Algorithm 4)"
    echo "   ✅ Multi-Criteria Desirability Scoring (Algorithm 4)"
    
else
    echo
    echo "❌ Advanced algorithms test failed"
    echo "🔧 Troubleshooting:"
    echo "   • Ensure LMStudio is running at $LMSTUDIO_URL"
    echo "   • Check that a model is loaded in LMStudio"
    echo "   • Verify API server is enabled in LMStudio settings"
    echo "   • Check network connectivity to LMStudio"
    exit 1
fi 
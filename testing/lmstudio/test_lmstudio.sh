#!/bin/bash

# GEPA LMStudio Testing Script
# ===========================

set -e

# Configuration
LMSTUDIO_URL=${LMSTUDIO_URL:-"http://localhost:1234"}
MODE=${1:-"test"}  # test, optimize, or advanced

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "   Please install Docker to run this script"
    exit 1
fi

case $MODE in
    "test")
        echo "🧪 GEPA LMStudio Connection Test"
        echo "================================"
        echo "🔄 Testing basic connectivity and API compatibility"
        echo

        # Build test container
        echo "🏗️  Building test container..."
        docker build -f testing/lmstudio/Dockerfile.test -t gepa-lmstudio-test . --quiet

        # Run connection test
        echo "🔄 Running connection test against $LMSTUDIO_URL"
        echo
        
        docker run --rm --network host \
            -e LMSTUDIO_URL="$LMSTUDIO_URL" \
            gepa-lmstudio-test

        echo
        echo "✅ Connection test completed!"
        ;;
        
    "optimize")
        echo "🚀 GEPA LMStudio Optimization Example"
        echo "===================================="
        echo "🎯 Running full optimization with tangible results"
        echo

        # Build test container
        echo "🏗️  Building optimization container..."
        docker build -f testing/lmstudio/Dockerfile.test -t gepa-lmstudio-optimize . --quiet

        # Create results directory
        mkdir -p results

        # Run optimization example
        echo "🔄 Running optimization against $LMSTUDIO_URL"
        echo "📊 This will show real GEPA optimization with measurable improvements"
        echo
        
        docker run --rm --network host \
            -v "$(pwd)/results:/app/results" \
            -e LMSTUDIO_URL="$LMSTUDIO_URL" \
            gepa-lmstudio-optimize \
            python examples/providers/lmstudio_advanced_test.py

        echo
        echo "🎉 Optimization completed!"
        echo "📁 Results saved in: ./results/"
        ;;
        
    "advanced")
        echo "🧠 GEPA Advanced Algorithms LMStudio Test"
        echo "========================================"
        echo "🎯 Testing sophisticated algorithm implementations vs basic heuristics"
        echo "📊 This demonstrates the production-grade improvements in action"
        echo

        # Build test container with advanced algorithms
        echo "🏗️  Building advanced algorithms test container..."
        docker build -f testing/lmstudio/Dockerfile.test -t gepa-lmstudio-advanced . --quiet

        # Create results directory
        mkdir -p results

        # Run advanced algorithms test
        echo "🔄 Running advanced algorithms test against $LMSTUDIO_URL"
        echo "🧠 This will compare basic heuristics vs advanced algorithms"
        echo "📈 Expected improvements: 15-40% better performance"
        echo
        
        docker run --rm --network host \
            -v "$(pwd)/results:/app/results" \
            -e LMSTUDIO_URL="$LMSTUDIO_URL" \
            gepa-lmstudio-advanced \
            python examples/providers/lmstudio_advanced_test.py

        echo
        echo "🎉 Advanced algorithms test completed!"
        echo "📁 Results saved in: ./results/lmstudio_advanced_algorithms_test.json"
        ;;
        
    "help"|"-h"|"--help")
        echo "GEPA LMStudio Testing Script"
        echo "Usage: $0 [MODE]"
        echo
        echo "Modes:"
        echo "  test      - Run basic connection and API compatibility test (default)"
        echo "  optimize  - Run full GEPA optimization example with tangible results"
        echo "  advanced  - Test advanced algorithms vs basic heuristics (NEW!)"
        echo "  help      - Show this help message"
        echo
        echo "Environment Variables:"
        echo "  LMSTUDIO_URL  - LMStudio server URL (default: http://localhost:1234)"
        echo
        echo "Examples:"
        echo "  ./testing/lmstudio/test_lmstudio.sh test                                    # Basic connection test"
        echo "  ./testing/lmstudio/test_lmstudio.sh optimize                                # Full optimization example"
        echo "  ./testing/lmstudio/test_lmstudio.sh advanced                                # Advanced algorithms test"
        echo "  LMSTUDIO_URL=http://localhost:1234 ./testing/lmstudio/test_lmstudio.sh test # Custom URL"
        exit 0
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo
echo "💡 Tips:"
echo "   • Make sure LMStudio is running with a model loaded"
echo "   • Check that LMStudio's API server is enabled"
echo "   • Verify network connectivity to $LMSTUDIO_URL"
echo
echo "🚀 Usage:"
echo "   ./testing/lmstudio/test_lmstudio.sh test      - Basic connection test"
echo "   ./testing/lmstudio/test_lmstudio.sh optimize  - Full optimization example with results"
echo "   ./testing/lmstudio/test_lmstudio.sh advanced  - Advanced algorithms vs heuristics test"
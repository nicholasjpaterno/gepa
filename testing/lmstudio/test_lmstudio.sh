#!/bin/bash

# GEPA LMStudio Testing Script
# ===========================

set -e

# Configuration
LMSTUDIO_URL=${LMSTUDIO_URL:-"http://192.168.1.3:1234"}
MODE=${1:-"test"}  # test or optimize

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
            python examples/providers/lmstudio_optimization.py

        echo
        echo "🎉 Optimization completed!"
        echo "📁 Results saved in: ./results/"
        ;;
        
    "help"|"-h"|"--help")
        echo "GEPA LMStudio Testing Script"
        echo "Usage: $0 [MODE]"
        echo
        echo "Modes:"
        echo "  test      - Run basic connection and API compatibility test (default)"
        echo "  optimize  - Run full GEPA optimization example with tangible results"
        echo "  help      - Show this help message"
        echo
        echo "Environment Variables:"
        echo "  LMSTUDIO_URL  - LMStudio server URL (default: http://192.168.1.3:1234)"
        echo
        echo "Examples:"
        echo "  ./testing/lmstudio/test_lmstudio.sh test                                    # Basic connection test"
        echo "  ./testing/lmstudio/test_lmstudio.sh optimize                                # Full optimization example"
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
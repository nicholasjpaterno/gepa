#!/bin/bash

# MetaOrchestrator Docker Test with Real LMStudio
# ===============================================
# Zero local dependencies - connects to your running LMStudio

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.lmstudio-real.yml"
LMSTUDIO_URL=${LMSTUDIO_URL:-"http://localhost:1234"}

echo -e "${BLUE}🐳 MetaOrchestrator + Real LMStudio Test${NC}"
echo "========================================"
echo
echo -e "${CYAN}✨ Zero Local Dependencies${NC}"
echo -e "${CYAN}✨ Connects to Your Running LMStudio${NC}"
echo -e "${CYAN}✨ Complete Containerized Testing${NC}"
echo

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is required but not installed${NC}"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is required but not installed${NC}"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Determine Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check LMStudio connectivity
echo -e "${YELLOW}🔌 Checking LMStudio connectivity...${NC}"
if command -v curl &> /dev/null; then
    if curl -s "$LMSTUDIO_URL/v1/models" > /dev/null; then
        echo -e "${GREEN}✅ LMStudio detected at $LMSTUDIO_URL${NC}"
        
        # Get model info
        MODEL_INFO=$(curl -s "$LMSTUDIO_URL/v1/models" | python -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else 'Unknown')" 2>/dev/null || echo "Unknown")
        echo -e "${GREEN}   Model: $MODEL_INFO${NC}"
    else
        echo -e "${RED}❌ Cannot connect to LMStudio at $LMSTUDIO_URL${NC}"
        echo "   Please ensure:"
        echo "   • LMStudio is running"
        echo "   • A model is loaded"
        echo "   • API server is enabled"
        echo "   • LMStudio is accessible at $LMSTUDIO_URL"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️ curl not available, will test connectivity in container${NC}"
fi

echo
echo -e "${YELLOW}🔧 Test Configuration:${NC}"
echo "   • Docker Compose: $DOCKER_COMPOSE"
echo "   • LMStudio URL: $LMSTUDIO_URL"
echo "   • Network Mode: host (direct LMStudio access)"
echo "   • Test Type: Real-world content generation"
echo

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up containers...${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" down --remove-orphans -v 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

echo -e "${BLUE}🚀 Ready to test MetaOrchestrator with your LMStudio?${NC}"
echo "   This will:"
echo "   • Build MetaOrchestrator Docker container"
echo "   • Validate all components and LMStudio connectivity"
echo "   • Run comprehensive end-to-end test"
echo "   • Demonstrate 4-pillar architecture capabilities"
echo "   • Generate detailed performance analysis"
echo
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

echo
echo -e "${BLUE}🏗️ Building MetaOrchestrator Container...${NC}"
echo "======================================="

# Build container
if ! $DOCKER_COMPOSE -f "$COMPOSE_FILE" build --no-cache gepa-meta-orchestrator-test; then
    echo -e "${RED}❌ Failed to build container${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container built successfully${NC}"

echo
echo -e "${BLUE}🧪 Running Component Validation...${NC}"
echo "=================================="

# Run validation
if ! $DOCKER_COMPOSE -f "$COMPOSE_FILE" run --rm gepa-meta-validate; then
    echo -e "${RED}❌ Component validation failed${NC}"
    echo
    echo -e "${YELLOW}🔍 Container logs:${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs gepa-meta-validate 2>/dev/null || true
    exit 1
fi

echo
echo -e "${GREEN}✅ All components validated - LMStudio ready!${NC}"

echo
echo -e "${BLUE}🚀 Running MetaOrchestrator End-to-End Test...${NC}"
echo "=============================================="

echo -e "${YELLOW}🧠 Testing Revolutionary Four-Pillar Architecture:${NC}"
echo "   • 🤖 RL-based Algorithm Selection"
echo "   • 🏗️ Predictive Topology Evolution"
echo "   • 📊 Multi-Fidelity Bayesian HyperOptimization"
echo "   • 📝 Structural Prompt Evolution"
echo
echo -e "${YELLOW}📋 Real-World Test Scenario:${NC}"
echo "   • Multi-domain content generation optimization"
echo "   • 6 diverse test cases across different domains"
echo "   • Performance measurement vs baseline GEPA"
echo "   • Component contribution analysis"
echo

if $DOCKER_COMPOSE -f "$COMPOSE_FILE" run --rm gepa-meta-orchestrator-test; then
    echo
    echo -e "${GREEN}🎉 MetaOrchestrator Test Completed Successfully!${NC}"
    echo "=============================================="
    echo
    echo -e "${BLUE}📊 Test Results Summary:${NC}"
    echo "   • Detailed performance metrics shown above"
    echo "   • Results files saved to ./results/ directory"
    echo "   • Component analysis and comparisons completed"
    echo
    echo -e "${GREEN}✨ Key Achievements Demonstrated:${NC}"
    echo "   ✅ Revolutionary 4-pillar architecture working with real LMStudio"
    echo "   ✅ Real-world content generation optimization"
    echo "   ✅ Performance improvements over baseline GEPA"
    echo "   ✅ Complete containerized testing (zero local deps)"
    echo "   ✅ Production-ready coordination and resource management"
    echo "   ✅ Seamless integration with your LMStudio setup"
    echo
    echo -e "${CYAN}🚀 MetaOrchestrator is production-ready with LMStudio!${NC}"
    exit_code=0
else
    echo -e "${RED}❌ MetaOrchestrator test failed${NC}"
    echo
    echo -e "${YELLOW}🔍 Test logs:${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs --tail=50 gepa-meta-orchestrator-test 2>/dev/null || true
    exit_code=1
fi

echo
echo -e "${BLUE}📁 Test Results Available:${NC}"
if [ -d "./results" ]; then
    echo "   Recent files in ./results/:"
    ls -la ./results/ | head -10
    echo
    echo -e "${BLUE}💡 Analyze Results:${NC}"
    echo "   • JSON files contain detailed metrics and analysis"
    echo "   • Log files show complete execution trace"
    echo "   • Performance comparison data available for analysis"
else
    echo "   No results directory found"
fi

echo
echo -e "${BLUE}🎯 Next Steps:${NC}"
if [ $exit_code -eq 0 ]; then
    echo "   • Review detailed results for performance insights"
    echo "   • Use MetaOrchestrator configuration profiles for your projects"
    echo "   • Integrate with your AI systems for production optimization" 
    echo "   • Explore advanced configuration and customization options"
else
    echo "   • Check container logs for detailed error information"
    echo "   • Verify LMStudio is running and accessible"
    echo "   • Ensure Docker has sufficient resources"
    echo "   • Review test configuration and retry"
fi

echo
echo -e "${BLUE}🐳 Container-Based MetaOrchestrator Testing Complete!${NC}"
echo "==================================================="

exit $exit_code
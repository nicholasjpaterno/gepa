# MetaOrchestrator End-to-End Test Container
# ==========================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install additional test dependencies
RUN pip install --no-cache-dir httpx pytest pytest-asyncio

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH="/app/src:/app"
ENV LMSTUDIO_URL="http://host.docker.internal:1234"

# Default command runs the MetaOrchestrator test
CMD ["python", "examples/meta_orchestrator_lmstudio_test.py"]
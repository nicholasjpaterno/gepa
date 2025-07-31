# MetaOrchestrator Test Container - Zero Local Dependencies
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Install GEPA and all dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    httpx \
    pytest \
    pytest-asyncio \
    jupyter \
    pandas \
    matplotlib \
    seaborn

# Create results directory with proper permissions
RUN mkdir -p /app/results && chmod 777 /app/results

# Copy test files
COPY examples/ ./examples/
COPY testing/ ./testing/

# Set environment variables
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from gepa.meta_orchestrator import MetaOrchestrator; print('✅ MetaOrchestrator healthy')"

# Default command runs the comprehensive test
CMD ["python", "examples/meta_orchestrator_lmstudio_test.py"]
# Development Environment - Complete GEPA development setup
FROM python:3.11-slim

# Install system dependencies including development tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    vim \
    nano \
    htop \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY pyproject.toml ./

# Install GEPA and all development dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    httpx \
    pytest \
    pytest-asyncio \
    pytest-cov \
    jupyter \
    ipython \
    black \
    flake8 \
    mypy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    streamlit

# Create directories
RUN mkdir -p /app/results /app/logs /app/notebooks && \
    chmod 777 /app/results /app/logs /app/notebooks

# Set environment variables
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Expose ports for Jupyter and other development tools
EXPOSE 8888 8501 8000

# Default command keeps container running for development
CMD ["tail", "-f", "/dev/null"]
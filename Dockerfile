FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r gepa && useradd --no-log-init -r -g gepa gepa

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency files and source code
COPY --chown=gepa:gepa pyproject.toml README.md ./
COPY --chown=gepa:gepa src/ ./src/

# Install Python dependencies
RUN pip install -e .
COPY --chown=gepa:gepa alembic.ini ./
COPY --chown=gepa:gepa migrations/ ./migrations/

# Switch to non-root user
USER gepa

# Expose port for metrics server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from gepa.database.connection import DatabaseManager; from gepa.config import get_config; asyncio.run(DatabaseManager(get_config().database).health_check())" || exit 1

# Default command
CMD ["python", "-m", "gepa.cli"]
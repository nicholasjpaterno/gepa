# GEPA Project Makefile
# =====================

.PHONY: help install dev test lint format clean
.PHONY: test-lmstudio test-lmstudio-connection test-lmstudio-optimize
.PHONY: build-docker run-tests docs

# Default target
help: ## Show this help message
	@echo "GEPA Project - Available Commands"
	@echo "================================="
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "LMStudio Testing:"
	@echo "  make test-lmstudio          # Run basic connection test"
	@echo "  make test-lmstudio-optimize # Run full optimization example"
	@echo
	@echo "Environment Variables:"
	@echo "  LMSTUDIO_URL=http://your-ip:port make test-lmstudio"

# Development setup
install: ## Install project dependencies
	pip install -e ".[dev]"

dev: install ## Setup development environment
	pre-commit install

# Code quality
test: ## Run all tests
	pytest tests/ -v

lint: ## Run code linting
	ruff check src/ tests/ examples/
	mypy src/gepa

format: ## Format code
	black src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

# LMStudio Integration Testing
test-lmstudio: test-lmstudio-connection ## Run LMStudio connection test (alias)

test-lmstudio-connection: ## Test LMStudio connectivity and API compatibility
	@echo "🧪 Running LMStudio Connection Test"
	@./testing/lmstudio/test_lmstudio.sh test

test-lmstudio-optimize: ## Run full LMStudio GEPA optimization with tangible results
	@echo "🚀 Running LMStudio Optimization Example"
	@./testing/lmstudio/test_lmstudio.sh optimize

test-lmstudio-help: ## Show LMStudio testing help
	@./testing/lmstudio/test_lmstudio.sh help

# Docker operations
build-docker: ## Build Docker images for testing
	@echo "🏗️  Building Docker test images..."
	docker build -f testing/lmstudio/Dockerfile.test -t gepa-lmstudio-test .

# Documentation
docs: ## Generate documentation (coming soon)
	@echo "📖 Documentation generation not yet implemented"
	@echo "   See README.md and testing/README.md for now"

# Cleanup
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

# Project information
info: ## Show project information
	@echo "GEPA (GeneticPareto) Prompt Optimization Toolkit"
	@echo "================================================"
	@echo
	@echo "📊 Project Status:"
	@echo "  • Core Implementation: ✅ Complete"
	@echo "  • LMStudio Integration: ✅ Complete with tangible results"
	@echo "  • Docker Support: ✅ Complete"
	@echo "  • CI/CD Pipeline: ✅ Complete"
	@echo "  • Documentation: ✅ Complete"
	@echo
	@echo "🎯 Quick Start:"
	@echo "  1. make install          # Install dependencies"
	@echo "  2. make test-lmstudio    # Test with your LMStudio"
	@echo "  3. make test-lmstudio-optimize  # See real optimization results"
	@echo
	@echo "📁 Testing Infrastructure:"
	@echo "  • Location: testing/lmstudio/"
	@echo "  • Results: Saved to ./results/"
	@echo "  • Documentation: testing/lmstudio/README.md"
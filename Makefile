# GEPA Project Makefile
# =====================

.PHONY: help install dev test lint format clean
.PHONY: test-lmstudio test-lmstudio-connection test-lmstudio-optimize test-lmstudio-advanced
.PHONY: build-docker run-tests docs
.PHONY: docker-test docker-lint docker-format docker-dev
.PHONY: meta-test meta-validate meta-quick meta-extended meta-configurable

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
	@echo "  make test-lmstudio-advanced # Test advanced algorithms vs heuristics"
	@echo
	@echo "Environment Variables:"
	@echo "  LMSTUDIO_URL=http://your-ip:port make test-lmstudio"
	@echo "  GEPA_TEST_ROUNDS=N make meta-configurable"
	@echo
	@echo "Docker-First Development (RECOMMENDED):"
	@echo "  make docker-dev             # Start development environment"
	@echo "  make docker-test            # Run tests in container"
	@echo "  make docker-lint            # Run linting in container"
	@echo
	@echo "MetaOrchestrator Testing (RECOMMENDED):"
	@echo "  make meta-validate          # Validate setup"
	@echo "  make meta-test              # Run MetaOrchestrator (1 round)"
	@echo "  make meta-quick             # Quick test (10 rounds)"
	@echo "  make meta-extended          # Extended test (100 rounds)"
	@echo "  GEPA_TEST_ROUNDS=50 make meta-configurable  # Custom rounds"
# Development setup (prefer Docker alternatives below)
install: ## Install project dependencies (local - not recommended)
	pip install -e ".[dev]"

dev: install ## Setup development environment (local - not recommended)
	pre-commit install

# Docker-first development (PREFERRED)
docker-dev: ## Start Docker development environment
	docker-compose up -d gepa-meta-dev
	@echo "🐳 Development environment running. Access with:"
	@echo "   docker-compose exec gepa-meta-dev bash"

# Code quality (local - prefer Docker versions below)
test: ## Run all tests (local)
	pytest tests/ -v

lint: ## Run code linting (local)
	ruff check src/ tests/ examples/
	mypy src/gepa

format: ## Format code (local)
	black src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

# Docker-first code quality (PREFERRED)
docker-test: ## Run all tests in Docker container
	docker-compose up -d
	docker-compose exec gepa-meta-dev pytest tests/ -v

docker-lint: ## Run code linting in Docker container
	docker-compose exec gepa-meta-dev ruff check src/ tests/ examples/
	docker-compose exec gepa-meta-dev mypy src/gepa

docker-format: ## Format code in Docker container
	docker-compose exec gepa-meta-dev black src/ tests/ examples/
	docker-compose exec gepa-meta-dev ruff check --fix src/ tests/ examples/

# LMStudio Integration Testing
test-lmstudio: test-lmstudio-connection ## Run LMStudio connection test (alias)

test-lmstudio-connection: ## Test LMStudio connectivity and API compatibility
	@echo "🧪 Running LMStudio Connection Test"
	@./testing/lmstudio/test_lmstudio.sh test

test-lmstudio-optimize: ## Run full LMStudio GEPA optimization with tangible results
	@echo "🚀 Running LMStudio Optimization Example"
	@./testing/lmstudio/test_lmstudio.sh optimize

test-lmstudio-advanced: ## Test advanced algorithms vs basic heuristics with LMStudio
	@echo "🧠 Testing Advanced Algorithms vs Basic Heuristics"
	@./testing/lmstudio/test_lmstudio.sh advanced

test-lmstudio-help: ## Show LMStudio testing help
	@./testing/lmstudio/test_lmstudio.sh help

# Docker operations
build-docker: ## Build Docker images for testing
	@echo "🏗️  Building Docker test images..."
	docker build -f testing/lmstudio/Dockerfile.test -t gepa-lmstudio-test .
	docker build -f docker/meta-orchestrator/Dockerfile -t gepa-meta-orchestrator .

# MetaOrchestrator Docker Testing (RECOMMENDED)
meta-validate: ## Validate MetaOrchestrator components and LMStudio connectivity
	@echo "🧪 Validating MetaOrchestrator setup..."
	docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-validate

meta-test: ## Run MetaOrchestrator test (1 round)
	@echo "🧠 Running MetaOrchestrator optimization (1 round)..."
	docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-orchestrator-test

meta-quick: ## Run quick MetaOrchestrator test (10 rounds)
	@echo "⚡ Running quick MetaOrchestrator test (10 rounds)..."
	docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-quick-test

meta-extended: ## Run extended MetaOrchestrator test (100 rounds)
	@echo "🚀 Running extended MetaOrchestrator test (100 rounds)..."
	docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-extended-test

meta-configurable: ## Run configurable MetaOrchestrator test (set GEPA_TEST_ROUNDS env var)
	@echo "⚙️  Running configurable MetaOrchestrator test..."
	@echo "   Rounds: $(GEPA_TEST_ROUNDS) (default: 25 if not set)"
	docker-compose -f docker-compose.lmstudio-real.yml run --rm gepa-meta-configurable-test

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
	@echo "  • MetaOrchestrator: ✅ Complete with Docker integration"
	@echo "  • Docker Support: ✅ Complete"
	@echo "  • CI/CD Pipeline: ✅ Complete"
	@echo "  • Documentation: ✅ Complete"
	@echo
	@echo "🎯 Quick Start:"
	@echo "  1. make docker-dev       # Start Docker development environment"
	@echo "  2. make meta-validate    # Validate MetaOrchestrator setup"
	@echo "  3. make meta-test        # Run MetaOrchestrator optimization"
	@echo "  4. make meta-extended    # Run comprehensive MetaOrchestrator test"
	@echo "  5. make test-lmstudio    # Legacy LMStudio testing"
	@echo
	@echo "📁 Testing Infrastructure:"
	@echo "  • Location: testing/lmstudio/"
	@echo "  • Results: Saved to ./results/"
	@echo "  • Documentation: testing/lmstudio/README.md"
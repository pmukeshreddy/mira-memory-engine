.PHONY: help install dev test lint format run docker-build docker-up docker-down clean

# Default target
help:
	@echo "Mira Memory Engine - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run         Run the development server"
	@echo "  make test        Run tests with coverage"
	@echo "  make lint        Run linter (ruff)"
	@echo "  make format      Format code (ruff)"
	@echo "  make typecheck   Run type checker (mypy)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       Remove build artifacts"
	@echo "  make benchmark   Run latency benchmarks"
	@echo "  make seed        Load test data"

# ============================================================================
# Setup
# ============================================================================

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# ============================================================================
# Development
# ============================================================================

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test:
	pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

lint:
	ruff check app/ tests/

format:
	ruff check app/ tests/ --fix
	ruff format app/ tests/

typecheck:
	mypy app/

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t mira-memory-engine:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ============================================================================
# Frontend
# ============================================================================

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ============================================================================
# Utilities
# ============================================================================

benchmark:
	python scripts/benchmark.py

seed:
	python scripts/seed_data.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf dist build *.egg-info
	rm -rf htmlcov .coverage
	rm -rf data/chroma/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ============================================================================
# Database
# ============================================================================

db-reset:
	rm -rf data/chroma/*
	@echo "ChromaDB data cleared"

# ============================================================================
# Observability
# ============================================================================

metrics:
	@echo "Opening Prometheus metrics at http://localhost:8000/metrics"
	open http://localhost:8000/metrics

health:
	curl -s http://localhost:8000/api/v1/health | python -m json.tool

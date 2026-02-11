# =============================================================================
# Makefile - Shopping AI Assistant
# =============================================================================

.PHONY: help setup dev servers backend frontend test clean \
        docker-build docker-up docker-down docker-logs docker-dev docker-clean

# Default
help:
	@echo "🛒 Shopping AI Assistant"
	@echo ""
	@echo "── Local Development ──────────────────────────────"
	@echo "  make setup        First-time setup (.env)"
	@echo "  make dev          Run agent CLI (debug mode)"
	@echo "  make servers      Start all MCP servers"
	@echo "  make backend      Start FastAPI backend"
	@echo "  make frontend     Start React dev server"
	@echo "  make test         Run tests"
	@echo "  make clean        Clean caches"
	@echo ""
	@echo "── Docker (Production) ────────────────────────────"
	@echo "  make docker-build   Build all images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    Follow all logs"
	@echo "  make docker-clean   Remove containers + volumes"
	@echo ""
	@echo "── Docker (Debug) ─────────────────────────────────"
	@echo "  make docker-dev     Start in debug mode"
	@echo "                      (hot-reload, no cache)"
	@echo ""

# =============================================================================
# Local Development
# =============================================================================

setup:
	@test -f .env || (cp .env.example .env && echo "✅ Created .env from .env.example")
	@echo "⚠️  Edit .env and add your API keys!"

dev:
	DEBUG_MODE=true python main.py

servers:
	python -m src.mcp_servers.run_servers

backend:
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

frontend:
	cd frontend && npm run dev -- --host 0.0.0.0

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache htmlcov .coverage 2>/dev/null || true

# =============================================================================
# Docker - Production
# =============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "   Frontend:  http://localhost:3000"
	@echo "   Backend:   http://localhost:8080"
	@echo "   API Docs:  http://localhost:8080/docs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --rmi local
	@echo "✅ Cleaned up containers, volumes, and images"

# =============================================================================
# Docker - Debug Mode
# =============================================================================

docker-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
	@echo ""
	@echo "🔧 Debug mode started!"
	@echo "   DEBUG_MODE=true (no caching)"
	@echo "   Hot-reload enabled"
	@echo "   Frontend:  http://localhost:3000"
	@echo "   Backend:   http://localhost:8080"

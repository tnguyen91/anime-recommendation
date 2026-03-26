.PHONY: dev test lint format migrate migrate-create

## ── Development ─────────────────────────────────────────────────────────────

dev:  ## Start all services locally
	docker compose up --build

dev-db:  ## Start only the database
	docker compose up db -d

## ── Quality ─────────────────────────────────────────────────────────────────

test:  ## Run tests with coverage
	pytest api/tests/ -v --cov=api --cov-report=term-missing

lint:  ## Check code style and common bugs
	ruff check api/
	ruff format --check api/

format:  ## Auto-fix code style
	ruff format api/
	ruff check --fix api/

## ── Database ────────────────────────────────────────────────────────────────

migrate:  ## Run pending database migrations
	alembic upgrade head

migrate-create:  ## Create a new migration (usage: make migrate-create msg="add users table")
	alembic revision --autogenerate -m "$(msg)"

## ── Help ────────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

.PHONY: help env up stop down logs shell migrate

NEXT_PUBLIC_USER_ID=$(USER)
NEXT_PUBLIC_API_URL=http://localhost:8765

help:
	@echo "Available commands:"
	@echo "  make env       - Copy .env.example files to .env"
	@echo "  make up        - Start all containers"
	@echo "  make stop      - Stop containers (preserves data)"
	@echo "  make down      - Stop containers and delete volumes (destructive)"
	@echo "  make logs      - Show container logs"
	@echo "  make shell     - Open a shell in the API container"
	@echo "  make migrate   - Run database migrations"

env:
	cd api && cp .env.example .env
	cd ui && cp .env.example .env

build:
	docker compose -f docker-compose.yml -f docker-compose.override.yml build

up:
	NEXT_PUBLIC_USER_ID=$(NEXT_PUBLIC_USER_ID) NEXT_PUBLIC_API_URL=$(NEXT_PUBLIC_API_URL) \
		docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

stop:
	docker compose -f docker-compose.yml -f docker-compose.override.yml stop

down:
	docker compose -f docker-compose.yml -f docker-compose.override.yml down -v

logs:
	docker compose -f docker-compose.yml -f docker-compose.override.yml logs -f

shell:
	docker compose -f docker-compose.yml -f docker-compose.override.yml exec openmemory-mcp bash

migrate:
	docker compose -f docker-compose.yml -f docker-compose.override.yml exec openmemory-mcp alembic upgrade head

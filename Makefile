# Simple Docker Compose commands

.PHONY: help build up down logs shell test clean

compose = docker compose

help:
	@echo "Available commands:"
	@echo "  build  - Build Docker image"
	@echo "  up     - Start the application"
	@echo "  down   - Stop the application"
	@echo "  logs   - View application logs"
	@echo "  shell  - Open shell in container"
	@echo "  test   - Run tests in container"
	@echo "  clean  - Remove containers and images"

build:
	$(compose) build

up:
	$(compose) up -d

down:
	$(compose) down

logs:
	$(compose) logs -f

shell:
	$(compose) exec app bash

test:
	$(compose) exec app uv run pytest tests/

clean:
	$(compose) down
	docker system prune -f
# Simple Docker Compose commands

.PHONY: help build up down logs shell clean

help:
	@echo "Available commands:"
	@echo "  build  - Build Docker image"
	@echo "  up     - Start the application"
	@echo "  down   - Stop the application"
	@echo "  logs   - View application logs"
	@echo "  shell  - Open shell in container"
	@echo "  clean  - Remove containers and images"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec app bash

clean:
	docker-compose down
	docker system prune -f
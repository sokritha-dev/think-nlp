# Makefile

# ▶ Run the dev server
dev:
	ENV=local uvicorn app.main:app --reload

# 🧹 Lint with ruff (fast Python linter)
lint:
	ruff app tests

# 🎨 Auto-format code with black
format:
	black app tests

# 🧪 Check formatting issues only (dry run)
format-check:
	black --check app tests

# 🔥 Clean up pycache
clean:
	find . -type d -name __pycache__ -exec rm -r {} +


# 🧪 Run tests with coverage
test:
	ENV=local @PYTHONPATH=. pytest --cov=app app/tests/ --cov-report=term-missing

# 🧪 Run tests with HTML report
test-html:
	ENV=local @PYTHONPATH=. pytest --cov=app app/tests/ --cov-report=html

# 🧪 Run LOAD tests with proper HTML report
load-test:
	@echo "Running Locust load test with CSV report..."
	@mkdir -p reports
	locust -f locustfile.py \
		--headless \
		--users 50 \
		--spawn-rate 10 \
		--run-time 2m \
		--host http://localhost:8000 \
		--csv reports/upload_test
		
render-report:
	python scripts/render_report.py reports/upload_test_stats.csv

view-report:
	python -m http.server 5500


# ----------------------------
# 🐳 Docker Database Commands
# ----------------------------
up:
	docker-compose --env-file .env.local up -d --build
	@echo "✅ PostgreSQL is up and running."

down:
	docker-compose down
	@echo "🛑 PostgreSQL stopped."

restart:
	docker-compose down
	docker-compose up -d --build
	@echo "🔁 PostgreSQL restarted."

logs:
	docker-compose logs -f postgres

ps:
	docker-compose ps

reset-db:
	docker-compose down -v
	docker-compose up -d
	@echo "🚨 PostgreSQL reset (all data wiped!)"

pgadmin:
	open http://localhost:5050


# Environment configuration
ENV ?= local
SERVICE_NAME ?= app  # Change this if your Docker service is named differently

# Alembic: Initialize migrations directory
init-alembic:
	docker compose exec $(SERVICE_NAME) alembic init migrations

# Alembic: Create new revision (requires message m="...")
revision:
ifndef m
	$(error ❌ Please provide a message with m="your message")
endif
	docker compose exec $(SERVICE_NAME) alembic revision --autogenerate -m "$(m)"

# Alembic: Apply latest migration
migrate:
	docker compose exec $(SERVICE_NAME) alembic upgrade head

# Alembic: Downgrade one revision
downgrade:
	docker compose exec $(SERVICE_NAME) alembic downgrade -1

# Alembic: Check current DB version
current:
	docker compose exec $(SERVICE_NAME) alembic current

# Alembic: Show migration history
history:
	docker compose exec $(SERVICE_NAME) alembic history

# Show help
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make init-alembic         Initialize Alembic in your project"
	@echo "  make revision m=\"msg\"     Create a migration with message"
	@echo "  make migrate              Upgrade DB to latest revision"
	@echo "  make downgrade            Downgrade one revision"
	@echo "  make current              Show current DB version"
	@echo "  make history              Show migration history"
	@echo ""






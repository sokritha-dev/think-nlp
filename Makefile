# ============================
# 🌐 ENVIRONMENT VARIABLES
# ============================

ENV ?= local
SERVICE_NAME ?= app  # Docker Compose service name (use "app" for prod, "app-dev" for dev)

# ============================
# 🚀 APP COMMANDS
# ============================

# ▶ Local dev without Docker
dev:
	ENV=development uvicorn app.main:app --reload

# ▶ Start app in development mode (Docker)
up-dev:
	docker compose -f docker-compose.dev.yml up -d --build
	@echo "🧪 NLP App running in development mode."

# ▶ Start app in production mode (Docker)
up-prod:
	docker compose -f docker-compose.yml up -d --build
	@echo "🚀 NLP App running in production mode."

# ▶ Stop dev/prod
down-dev:
	docker compose -f docker-compose.dev.yml down
	@echo "🛑 Dev environment stopped."

down-prod:
	docker compose -f docker-compose.yml down
	@echo "🛑 Production environment stopped."

# ▶ View logs
logs-dev:
	docker compose -f docker-compose.dev.yml logs -f app

logs-prod:
	docker compose -f docker-compose.yml logs -f app

# ▶ Build images manually
build-prod:
	docker compose -f docker-compose.yml build

build-dev:
	docker compose -f docker-compose.dev.yml build

# ▶ Reset all volumes
reset-all:
	docker compose -f docker-compose.yml down -v
	docker compose -f docker-compose.dev.yml down -v
	@echo "🧨 Reset all volumes and containers!"

# ============================
# ⚙️ CODE QUALITY
# ============================

# 🧹 Lint with ruff
lint:
	ruff app tests

# 🎨 Auto-format with black
format:
	black app tests

# 🧪 Check formatting only
format-check:
	black --check app tests

# 🔥 Clean Python cache
clean:
	find . -type d -name __pycache__ -exec rm -r {} +

# ============================
# 🧪 TESTING
# ============================

test:
	ENV=local PYTHONPATH=. pytest --cov=app app/tests/ --cov-report=term-missing

test-html:
	ENV=local PYTHONPATH=. pytest --cov=app app/tests/ --cov-report=html

# ============================
# 📈 LOAD TESTING (Locust)
# ============================

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

# ============================
# 🐘 POSTGRES & ALEMBICd
# ============================

init-alembic:
	docker compose exec $(SERVICE_NAME) alembic init migrations

revision:
ifndef m
	$(error ❌ Please provide a message with m="your message")
endif
	docker compose exec $(SERVICE_NAME) alembic revision --autogenerate -m "$(m)"

migrate:
	docker compose exec $(SERVICE_NAME) alembic upgrade head

downgrade:
	docker compose exec $(SERVICE_NAME) alembic downgrade -1

current:
	docker compose exec $(SERVICE_NAME) alembic current

history:
	docker compose exec $(SERVICE_NAME) alembic history

pgadmin:
	open http://localhost:5050

# ============================
# 📚 HELP
# ============================

help:
	@echo ""
	@echo "🔧 Docker:"
	@echo "  make up-dev           Start development server"
	@echo "  make up-prod          Start production server"
	@echo "  make down-dev         Stop development server"
	@echo "  make down-prod        Stop production server"
	@echo "  make reset-all        Remove all containers and volumes"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test             Run unit tests with coverage"
	@echo "  make test-html        Generate HTML test report"
	@echo "  make load-test        Run Locust performance test"
	@echo ""
	@echo "🎨 Quality:"
	@echo "  make lint             Lint code with ruff"
	@echo "  make format           Auto-format code with black"
	@echo "  make clean            Delete __pycache__ folders"
	@echo ""
	@echo "🗄️  Database:"
	@echo "  make migrate          Apply latest Alembic migration"
	@echo "  make downgrade        Revert one migration"
	@echo "  make revision m=\"msg\"  Create a migration revision"
	@echo ""

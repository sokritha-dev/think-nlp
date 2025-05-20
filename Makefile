# ============================
# 🌍 Active Environment
# ============================
ENV ?= local
ENV_FILE := .env.$(ENV)

# Load ENV-specific values if the file exists
ifneq (,$(wildcard $(ENV_FILE)))
  include $(ENV_FILE)
  export
endif

SERVICE_NAME ?= app
APP_IMAGE ?= $(DOCKERHUB_USERNAME)/think-nlp-app
TAG ?= latest

# ============================
# Debugging
# ============================
debug:
	@echo "ENV: $(ENV)"
	@echo "DOCKERHUB_USERNAME: $(DOCKERHUB_USERNAME)"
	@echo "APP_IMAGE: $(APP_IMAGE)"
	@echo "DROPLET_HOST: $(DROPLET_HOST)"


# ============================
# 🚀 DEPLOYMENT
# ============================

# ▶ Deploy a specific image tag to the Droplet
deploy:
	ssh $(DROPLET_USER)@$(DROPLET_HOST) "docker pull $(APP_IMAGE):$(TAG) && docker compose -f docker-compose.production.yml up -d"

# ▶ Rollback to a specific image tag manually
rollback:
ifndef TAG
	$(error ❌ Please provide a rollback TAG like TAG=2024-05-12)
endif
	ssh $(DROPLET_USER)@$(DROPLET_HOST) "./rollback.sh $(TAG)"

# ▶ Fuzzy-select rollback tag using fzf
rollback-select:
	ssh $(DROPLET_USER)@$(DROPLET_HOST) "./rollback.sh"

# ▶ View currently deployed tag
deployed-tag:
	ssh $(DROPLET_USER)@$(DROPLET_HOST) "docker ps --filter name=$(SERVICE_NAME) --format '{{.Image}}'"

# ============================
# 🚀 APP COMMANDS (LOCAL DEV)
# ============================

# ▶ Local dev without Docker
dev:
	ENV=development uvicorn app.main:app --reload

# ▶ Start dev/prod with Docker Compose
up-dev:
	docker compose -f docker-compose.development.yml up -d --build

up-local:
	docker compose -f docker-compose.local.yml up -d --build

up-prod:
	docker compose -f docker-compose.production.yml up -d --build

# ▶ Stop dev/prod
down-dev:
	docker compose -f docker-compose.development.yml down

down-local:
	docker compose -f docker-compose.local.yml down

down-prod:
	docker compose -f docker-compose.production.yml down

# ▶ View logs
logs-local:
	docker compose -f docker-compose.local.yml logs -f app

logs-prod:
	docker compose -f docker-compose.production.yml logs -f app

# ▶ Reset all volumes
reset-all:
	docker compose -f docker-compose.yml down -v
	docker compose -f docker-compose.dev.yml down -v

# ============================
# ⚙️ CODE QUALITY
# ============================

lint:
	ruff app tests

format:
	black app tests

format-check:
	black --check app tests

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
# 📈 LOAD TESTING
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
# 🐘 POSTGRES & ALEMBIC
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
	@echo "🔧 Docker Deployment:"
	@echo "  make deploy TAG=2024-05-19 DROPLET_HOST=ip ..."
	@echo "  make rollback TAG=2024-05-12"
	@echo "  make rollback-select         # Uses fuzzy select"
	@echo ""
	@echo "🚀 App Dev/Prod Commands:"
	@echo "  make up-dev / down-dev / logs-dev"
	@echo "  make up-prod / down-prod / logs-prod"
	@echo ""
	@echo "🧪 Testing & Load Test:"
	@echo "  make test / test-html / load-test"
	@echo ""
	@echo "🎨 Code Quality:"
	@echo "  make lint / format / clean"
	@echo ""
	@echo "🗄️ Database Migrations (Alembic):"
	@echo "  make migrate / downgrade / current / history"
	@echo ""
	@echo "ℹ️ All targets are overrideable with variables like TAG, DOCKERHUB_USERNAME, etc."

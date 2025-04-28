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

# Run Alembic migrations
init-alembic:
	alembic init migrations

revision:
ifndef m
	$(error ❌ Please provide a message with m="message")
endif
	ENV=local alembic revision --autogenerate -m "$(m)"

migrate:
	ENV=local alembic upgrade head






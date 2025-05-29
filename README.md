# ThinkNLP 🧠💬
*Interactive NLP Learning Platform for Beginners*

[![🚀 Build and Deploy to DigitalOcean Kubernetes](https://github.com/sokritha-dev/think-nlp/actions/workflows/deploy.yml/badge.svg)](https://github.com/sokritha-dev/think-nlp/actions/workflows/deploy.yml)
[![License](https://img.shields.io/github/license/sokritha-dev/think-nlp)]()

🔗 [Live Demo](https://www.thinknlp.xyz)

---

## 📚 Project Overview
**ThinkNLP** is an educational web application designed to help beginners in Natural Language Processing (NLP) understand the full pipeline of sentiment and topic analysis using real-world review data. It provides a step-by-step, no-code interface to interactively explore how NLP models work.

---

## ✨ Features
- Full NLP pipeline walkthrough
- Upload and process real review data
- Compare sentiment models (VADER, TextBlob, BERT)
- Topic modeling with LDA and interactive visualization (pyLDAvis)
- Manual and auto topic labeling
- Sentiment distribution per topic
- Beginner-friendly UI with visual explanations

---

## ⚙️ NLP Pipeline

1. **Upload Review File**
   - Supports CSV input, stored in AWS S3 (gzip compressed)

2. **Data Cleaning**
   - Normalization: Lowercasing, typo correction
   - Special character removal: special character, number, and emoji
   - Tokenization: Breaking sentences into words 
   - Stopword Removal: remove common words that not provide useful context 
   - Lemmatization: Converting words to their base forms

3. **EDA (Exploratory Data Analysis)**
   - Word clouds, frequency charts, sentence length plots

4. **Topic Modeling**
   - LDA with auto or manual topic count
   - pyLDAvis visualization

5. **Topic Labeling**
   - Manual, keyword-based, or auto-inferred labels

6. **Sentiment Analysis**
   - VADER: Rule-based model
   - TextBlob: Lexicon-based
   - BERT: Transformer-based classifier (optional)

7. **Sentiment-Topic Mapping**
   - Each sentence assigned a dominant topic
   - Sentiment computed per topic
   - Output: Sentiment distribution per topic (Positive, Neutral, Negative)

---

## 🧱 Architecture
- **Frontend**: React + TanStack Query (Vercel)
- **Backend**: FastAPI + PostgreSQL (Docker, DigitalOcean)
- **Storage**: AWS S3 for file uploads
- **Monitoring**: BetterStack for logs and metrics

---

## 🛡️ Security & Optimization
- Gzip file compression
- Rate limiting & security headers
- Future: Background processing with Celery

---

## 🚧 Future Roadmap
- ✅ User authentication & file history
- ✅ Background task support (Celery + status UI)
- ✅ Expanded model selection and interpretability features
- ✅ Beginner tutorials and automatic result summaries
- ✅ Support with Multiple Languages beside English Language

---

## 👩‍🎓 Target Audience
- NLP beginners and students
- Educators and instructors
- Developers interested in NLP and no-code tools

---

## 🧪 Getting Started

### Prerequisites
- Docker
- Python 3.11

### Backend Setup
```bash
cd root_project
cp .env.example .env
make migrate        # Setup initial database schema
make up-local       # Run the backend server
```

### Frontend Setup (Separate Repo)
```bash
git clone https://github.com/sokritha-dev/think-nlp-frontend.git
cd root_project
yarn add
yarn dev
```

### 📁 Folder Structure
```bash
think-nlp/
├── .github/workflows/        # GitHub Actions workflows
├── .vscode/                  # VSCode editor settings
├── app/                      # FastAPI application code
├── k8s/                      # Kubernetes manifests
├── metric/                   # Monitoring & metrics utilities
├── migrations/               # Alembic migration files
├── reports/                  # Load test and analysis reports
├── scripts/                  # Helper and automation scripts
├── .autoenv.zsh              # Autoenv activation for Zsh
├── .dockerignore             # Docker ignore rules
├── .env.sample               # Example environment variables
├── .gitignore                # Git ignore rules
├── .python-version           # Python version pinning
├── Dockerfile                # Production Dockerfile
├── Dockerfile.dev            # Development Dockerfile
├── LICENSE                   # MIT License
├── Makefile                  # CLI automation for dev/test/deploy
├── alembic.ini               # Alembic configuration
├── docker-compose.*.yml      # Docker Compose files for different envs
├── locustfile.py             # Locust load testing script
├── pytest.ini                # Pytest config
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
```

### 📝 License
This project is licensed under the MIT License.

### ❤️ Acknowledgements
- Built using FastAPI, React, and pyLDAvis
- NLP components inspired by open-source models and tutorials
# ---------------------
# Base image
# ---------------------
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install only minimal OS packages for production
RUN apt-get update && apt-get install -y build-essential wget curl && apt-get clean

# ---------------------
# Install dependencies
# ---------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -m nltk.downloader vader_lexicon stopwords wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng

# (Optional) Preload BERT model
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

# Copy application code
COPY . .

# Set environment
ENV ENV=production

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

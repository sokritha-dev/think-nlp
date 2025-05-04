# Dockerfile

# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, but safer for building some packages)
RUN apt-get update && apt-get install -y build-essential wget curl

# Copy only necessary files first (to cache pip install)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK vader_lexicon during image build
RUN python -m nltk.downloader vader_lexicon stopwords wordnet

# Pre-download HuggingFace models
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

# Then copy the rest of the application code
COPY . .

# Set environment variables (optional)
ENV ENV=local

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]


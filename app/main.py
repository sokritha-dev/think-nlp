from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import clean, upload
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


app = FastAPI(
    title="NLP Pipeline API",
    description="Step-by-step NLP API for topic modeling and sentiment analysis",
    version="1.0.0",
)

# Allow CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload.router)
app.include_router(clean.router)
# app.include_router(eda.router, prefix="/api/eda")
# app.include_router(topic_model.router, prefix="/api/topic-model")
# app.include_router(topic_label.router, prefix="/api/topic-label")
# app.include_router(sentence_analysis.router, prefix="/api/sentence-analysis")

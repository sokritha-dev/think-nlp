from fastapi import APIRouter
from app.services.sentiment import predict_sentiment

router = APIRouter(prefix="/reviews", tags=["Reviews"])

@router.post("/predict_sentiment")
def analyze_sentiment(review_text: str):
    """Predict the sentiment of a hotel review."""
    sentiment = predict_sentiment(review_text)
    return {"review": review_text, "sentiment": sentiment}

from fastapi import FastAPI
# from app.routes import reviews

app = FastAPI(title="Hotel Review Analysis API")

# Include API routes
# app.include_router(reviews.router)


@app.get("/")
def home():
    return {"message": "Welcome to the Hotel Review Analysis API"}

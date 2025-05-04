from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import RecommendRequest
from .data_loader import load_and_process_data
from .recommender import get_recommendations

app = FastAPI(title="Movie Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

movies_with_tmdb, vectorizer, tfidf_matrix, knn_model = load_and_process_data()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Movie Recommendation API",
        "endpoints": {
            "/recommend": "POST endpoint for getting movie recommendations"
        }
    }

@app.post("/recommend")
def recommend(request: RecommendRequest):
    recommendations = get_recommendations(
        request.movie_ids, movies_with_tmdb, vectorizer, tfidf_matrix, knn_model
    )
    return {"recommendations": recommendations}

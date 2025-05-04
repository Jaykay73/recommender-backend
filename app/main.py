from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .recommender import (
    load_and_prepare_data, load_models, build_models, get_recommendations
)

app = FastAPI()

movies_with_tmdb = load_and_prepare_data()

vectorizer, knn_model = load_models()
if not vectorizer or not knn_model:
    vectorizer, knn_model, tfidf_matrix = build_models(movies_with_tmdb)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_matrix = vectorizer.transform(movies_with_tmdb['metadata'])

class RecommendRequest(BaseModel):
    movie_ids: List[int]

@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        if not request.movie_ids:
            raise HTTPException(status_code=400, detail="No movie IDs provided")
        
        # Ensure that all movie IDs exist in the data
        missing_ids = [id for id in request.movie_ids if id not in movies_with_tmdb['tmdb_id'].values]
        if missing_ids:
            raise HTTPException(status_code=400, detail=f"Movie IDs {missing_ids} not found.")
        
        recommendations = get_recommendations(
            request.movie_ids, movies_with_tmdb, vectorizer, tfidf_matrix, knn_model
        )
        return {"recommendations": recommendations}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

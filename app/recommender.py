from scipy.sparse import vstack
from fastapi import HTTPException
from .tmdb import fetch_tmdb_movie

def get_recommendations(movie_ids, movies_with_tmdb, vectorizer, tfidf_matrix, knn_model):
    if not movie_ids:
        raise HTTPException(status_code=400, detail="No movie IDs provided")

    tmdb_set = set(movies_with_tmdb['tmdbId'])
    query_vectors = []

    for tmdb_id in movie_ids:
        if tmdb_id in tmdb_set:
            idx = movies_with_tmdb.index[movies_with_tmdb['tmdbId'] == tmdb_id][0]
            metadata = movies_with_tmdb.at[idx, 'metadata']
        else:
            tmdb_data = fetch_tmdb_movie(tmdb_id)
            if tmdb_data is None:
                continue
            title, genres, keywords = tmdb_data
            metadata = f"{title} {genres} {keywords}"
        query_vec = vectorizer.transform([metadata])
        query_vectors.append(query_vec)

    if not query_vectors:
        raise HTTPException(status_code=400, detail="No valid movie metadata found for given IDs")

    all_vecs = vstack(query_vectors)
    combined_vec = all_vecs.mean(axis=0).A  # `.A` converts sparse matrix to ndarray

    n_neighbors = min(tfidf_matrix.shape[0], 15 + len(movie_ids))
    distances, indices = knn_model.kneighbors(combined_vec, n_neighbors=n_neighbors)

    recommendations = []
    for idx in indices.flatten():
        candidate = int(movies_with_tmdb.iloc[idx]['tmdbId'])
        if candidate not in movie_ids:
            recommendations.append(candidate)
        if len(recommendations) >= 15:
            break

    return recommendations

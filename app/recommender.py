import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack
from .model_utils import load_models, save_models

TMDB_API_KEY = "YOUR_TMDB_API_KEY"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def load_and_prepare_data():
    movies_df = pd.read_csv("movies.csv")
    tags_df = pd.read_csv("tags.csv")
    links_df = pd.read_csv("links.csv")

    tags_grouped = tags_df.groupby('movieId')['tag'] \
        .apply(lambda tags: ' '.join(str(tag) for tag in tags if pd.notna(tag))).reset_index()
    tags_grouped.rename(columns={'tag': 'tags'}, inplace=True)
    movies_df = pd.merge(movies_df, tags_grouped, on='movieId', how='left')
    movies_df['tags'] = movies_df['tags'].fillna('')
    movies_df['metadata'] = movies_df['title'] + ' ' + movies_df['genres'].fillna('') + ' ' + movies_df['tags']

    movies_with_tmdb = pd.merge(movies_df, links_df[['movieId', 'tmdbId']], on='movieId', how='inner')
    movies_with_tmdb = movies_with_tmdb.dropna(subset=['tmdbId'])
    movies_with_tmdb['tmdbId'] = movies_with_tmdb['tmdbId'].astype(int)

    return movies_with_tmdb

def build_models(movies_with_tmdb):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_with_tmdb['metadata'])
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    save_models(vectorizer, knn_model)
    return vectorizer, knn_model, tfidf_matrix

def fetch_tmdb_movie(tmdb_id):
    details_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    keywords_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/keywords?api_key={TMDB_API_KEY}"
    try:
        details_data = requests.get(details_url).json()
        title = details_data.get('title', '')
        genres = ' '.join([g['name'] for g in details_data.get('genres', [])])
        keywords_data = requests.get(keywords_url).json()
        keywords = ' '.join([k['name'] for k in keywords_data.get('keywords', [])])
    except Exception:
        return None
    return f"{title} {genres} {keywords}"

def get_recommendations(input_tmdb_ids, movies_with_tmdb, vectorizer, tfidf_matrix, knn_model):
    tmdb_set = set(movies_with_tmdb['tmdbId'])
    query_vectors = []

    for tmdb_id in input_tmdb_ids:
        if tmdb_id in tmdb_set:
            idx = movies_with_tmdb.index[movies_with_tmdb['tmdbId'] == tmdb_id][0]
            metadata = movies_with_tmdb.at[idx, 'metadata']
        else:
            metadata = fetch_tmdb_movie(tmdb_id)
            if not metadata:
                continue
        query_vec = vectorizer.transform([metadata])
        query_vectors.append(query_vec)

    if not query_vectors:
        return []

    combined_vec = vstack(query_vectors).mean(axis=0)
    n_neighbors = min(tfidf_matrix.shape[0], 15 + len(input_tmdb_ids))
    distances, indices = knn_model.kneighbors(combined_vec, n_neighbors=n_neighbors)

    recommendations = []
    for idx in indices.flatten():
        candidate = int(movies_with_tmdb.iloc[idx]['tmdbId'])
        if candidate not in input_tmdb_ids:
            recommendations.append(candidate)
        if len(recommendations) >= 15:
            break
    return recommendations


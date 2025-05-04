import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_and_process_data():
    movies_df = pd.read_csv('data/movies.csv')
    tags_df = pd.read_csv('data/tags.csv')
    links_df = pd.read_csv('data/links.csv')

    tags_grouped = tags_df.groupby('movieId')['tag'] \
        .apply(lambda tags: ' '.join(str(tag) for tag in tags if pd.notna(tag))).reset_index()
    tags_grouped.rename(columns={'tag': 'tags'}, inplace=True)

    movies_df = pd.merge(movies_df, tags_grouped, on='movieId', how='left')
    movies_df['tags'] = movies_df['tags'].fillna('')
    movies_df['metadata'] = (
        movies_df['title'] + ' ' +
        movies_df['genres'].fillna('') + ' ' +
        movies_df['tags']
    )

    movies_with_tmdb = pd.merge(movies_df, links_df[['movieId','tmdbId']], on='movieId', how='inner')
    movies_with_tmdb = movies_with_tmdb.dropna(subset=['tmdbId'])
    movies_with_tmdb['tmdbId'] = movies_with_tmdb['tmdbId'].astype(int)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_with_tmdb['metadata'])

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)

    return movies_with_tmdb, vectorizer, tfidf_matrix, knn_model

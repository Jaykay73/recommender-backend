import requests

TMDB_API_KEY = "bb22ff5ccc9a19ec56cf83148370714c"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def fetch_tmdb_movie(tmdb_id: int):
    details_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    keywords_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/keywords?api_key={TMDB_API_KEY}"
    try:
        details_data = requests.get(details_url).json()
        title = details_data.get('title', '')
        genres = ' '.join([g['name'] for g in details_data.get('genres', [])])
    except Exception:
        return None
    try:
        keywords_data = requests.get(keywords_url).json()
        keywords = ' '.join([k['name'] for k in keywords_data.get('keywords', [])])
    except Exception:
        keywords = ''
    return title, genres, keywords

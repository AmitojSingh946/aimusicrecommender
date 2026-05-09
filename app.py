"""
AI Music Recommender System
============================
Uses Content-Based Filtering with Cosine Similarity (scikit-learn)
to recommend songs based on user preferences.

Each song is represented as a numerical feature vector:
  [energy, danceability, valence, acousticness, tempo_norm]

The user's preferences are converted into an "ideal" feature vector,
and we compute cosine similarity between the user vector and every
song vector to find the best matches.

Spotify API (via spotipy) is used to fetch album art, Spotify links,
and 30-second preview URLs for the recommended songs.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# --- Spotify Integration (graceful fallback if credentials missing) ---
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
# SONG DATABASE
# Each song has metadata + numerical audio features (0.0 - 1.0)
# These features simulate what Spotify's audio analysis provides:
#   energy:       How intense/active (0=calm, 1=explosive)
#   danceability: How suitable for dancing (0=not, 1=very)
#   valence:      Musical positiveness (0=sad, 1=happy)
#   acousticness: How acoustic vs electronic (0=electronic, 1=acoustic)
#   tempo_norm:   Normalized tempo (0=very slow, 1=very fast)
# ============================================================
SONGS = [
    # --- POP ---
    {"id": 1,  "title": "Shape of You",           "artist": "Ed Sheeran",                "genre": "Pop",        "energy": 0.65, "danceability": 0.82, "valence": 0.93, "acousticness": 0.58, "tempo_norm": 0.55},
    {"id": 2,  "title": "Blinding Lights",         "artist": "The Weeknd",                "genre": "Pop",        "energy": 0.73, "danceability": 0.51, "valence": 0.33, "acousticness": 0.00, "tempo_norm": 0.72},
    {"id": 3,  "title": "Uptown Funk",             "artist": "Mark Ronson ft. Bruno Mars", "genre": "Pop",        "energy": 0.89, "danceability": 0.86, "valence": 0.96, "acousticness": 0.03, "tempo_norm": 0.57},
    {"id": 4,  "title": "Someone Like You",         "artist": "Adele",                     "genre": "Pop",        "energy": 0.17, "danceability": 0.33, "valence": 0.07, "acousticness": 0.87, "tempo_norm": 0.34},
    {"id": 5,  "title": "Levitating",              "artist": "Dua Lipa",                  "genre": "Pop",        "energy": 0.67, "danceability": 0.70, "valence": 0.91, "acousticness": 0.01, "tempo_norm": 0.60},
    {"id": 6,  "title": "Fix You",                 "artist": "Coldplay",                  "genre": "Pop",        "energy": 0.45, "danceability": 0.25, "valence": 0.12, "acousticness": 0.55, "tempo_norm": 0.35},
    {"id": 7,  "title": "Happy",                   "artist": "Pharrell Williams",         "genre": "Pop",        "energy": 0.82, "danceability": 0.90, "valence": 0.96, "acousticness": 0.22, "tempo_norm": 0.65},
    {"id": 8,  "title": "Stay With Me",            "artist": "Sam Smith",                 "genre": "Pop",        "energy": 0.42, "danceability": 0.41, "valence": 0.18, "acousticness": 0.59, "tempo_norm": 0.42},

    # --- ROCK ---
    {"id": 9,  "title": "Bohemian Rhapsody",       "artist": "Queen",                     "genre": "Rock",       "energy": 0.40, "danceability": 0.30, "valence": 0.22, "acousticness": 0.31, "tempo_norm": 0.38},
    {"id": 10, "title": "Smells Like Teen Spirit",  "artist": "Nirvana",                   "genre": "Rock",       "energy": 0.91, "danceability": 0.50, "valence": 0.41, "acousticness": 0.05, "tempo_norm": 0.58},
    {"id": 11, "title": "Hotel California",         "artist": "Eagles",                    "genre": "Rock",       "energy": 0.50, "danceability": 0.44, "valence": 0.32, "acousticness": 0.28, "tempo_norm": 0.40},
    {"id": 12, "title": "Sweet Child O' Mine",      "artist": "Guns N' Roses",             "genre": "Rock",       "energy": 0.82, "danceability": 0.42, "valence": 0.51, "acousticness": 0.06, "tempo_norm": 0.62},
    {"id": 13, "title": "Stairway to Heaven",       "artist": "Led Zeppelin",              "genre": "Rock",       "energy": 0.34, "danceability": 0.28, "valence": 0.20, "acousticness": 0.52, "tempo_norm": 0.30},
    {"id": 14, "title": "Back in Black",            "artist": "AC/DC",                     "genre": "Rock",       "energy": 0.93, "danceability": 0.56, "valence": 0.60, "acousticness": 0.03, "tempo_norm": 0.70},
    {"id": 15, "title": "Wonderwall",               "artist": "Oasis",                     "genre": "Rock",       "energy": 0.44, "danceability": 0.37, "valence": 0.36, "acousticness": 0.62, "tempo_norm": 0.43},

    # --- HIP-HOP ---
    {"id": 16, "title": "Lose Yourself",            "artist": "Eminem",                    "genre": "Hip-Hop",    "energy": 0.87, "danceability": 0.72, "valence": 0.35, "acousticness": 0.07, "tempo_norm": 0.72},
    {"id": 17, "title": "SICKO MODE",               "artist": "Travis Scott",              "genre": "Hip-Hop",    "energy": 0.73, "danceability": 0.83, "valence": 0.45, "acousticness": 0.05, "tempo_norm": 0.75},
    {"id": 18, "title": "God's Plan",               "artist": "Drake",                     "genre": "Hip-Hop",    "energy": 0.45, "danceability": 0.75, "valence": 0.60, "acousticness": 0.11, "tempo_norm": 0.40},
    {"id": 19, "title": "Humble",                   "artist": "Kendrick Lamar",            "genre": "Hip-Hop",    "energy": 0.62, "danceability": 0.91, "valence": 0.42, "acousticness": 0.03, "tempo_norm": 0.78},
    {"id": 20, "title": "Old Town Road",            "artist": "Lil Nas X",                 "genre": "Hip-Hop",    "energy": 0.61, "danceability": 0.88, "valence": 0.80, "acousticness": 0.15, "tempo_norm": 0.68},
    {"id": 21, "title": "Hotline Bling",            "artist": "Drake",                     "genre": "Hip-Hop",    "energy": 0.40, "danceability": 0.85, "valence": 0.58, "acousticness": 0.10, "tempo_norm": 0.44},
    {"id": 22, "title": "Stronger",                 "artist": "Kanye West",                "genre": "Hip-Hop",    "energy": 0.73, "danceability": 0.67, "valence": 0.58, "acousticness": 0.02, "tempo_norm": 0.62},

    # --- ELECTRONIC ---
    {"id": 23, "title": "Strobe",                   "artist": "deadmau5",                  "genre": "Electronic", "energy": 0.55, "danceability": 0.42, "valence": 0.25, "acousticness": 0.02, "tempo_norm": 0.62},
    {"id": 24, "title": "Levels",                   "artist": "Avicii",                    "genre": "Electronic", "energy": 0.89, "danceability": 0.56, "valence": 0.85, "acousticness": 0.01, "tempo_norm": 0.63},
    {"id": 25, "title": "Titanium",                 "artist": "David Guetta ft. Sia",      "genre": "Electronic", "energy": 0.78, "danceability": 0.60, "valence": 0.40, "acousticness": 0.05, "tempo_norm": 0.63},
    {"id": 26, "title": "Midnight City",            "artist": "M83",                       "genre": "Electronic", "energy": 0.81, "danceability": 0.53, "valence": 0.38, "acousticness": 0.02, "tempo_norm": 0.58},
    {"id": 27, "title": "Faded",                    "artist": "Alan Walker",               "genre": "Electronic", "energy": 0.49, "danceability": 0.60, "valence": 0.15, "acousticness": 0.03, "tempo_norm": 0.55},
    {"id": 28, "title": "Lean On",                  "artist": "Major Lazer & DJ Snake",    "genre": "Electronic", "energy": 0.71, "danceability": 0.77, "valence": 0.50, "acousticness": 0.04, "tempo_norm": 0.58},

    # --- CLASSICAL ---
    {"id": 29, "title": "Clair de Lune",            "artist": "Claude Debussy",            "genre": "Classical",  "energy": 0.05, "danceability": 0.12, "valence": 0.18, "acousticness": 0.99, "tempo_norm": 0.15},
    {"id": 30, "title": "Moonlight Sonata",         "artist": "Beethoven",                 "genre": "Classical",  "energy": 0.08, "danceability": 0.10, "valence": 0.10, "acousticness": 0.98, "tempo_norm": 0.12},
    {"id": 31, "title": "Canon in D",               "artist": "Pachelbel",                 "genre": "Classical",  "energy": 0.15, "danceability": 0.18, "valence": 0.55, "acousticness": 0.97, "tempo_norm": 0.22},
    {"id": 32, "title": "The Four Seasons - Spring","artist": "Vivaldi",                   "genre": "Classical",  "energy": 0.52, "danceability": 0.30, "valence": 0.72, "acousticness": 0.95, "tempo_norm": 0.50},
    {"id": 33, "title": "Ride of the Valkyries",    "artist": "Wagner",                    "genre": "Classical",  "energy": 0.88, "danceability": 0.22, "valence": 0.45, "acousticness": 0.90, "tempo_norm": 0.70},
    {"id": 34, "title": "Gymnopédie No.1",          "artist": "Erik Satie",                "genre": "Classical",  "energy": 0.03, "danceability": 0.14, "valence": 0.22, "acousticness": 0.99, "tempo_norm": 0.10},

    # --- R&B ---
    {"id": 35, "title": "Blinding Lights",          "artist": "The Weeknd",                "genre": "R&B",        "energy": 0.73, "danceability": 0.51, "valence": 0.33, "acousticness": 0.00, "tempo_norm": 0.72},
    {"id": 36, "title": "Earned It",                "artist": "The Weeknd",                "genre": "R&B",        "energy": 0.35, "danceability": 0.52, "valence": 0.30, "acousticness": 0.30, "tempo_norm": 0.38},
    {"id": 37, "title": "No Scrubs",                "artist": "TLC",                       "genre": "R&B",        "energy": 0.65, "danceability": 0.80, "valence": 0.70, "acousticness": 0.10, "tempo_norm": 0.55},
    {"id": 38, "title": "Say My Name",              "artist": "Destiny's Child",           "genre": "R&B",        "energy": 0.58, "danceability": 0.78, "valence": 0.55, "acousticness": 0.15, "tempo_norm": 0.50},
    {"id": 39, "title": "Thinking Out Loud",        "artist": "Ed Sheeran",                "genre": "R&B",        "energy": 0.44, "danceability": 0.78, "valence": 0.59, "acousticness": 0.70, "tempo_norm": 0.40},
    {"id": 40, "title": "All of Me",                "artist": "John Legend",               "genre": "R&B",        "energy": 0.25, "danceability": 0.42, "valence": 0.33, "acousticness": 0.78, "tempo_norm": 0.32},

    # --- JAZZ ---
    {"id": 41, "title": "Take Five",                "artist": "Dave Brubeck",              "genre": "Jazz",       "energy": 0.35, "danceability": 0.55, "valence": 0.50, "acousticness": 0.88, "tempo_norm": 0.45},
    {"id": 42, "title": "So What",                  "artist": "Miles Davis",               "genre": "Jazz",       "energy": 0.20, "danceability": 0.40, "valence": 0.35, "acousticness": 0.90, "tempo_norm": 0.35},
    {"id": 43, "title": "Fly Me to the Moon",       "artist": "Frank Sinatra",             "genre": "Jazz",       "energy": 0.38, "danceability": 0.60, "valence": 0.75, "acousticness": 0.80, "tempo_norm": 0.50},
    {"id": 44, "title": "Autumn Leaves",            "artist": "Nat King Cole",             "genre": "Jazz",       "energy": 0.18, "danceability": 0.35, "valence": 0.25, "acousticness": 0.92, "tempo_norm": 0.28},
    {"id": 45, "title": "What a Wonderful World",   "artist": "Louis Armstrong",           "genre": "Jazz",       "energy": 0.15, "danceability": 0.30, "valence": 0.80, "acousticness": 0.88, "tempo_norm": 0.22},

    # --- INDIE ---
    {"id": 46, "title": "Sweater Weather",          "artist": "The Neighbourhood",         "genre": "Indie",      "energy": 0.56, "danceability": 0.61, "valence": 0.32, "acousticness": 0.12, "tempo_norm": 0.52},
    {"id": 47, "title": "Electric Feel",            "artist": "MGMT",                      "genre": "Indie",      "energy": 0.60, "danceability": 0.74, "valence": 0.62, "acousticness": 0.08, "tempo_norm": 0.55},
    {"id": 48, "title": "Do I Wanna Know?",         "artist": "Arctic Monkeys",            "genre": "Indie",      "energy": 0.55, "danceability": 0.54, "valence": 0.17, "acousticness": 0.05, "tempo_norm": 0.42},
    {"id": 49, "title": "Pumped Up Kicks",          "artist": "Foster The People",         "genre": "Indie",      "energy": 0.57, "danceability": 0.73, "valence": 0.65, "acousticness": 0.06, "tempo_norm": 0.62},
    {"id": 50, "title": "Somebody That I Used to Know","artist": "Gotye",                  "genre": "Indie",      "energy": 0.52, "danceability": 0.64, "valence": 0.35, "acousticness": 0.25, "tempo_norm": 0.50},
]

# Feature columns used for AI similarity computation
FEATURE_KEYS = ["energy", "danceability", "valence", "acousticness", "tempo_norm"]

# ============================================================
# PRECOMPUTE: Build the feature matrix once at startup
# This is a numpy array where each row is a song's feature vector
# ============================================================
def build_feature_matrix():
    """Convert song features into a normalized numpy matrix for cosine similarity."""
    raw = np.array([[song[k] for k in FEATURE_KEYS] for song in SONGS])
    # MinMaxScaler normalizes each feature column to [0, 1]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(raw)
    return normalized, scaler

FEATURE_MATRIX, SCALER = build_feature_matrix()
app.logger.info(f"Feature matrix built: {FEATURE_MATRIX.shape[0]} songs × {FEATURE_MATRIX.shape[1]} features")

# ============================================================
# SPOTIFY CLIENT (optional — graceful fallback)
# ============================================================
def get_spotify_client():
    """Initialize Spotify client using environment variables. Returns None if unavailable."""
    if not SPOTIPY_AVAILABLE:
        app.logger.warning("spotipy not installed — Spotify features disabled")
        return None

    client_id = os.environ.get("SPOTIPY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET", "")

    if not client_id or not client_secret or client_id == "your_client_id_here":
        app.logger.warning("Spotify credentials not configured — Spotify features disabled")
        return None

    try:
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        # Quick test to verify credentials work
        sp.search(q="test", type="track", limit=1)
        app.logger.info("Spotify client initialized successfully")
        return sp
    except Exception as e:
        app.logger.warning(f"Spotify initialization failed: {e}")
        return None

spotify_client = get_spotify_client()

def fetch_spotify_data(title, artist):
    """Search Spotify for a track and return album art, URL, and preview."""
    if not spotify_client:
        return {"album_art": None, "spotify_url": None, "preview_url": None}

    try:
        query = f"track:{title} artist:{artist}"
        results = spotify_client.search(q=query, type="track", limit=1)
        tracks = results.get("tracks", {}).get("items", [])

        if tracks:
            track = tracks[0]
            album_art = track["album"]["images"][0]["url"] if track["album"]["images"] else None
            return {
                "album_art": album_art,
                "spotify_url": track["external_urls"].get("spotify"),
                "preview_url": track.get("preview_url")
            }
    except Exception as e:
        app.logger.error(f"Spotify search error for '{title}': {e}")

    return {"album_art": None, "spotify_url": None, "preview_url": None}

# ============================================================
# MOOD → FEATURE VECTOR MAPPING
# Converts human-readable mood into ideal audio feature values
# ============================================================
MOOD_PROFILES = {
    "Happy":     {"energy": 0.75, "danceability": 0.80, "valence": 0.90, "acousticness": 0.20, "tempo_norm": 0.65},
    "Sad":       {"energy": 0.20, "danceability": 0.30, "valence": 0.10, "acousticness": 0.70, "tempo_norm": 0.25},
    "Energetic": {"energy": 0.95, "danceability": 0.75, "valence": 0.60, "acousticness": 0.05, "tempo_norm": 0.80},
    "Relaxed":   {"energy": 0.15, "danceability": 0.35, "valence": 0.45, "acousticness": 0.80, "tempo_norm": 0.20},
    "Romantic":  {"energy": 0.30, "danceability": 0.55, "valence": 0.50, "acousticness": 0.60, "tempo_norm": 0.35},
    "Angry":     {"energy": 0.95, "danceability": 0.50, "valence": 0.15, "acousticness": 0.05, "tempo_norm": 0.85},
}

# ============================================================
# AI RECOMMENDATION ENGINE
# ============================================================
def get_recommendations(genre, mood, energy_level):
    """
    Core AI function: Content-Based Filtering with Cosine Similarity.

    Steps:
    1. Build a user preference vector from mood profile + energy level
    2. Filter songs by genre (if specified)
    3. Compute cosine similarity between user vector and song vectors
    4. Return top 5 most similar songs
    """

    # Step 1: Build user preference vector
    mood_profile = MOOD_PROFILES.get(mood, MOOD_PROFILES["Happy"])
    user_vector = np.array([[
        mood_profile["energy"] * (energy_level / 10),       # Scale energy by user's slider
        mood_profile["danceability"],
        mood_profile["valence"],
        mood_profile["acousticness"],
        mood_profile["tempo_norm"] * (energy_level / 10)    # Tempo also influenced by energy
    ]])

    # Normalize user vector using the same scaler fitted on songs
    user_vector_normalized = SCALER.transform(user_vector)

    # Step 2: Filter by genre (or use all songs)
    if genre and genre != "All":
        indices = [i for i, s in enumerate(SONGS) if s["genre"] == genre]
    else:
        indices = list(range(len(SONGS)))

    if not indices:
        return []

    # Step 3: Compute cosine similarity between user vector and filtered songs
    filtered_matrix = FEATURE_MATRIX[indices]
    similarities = cosine_similarity(user_vector_normalized, filtered_matrix)[0]

    # Step 4: Rank songs by similarity score
    ranked = sorted(zip(indices, similarities), key=lambda x: x[1], reverse=True)

    # Step 5: Build results with metadata + Spotify data
    results = []
    for song_idx, sim_score in ranked[:5]:
        song = SONGS[song_idx]
        spotify_data = fetch_spotify_data(song["title"], song["artist"])

        # Calculate why it matched — feature-level breakdown
        song_vector = FEATURE_MATRIX[song_idx]
        user_vec = user_vector_normalized[0]
        feature_contributions = {}
        for i, key in enumerate(FEATURE_KEYS):
            diff = abs(song_vector[i] - user_vec[i])
            match_pct = max(0, (1 - diff)) * 100
            feature_contributions[key] = round(match_pct)

        results.append({
            "id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "similarity_score": round(float(sim_score) * 100, 1),
            "features": {k: round(song[k], 2) for k in FEATURE_KEYS},
            "feature_match": feature_contributions,
            "album_art": spotify_data["album_art"],
            "spotify_url": spotify_data["spotify_url"],
            "preview_url": spotify_data["preview_url"],
        })

    return results

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/genres")
def get_genres():
    """Return list of available genres."""
    genres = sorted(set(song["genre"] for song in SONGS))
    return jsonify({"genres": genres})

@app.route("/recommend", methods=["POST"])
def recommend():
    """Main recommendation endpoint — receives user prefs, returns AI-matched songs."""
    data = request.json
    genre = data.get("genre", "All")
    mood = data.get("mood", "Happy")
    energy_level = int(data.get("energy", 5))

    results = get_recommendations(genre, mood, energy_level)

    return jsonify({
        "recommendations": results,
        "ai_method": "Content-Based Filtering with Cosine Similarity",
        "total_songs_analyzed": len(SONGS),
        "features_used": FEATURE_KEYS,
        "spotify_connected": spotify_client is not None
    })

# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)

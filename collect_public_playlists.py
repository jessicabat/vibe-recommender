import os
import time
import math
import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# =========================
# 1. AUTH: PUBLIC-ONLY CLIENT
# =========================

# Load SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET from .env
# load_dotenv()
# CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
# CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")


# if CLIENT_ID is None or CLIENT_SECRET is None:
#     raise ValueError("Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in your .env file")
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
print("Looking for .env at:", env_path)
print("Exists?", env_path.exists())

load_dotenv(env_path)
print("Loaded CLIENT_ID:", os.getenv("SPOTIPY_CLIENT_ID"))
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")


# Use client credentials flow (no user data, public only)
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(auth_manager=auth_manager)


# =========================
# 2. CONFIG: WHICH MOODS / PLAYLISTS TO COLLECT
# =========================

# Mood-ish queries to search for public playlists
MOOD_QUERIES = [
    "chill",
    "study",
    "focus",
    "sleep",
    "happy",
    "sad",
    "workout",
    "party",
    "rainy day",
    "romantic"
]

PLAYLISTS_PER_QUERY = 5      # how many playlists to grab per search query
MAX_TRACKS_PER_PLAYLIST = 80 # cap per playlist so it doesn’t explode


# =========================
# 3. HELPER: GET TRACKS FROM A PLAYLIST
# =========================

def get_tracks_from_playlist(playlist_id, max_tracks=MAX_TRACKS_PER_PLAYLIST):
    """
    Fetch up to max_tracks track objects from a Spotify playlist.
    Skips local tracks and episodes.
    Returns a list of track dicts.
    """
    tracks = []
    limit = 100
    offset = 0

    while True:
        remaining = max_tracks - len(tracks)
        if remaining <= 0:
            break

        batch = sp.playlist_items(
            playlist_id,
            offset=offset,
            limit=min(limit, remaining),
            additional_types=['track']  # only music tracks
        )

        items = batch.get("items", [])
        if not items:
            break

        for item in items:
            track = item.get("track")
            if track is None:
                continue
            # Skip local tracks, podcasts, etc.
            if track.get("id") is None:
                continue
            tracks.append(track)

        offset += len(items)

        if len(items) < limit:
            break

    return tracks


# =========================
# 4. MAIN COLLECTION LOOP
# =========================

all_rows = []

for query in MOOD_QUERIES:
    print(f"Searching playlists for query: '{query}'")

    search_results = sp.search(q=query, type='playlist', limit=PLAYLISTS_PER_QUERY)
    playlists = search_results.get('playlists', {}).get('items', [])

    for pl in playlists:
    # Skip weird / empty entries
        if pl is None:
            continue

        playlist_id = pl.get("id")
        playlist_name = pl.get("name", "Unknown playlist")

        owner = pl.get("owner") or {}
        owner_name = owner.get("display_name") or owner.get("id", "Unknown owner")

        if playlist_id is None:
            print("  -> Skipping playlist with no ID:", playlist_name)
            continue

        print(f"  -> Collecting from playlist: {playlist_name} (owner: {owner_name})")

        try:
            tracks = get_tracks_from_playlist(playlist_id, max_tracks=MAX_TRACKS_PER_PLAYLIST)
            print(f"     Found {len(tracks)} tracks")

            # ... audio_features batch code from above ...
            # ... building rows and appending to all_rows ...

        except SpotifyException as e:
            print(f"  !! Error while processing playlist {playlist_name} ({playlist_id}), skipping it:", e)
            continue

        # Collect track IDs for audio_features
        track_ids = [t.get("id") for t in tracks if t.get("id")]

        audio_features_map = {}
        BATCH_SIZE = 50  # smaller than 100 to be extra safe

        for i in range(0, len(track_ids), BATCH_SIZE):
            batch_ids = track_ids[i:i + BATCH_SIZE]

            try:
                feats = sp.audio_features(batch_ids)
            except SpotifyException as e:
                print("     !! Skipping a batch of audio_features because of error:", e)
                # skip this batch and move on to the next one
                continue

            if not feats:
                continue

            # feats is a list (same length as batch_ids), some entries may be None
            for tid, feat in zip(batch_ids, feats):
                if feat is None:
                    continue
                audio_features_map[tid] = feat

            # tiny sleep to avoid hammering the API
            time.sleep(0.1)

        # Build rows
        for t in tracks:
            tid = t.get("id")
            if tid is None:
                continue
            if tid not in audio_features_map:
                continue

            feat = audio_features_map[tid]
            artists = ", ".join([a["name"] for a in t.get("artists", [])])

            row = {
                "track_id": tid,
                "track_name": t.get("name", ""),
                "artist_names": artists,
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "playlist_query": query,

                "danceability": feat["danceability"],
                "energy": feat["energy"],
                "valence": feat["valence"],
                "acousticness": feat["acousticness"],
                "instrumentalness": feat["instrumentalness"],
                "liveness": feat["liveness"],
                "speechiness": feat["speechiness"],
                "tempo": feat["tempo"],
                "loudness": feat["loudness"],
                "key": feat["key"],
                "mode": feat["mode"],
                "time_signature": feat["time_signature"],
            }

            all_rows.append(row)

        # Be nice to the API
        time.sleep(0.2)


# =========================
# 5. SAVE TO CSV
# =========================

df = pd.DataFrame(all_rows)
print(f"\nCollected {len(df)} rows total")

# (Optional) Drop exact duplicates (same track in same playlist)
df = df.drop_duplicates(subset=["track_id", "playlist_id"])

print(f"After dropping duplicates: {len(df)} rows")

output_path = "spotify_vibes_public_tracks.csv"
df.to_csv(output_path, index=False)
print(f"Saved data to {output_path}")
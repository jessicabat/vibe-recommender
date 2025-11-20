from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

scope = "user-top-read user-library-read playlist-read-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope
))

print("REDIRECT_URI from env:", REDIRECT_URI)

results = sp.current_user_top_tracks(limit=10, time_range="medium_term")

print("Your top 10 tracks:")
for i, item in enumerate(results["items"], start=1):
    name = item["name"]
    artists = ", ".join([a["name"] for a in item["artists"]])
    print(f"{i}. {name} — {artists}")
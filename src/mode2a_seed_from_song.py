from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from vibe_engine_full import VibeEngine


class Mode2ASeedFromSong:

    def __init__(self, engine: VibeEngine):
        self.engine = engine
        self.df = engine.df

    def search_tracks(
        self,
        query: str,
        max_results: int = 10,
    ) -> pd.DataFrame:

        if not query:
            return self.df.head(0).copy()

        q = query.lower()

        has_track_name = "track_name" in self.df.columns
        has_artists = "artists" in self.df.columns

        if not (has_track_name or has_artists):
            raise ValueError("Dataframe must have 'track_name' or 'artists' columns.")

        conds = []
        if has_track_name:
            conds.append(self.df["track_name"].str.lower().str.contains(q, na=False))
        if has_artists:
            conds.append(self.df["artists"].str.lower().str.contains(q, na=False))

        mask = conds[0]
        for c in conds[1:]:
            mask = mask | c

        results = self.df[mask].copy()
        return results.head(max_results)

    def _resolve_seed_index(
        self,
        track_id: Optional[str] = None,
        df_index: Optional[int] = None,
    ) -> int:

        if df_index is not None:
            if df_index not in self.df.index:
                raise ValueError(f"df_index {df_index} is not a valid index in engine.df")
            return int(df_index)

        if track_id is None:
            raise ValueError("Either track_id or df_index must be provided.")

        if hasattr(self.engine, "trackid_to_idx") and self.engine.trackid_to_idx:
            mapping = self.engine.trackid_to_idx
            if track_id not in mapping:
                raise ValueError(f"track_id {track_id} not found in engine.trackid_to_idx")
            return int(mapping[track_id])

        if "track_id" not in self.df.columns:
            raise ValueError(
                "engine.trackid_to_idx is empty and df lacks track_id column; "
                "cannot resolve track_id."
            )

        matches = self.df.index[self.df["track_id"] == track_id].tolist()
        if not matches:
            raise ValueError(f"track_id {track_id} not found in dataframe.")

        return int(matches[0])

    def recommend_from_seed(
        self,
        track_id: Optional[str] = None,
        df_index: Optional[int] = None,
        top_k: int = 10,
        lambda_vibe: float = 0.8,
        min_popularity: Optional[int] = None,
        hide_explicit: bool = True,
        allowed_genres: Optional[List[str]] = None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
    ) -> Tuple[pd.Series, pd.DataFrame, int]:

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        seed_idx = self._resolve_seed_index(track_id=track_id, df_index=df_index)

        seed_vec = self.engine.X[seed_idx]


        raw_recs = self.engine.score_from_target_vec(
            target_vec=seed_vec,
            top_k=top_k + 1,
            lambda_vibe=lambda_vibe,
            min_popularity=min_popularity,
            hide_explicit=hide_explicit,
            allowed_genres=allowed_genres,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )

        if "track_id" in raw_recs.columns and "track_id" in self.df.columns:
            seed_track_id = self.df.loc[seed_idx, "track_id"]
            recs = raw_recs[raw_recs["track_id"] != seed_track_id]
        else:
            if seed_idx in raw_recs.index:
                recs = raw_recs.drop(index=seed_idx)
            else:
                recs = raw_recs

        if recs.empty:
            raise RuntimeError("No recommendations could be generated from this seed.")

        recs = recs.iloc[:top_k]

        now_playing = recs.iloc[0]
        up_next = recs.iloc[1:].copy()

        return now_playing, up_next, seed_idx


    def explain_recommendation(
        self,
        seed_idx: int,
        rec_df_index: int,
        top_n_features: int = 3,
    ) -> str:

        if seed_idx not in self.df.index:
            raise ValueError(f"seed_idx {seed_idx} not found in df index.")
        if rec_df_index not in self.df.index:
            raise ValueError(f"rec_df_index {rec_df_index} not found in df index.")

        seed_vec = self.engine.X[self.df.index.get_loc(seed_idx)]
        rec_vec = self.engine.X[self.df.index.get_loc(rec_df_index)]

        diff = np.abs(seed_vec - rec_vec)

        feature_diffs = list(zip(self.engine.vibe_cols, diff))
        feature_diffs.sort(key=lambda x: x[1])

        top_features = [name for name, _ in feature_diffs[:top_n_features]]

        friendly_names = {
            "danceability": "danceability 💃",
            "energy": "energy ⚡",
            "valence": "positivity 😃",
            "tempo": "tempo 🥁",
            "acousticness": "acoustic feel 🎸",
            "instrumentalness": "instrumental vibe 🎼",
            "speechiness": "speechiness 🗣️",
        }

        pretty_features = [friendly_names.get(f, f) for f in top_features]

        seed_row = self.df.loc[seed_idx]
        rec_row = self.df.loc[rec_df_index]

        seed_name = seed_row.get("track_name", "your seed song")
        rec_name = rec_row.get("track_name", "this track")
        rec_artist = rec_row.get("artists", "Unknown Artist")

        return (
            f"Recommended **{rec_name}** by {rec_artist} because its vibe closely "
            f"matches *{seed_name}* on: " + ", ".join(pretty_features) + "."
        )

    def print_now_playing_and_queue(
        self,
        now_playing: pd.Series,
        up_next: pd.DataFrame,
        seed_idx: Optional[int] = None,
        show_explanation: bool = True,
        show_spotify_embed: bool = True,
    ) -> None:
        track_name = now_playing.get("track_name", "Unknown track")
        artist = now_playing.get("artists", "Unknown artist")
        track_id = now_playing.get("track_id")

        print(f"🎧 Now playing: {track_name} — {artist}")

        if track_id:
            url = f"https://open.spotify.com/track/{track_id}"
            print(f"▶ Open in Spotify: {url}")

            if show_spotify_embed:
                display(HTML(f"""
                <iframe style="border-radius:12px" 
                        src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                        width="100%" height="80" frameborder="0" allowfullscreen=""
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture">
                </iframe>
                """))

        if show_explanation and seed_idx is not None:
            try:
                rec_df_index = int(now_playing.name)
                explanation = self.explain_recommendation(
                    seed_idx=seed_idx,
                    rec_df_index=rec_df_index,
                    top_n_features=3,
                )
                print()
                print(explanation)
            except Exception as e:
                print()
                print(f"(Could not generate explanation: {e})")

        if not up_next.empty:
            print("\n🎵 Up Next:")
            cols = []
            for c in ["track_name", "artists", "track_genre", "vibe_score", "vibe_similarity"]:
                if c in up_next.columns:
                    cols.append(c)
            if not cols:
                cols = up_next.columns.tolist()

            display(up_next[cols])
        else:
            print("\n(No additional tracks in queue.)")

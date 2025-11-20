"""
Mode 2A — Seed-from-Song Recommender

User flow:
    1. User types a song or artist name (e.g. "Feel Good Inc", "Bad Bunny").
    2. We search inside the Kaggle track library and show matching candidates.
    3. User picks one seed song.
    4. We use that song's normalized vibe vector as the target in your
       cosine+hybrid recommender (VibeEngine).
    5. We:
        - Autoplay the #1 recommended track (now playing)
        - Use the next rows as a queue
        - Optionally explain WHY the top track was recommended
          (which vibe features it matches).

This module depends on a core engine class (from Mode 1), e.g.:

    from vibe_engine import VibeEngine

The engine must expose:
    - df: pd.DataFrame of tracks
    - X: np.ndarray of normalized vibe features (shape: N x D)
    - vibe_cols: list of feature column names
    - trackid_to_idx: dict mapping track_id -> row index (optional but preferred)
    - score_from_target_vec(target_vec, top_k, lambda_vibe, min_popularity,
                            hide_explicit, allowed_genres, diversity,
                            diversity_threshold) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from vibe_engine import VibeEngine


class Mode2ASeedFromSong:
    """
    Mode 2A: User gives a song they like (from your Kaggle library),
    we use that song's vibe as the seed and recommend similar tracks.

    Responsibilities:
      - Searching for candidate songs by track_name / artist
      - Resolving a chosen seed into an index in engine.df
      - Calling the engine scoring with the seed's vibe vector
      - Returning (now_playing, up_next) and offering explanation helpers
    """

    def __init__(self, engine: VibeEngine):
        self.engine = engine
        self.df = engine.df

    # ------------------------------------------------------------------
    # 1. Search inside the track library
    # ------------------------------------------------------------------
    def search_tracks(
        self,
        query: str,
        max_results: int = 10,
    ) -> pd.DataFrame:
        """
        Simple case-insensitive substring search over track_name and artists.

        Parameters
        ----------
        query : str
            User text (track name, artist, part of either).
        max_results : int
            Max number of rows to return.

        Returns
        -------
        pd.DataFrame
            Subset of self.df with matches. You can display this in a notebook / UI
            to let the user pick a seed.
        """
        if not query:
            return self.df.head(0).copy()

        q = query.lower()

        # Avoid errors if columns are missing (very defensive)
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

    # ------------------------------------------------------------------
    # 2. Resolve seed track (track_id or dataframe index) into df index
    # ------------------------------------------------------------------
    def _resolve_seed_index(
        self,
        track_id: Optional[str] = None,
        df_index: Optional[int] = None,
    ) -> int:
        """
        Resolve which row in self.df is the seed track.

        Priority:
          - If df_index is provided, we trust it.
          - Else if track_id is provided, we try engine.trackid_to_idx.
          - Else raise.

        Returns
        -------
        int
            Index into self.df (and engine.X).
        """
        # Case 1: direct index
        if df_index is not None:
            if df_index not in self.df.index:
                raise ValueError(f"df_index {df_index} is not a valid index in engine.df")
            return int(df_index)

        # Case 2: track_id based
        if track_id is None:
            raise ValueError("Either track_id or df_index must be provided.")

        # Prefer the engine's mapping if available
        if hasattr(self.engine, "trackid_to_idx") and self.engine.trackid_to_idx:
            mapping = self.engine.trackid_to_idx
            if track_id not in mapping:
                raise ValueError(f"track_id {track_id} not found in engine.trackid_to_idx")
            return int(mapping[track_id])

        # Fallback: scan dataframe for a track_id column
        if "track_id" not in self.df.columns:
            raise ValueError(
                "engine.trackid_to_idx is empty and df lacks track_id column; "
                "cannot resolve track_id."
            )

        matches = self.df.index[self.df["track_id"] == track_id].tolist()
        if not matches:
            raise ValueError(f"track_id {track_id} not found in dataframe.")

        return int(matches[0])

    # ------------------------------------------------------------------
    # 3. Recommend from seed
    # ------------------------------------------------------------------
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
        """
        Use a single seed track as the vibe anchor.

        Parameters
        ----------
        track_id : str, optional
            Spotify track_id present in df["track_id"].
        df_index : int, optional
            Direct index (row) in the dataframe; useful if the user clicked
            a search result row.
        top_k : int
            Total number of recommendations you want (including "now playing").
        lambda_vibe : float
            Weight on vibe vs popularity in the hybrid score (0..1).
        min_popularity : int, optional
            Drop tracks with popularity < this (if df has 'popularity').
        hide_explicit : bool
            If True and df has 'explicit', drop explicit tracks.
        allowed_genres : list of str, optional
            If provided and df has 'track_genre', restrict to those.
        diversity : bool
            If True, apply diversity-aware re-ranking inside the engine.
        diversity_threshold : float
            Max cosine similarity allowed between selected tracks.

        Returns
        -------
        now_playing : pd.Series
            The top recommended track row (autoplay candidate).
        up_next : pd.DataFrame
            The remaining recommendation rows (queue).
        seed_idx : int
            The dataframe index of the seed track (for explanations).
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        seed_idx = self._resolve_seed_index(track_id=track_id, df_index=df_index)

        # Seed vector in normalized vibe space
        seed_vec = self.engine.X[seed_idx]

        # Ask engine for slightly more than top_k, because we may drop the seed
        # itself from the result if it appears.
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

        # Drop the seed track from the recommendations if present
        if "track_id" in raw_recs.columns and "track_id" in self.df.columns:
            seed_track_id = self.df.loc[seed_idx, "track_id"]
            recs = raw_recs[raw_recs["track_id"] != seed_track_id]
        else:
            # Fallback: drop by df index if possible
            if seed_idx in raw_recs.index:
                recs = raw_recs.drop(index=seed_idx)
            else:
                recs = raw_recs

        # Make sure we have at least one recommendation
        if recs.empty:
            raise RuntimeError("No recommendations could be generated from this seed.")

        # Trim to desired top_k
        recs = recs.iloc[:top_k]

        # Autoplay candidate is the first row (by hybrid score)
        now_playing = recs.iloc[0]
        up_next = recs.iloc[1:].copy()

        return now_playing, up_next, seed_idx

    # ------------------------------------------------------------------
    # 4. Simple explanation: why was this track recommended?
    # ------------------------------------------------------------------
    def explain_recommendation(
        self,
        seed_idx: int,
        rec_df_index: int,
        top_n_features: int = 3,
    ) -> str:
        """
        Very simple explanation: which vibe dimensions are MOST SIMILAR between
        the seed and the recommended track?

        Parameters
        ----------
        seed_idx : int
            Dataframe index of seed track (from recommend_from_seed return).
        rec_df_index : int
            Dataframe index of the recommended track (e.g. int(now_playing.name)).
        top_n_features : int
            How many key features to mention in the explanation.

        Returns
        -------
        str
            A natural language explanation string you can print or show in UI.
        """
        # Make sure indexes exist
        if seed_idx not in self.df.index:
            raise ValueError(f"seed_idx {seed_idx} not found in df index.")
        if rec_df_index not in self.df.index:
            raise ValueError(f"rec_df_index {rec_df_index} not found in df index.")

        seed_vec = self.engine.X[self.df.index.get_loc(seed_idx)]
        rec_vec = self.engine.X[self.df.index.get_loc(rec_df_index)]

        # Absolute difference in normalized space
        diff = np.abs(seed_vec - rec_vec)

        # Smaller diff = more similar on that vibe dimension
        feature_diffs = list(zip(self.engine.vibe_cols, diff))
        feature_diffs.sort(key=lambda x: x[1])

        top_features = [name for name, _ in feature_diffs[:top_n_features]]

        # Friendly renaming if you want to be extra cute
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

    # ------------------------------------------------------------------
    # 5. Convenience: pretty printing in a notebook
    # ------------------------------------------------------------------
    # def print_now_playing_and_queue(
    #     self,
    #     now_playing: pd.Series,
    #     up_next: pd.DataFrame,
    #     seed_idx: Optional[int] = None,
    #     show_explanation: bool = True,
    # ) -> None:
    #     """
    #     Convenience helper for Jupyter demos.

    #     Prints:
    #       - a "Now playing" line
    #       - optional explanation for the top track (if seed_idx provided)
    #       - a tidy "Up Next" table
    #     """

    #     track_name = now_playing.get("track_name", "Unknown track")
    #     artist = now_playing.get("artists", "Unknown artist")

    #     print(f"🎧 Now playing: {track_name} — {artist}")
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
            # Choose a nice subset of columns if available
            cols = []
            for c in ["track_name", "artists", "track_genre", "vibe_score", "vibe_similarity"]:
                if c in up_next.columns:
                    cols.append(c)
            if not cols:
                cols = up_next.columns.tolist()

            display(up_next[cols])
        else:
            print("\n(No additional tracks in queue.)")

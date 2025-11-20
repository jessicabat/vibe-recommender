import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# -----------------------
# Config: vibe dimensions
# -----------------------

VIBE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
]

# Optional: fallback weights if you want some vibes to matter more
DEFAULT_WEIGHTS = {
    "danceability": 1.0,
    "energy": 1.2,
    "valence": 1.2,
    "acousticness": 1.0,
    "instrumentalness": 1.0,
    "speechiness": 0.8,
}

# -----------------------
# Persona sliders (0–100)
# -----------------------

PERSONAS = {
    "night_drive": {
        "danceability": 70,
        "energy": 80,
        "valence": 40,
        "acousticness": 20,
        "instrumentalness": 30,
        "speechiness": 20,
    },
    "sad_girl_autumn": {
        "danceability": 35,
        "energy": 30,
        "valence": 20,
        "acousticness": 70,
        "instrumentalness": 40,
        "speechiness": 25,
    },
    "deep_focus": {
        "danceability": 40,
        "energy": 35,
        "valence": 45,
        "acousticness": 60,
        "instrumentalness": 70,
        "speechiness": 10,
    },
    "happy_pop": {
        "danceability": 80,
        "energy": 75,
        "valence": 85,
        "acousticness": 30,
        "instrumentalness": 10,
        "speechiness": 20,
    },
}


class VibeRecommender:
    """
    Core recommender operating purely on content (audio features) from the Kaggle dataset.
    - Mode 1: recommend_by_sliders(...)
    - Mode 2: surprise_me(...), which can use seed tracks or a persona.
    """

    def __init__(self, df_tracks: pd.DataFrame, vibe_cols=None, weight_dict=None):
        # Keep a clean copy of the dataframe, reset index for stable indexing
        self.df = df_tracks.reset_index(drop=True)

        # Which columns define "vibe" space?
        self.vibe_cols = vibe_cols if vibe_cols is not None else VIBE_COLS

        # Optional per-dimension weights
        self.weights = weight_dict if weight_dict is not None else DEFAULT_WEIGHTS

        # Make sure all vibe columns exist
        for c in self.vibe_cols:
            if c not in self.df.columns:
                raise ValueError(f"Column {c} not found in dataframe")

        # Store original numeric matrix
        X_raw = self.df[self.vibe_cols].values.astype(float)

        # Standardize to mean 0, std 1
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_raw)  # shape: (N_tracks, D)

        # Precompute norm for cosine sim if needed
        self.X_norms = np.linalg.norm(self.X, axis=1) + 1e-8

        # Keep track_id lookup for convenience (if present)
        if "track_id" in self.df.columns:
            self.trackid_to_idx = {
                tid: idx for idx, tid in enumerate(self.df["track_id"].values)
            }
        else:
            self.trackid_to_idx = {}

    # ------------- internal helpers -------------

    def _sliders_to_target_vec(self, sliders: dict) -> np.ndarray:
        """
        sliders: dict with keys in self.vibe_cols, values in [0, 100]
        → convert to 0–1, then standardize into vibe space.
        """
        target_raw = []
        for col in self.vibe_cols:
            # default: mid if not provided
            v = sliders.get(col, 50)
            v01 = max(0.0, min(100.0, float(v))) / 100.0
            target_raw.append(v01)

        target_raw = np.array(target_raw).reshape(1, -1)
        target_scaled = self.scaler.transform(target_raw)[0]  # 1D vector
        return target_scaled

    def _distance_scores(self, target_vec: np.ndarray) -> np.ndarray:
        """
        Compute weighted squared distance between each track and target_vec.
        Return distances as a 1D array (lower = better).
        """
        # Broadcast subtraction: (N, D)
        diff = self.X - target_vec  # shape (N, D)

        # Apply weights per dimension
        w = np.array([self.weights[c] for c in self.vibe_cols])
        # Weighted squared distance
        d2 = np.sum(w * (diff ** 2), axis=1)  # shape (N,)
        return d2

    def _score_tracks(
        self,
        target_vec: np.ndarray,
        top_k: int = 20,
        lambda_vibe: float = 0.7,
        min_popularity: int = None,
        hide_explicit: bool = False,
        allowed_genres=None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """
        Core ranking function used by both modes.
        - target_vec: point in vibe space
        - lambda_vibe: weight on vibe vs popularity
        - min_popularity: drop tracks with popularity < this (if not None)
        - hide_explicit: drop explicit tracks if True
        - allowed_genres: list of genres or None
        - diversity: if True, greedily ensure selected tracks aren't too similar
        """

        N = self.X.shape[0]

        # 1) Compute vibe distance and convert to similarity
        d2 = self._distance_scores(target_vec)  # lower = better
        # Convert to similarity in (0,1], higher is better
        # tau controls how fast similarity decays with distance
        tau = np.median(d2) + 1e-8
        vibe_sim = np.exp(-d2 / (tau + 1e-8))

        # 2) Popularity normalization
        if "popularity" in self.df.columns:
            pop = self.df["popularity"].values.astype(float)
            pop_norm = pop / 100.0
        else:
            pop_norm = np.zeros(N)

        # 3) Combine
        score = lambda_vibe * vibe_sim + (1.0 - lambda_vibe) * pop_norm

        # 4) Filters: explicit, popularity, genre
        candidate_idx = np.arange(N)

        if hide_explicit and "explicit" in self.df.columns:
            mask = ~self.df["explicit"].astype(bool).values
            candidate_idx = candidate_idx[mask]
            score = score[mask]

        if min_popularity is not None and "popularity" in self.df.columns:
            mask = self.df["popularity"].values[candidate_idx] >= min_popularity
            candidate_idx = candidate_idx[mask]
            score = score[mask]

        if allowed_genres is not None and "track_genre" in self.df.columns:
            allowed_set = set(allowed_genres)
            genres = self.df["track_genre"].values[candidate_idx]
            mask = np.array([g in allowed_set for g in genres])
            candidate_idx = candidate_idx[mask]
            score = score[mask]

        if len(candidate_idx) == 0:
            # no candidates after filtering
            return self.df.head(0).copy()

        # 5) Sort by score
        order = np.argsort(-score)  # descending
        candidate_idx = candidate_idx[order]
        score_sorted = score[order]

        # 6) Diversity-aware greedy selection
        if diversity:
            selected = []
            selected_feats = []

            for idx in candidate_idx:
                x = self.X[idx]
                if not selected:
                    selected.append(idx)
                    selected_feats.append(x)
                else:
                    # compute max cosine sim with already selected
                    sims = []
                    for xf in selected_feats:
                        num = np.dot(x, xf)
                        den = (np.linalg.norm(x) * np.linalg.norm(xf)) + 1e-8
                        sims.append(num / den)
                    max_sim = max(sims)

                    if max_sim < diversity_threshold:
                        selected.append(idx)
                        selected_feats.append(x)

                if len(selected) >= top_k:
                    break

            final_idx = selected
        else:
            final_idx = candidate_idx[:top_k].tolist()

        result = self.df.iloc[final_idx].copy()
        result["vibe_score"] = score_sorted[: len(final_idx)]
        return result

    # ------------- public API -------------

    # MODE 1: Sliders
    def recommend_by_sliders(
        self,
        sliders: dict,
        top_k: int = 20,
        lambda_vibe: float = 0.7,
        min_popularity: int = None,
        hide_explicit: bool = False,
        allowed_genres=None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """
        Mode 1: user explicitly sets sliders.
        sliders: dict of { "danceability": 0–100, "energy": 0–100, ... }
        Returns a dataframe of top_k recommended tracks.
        """
        target_vec = self._sliders_to_target_vec(sliders)
        recs = self._score_tracks(
            target_vec,
            top_k=top_k,
            lambda_vibe=lambda_vibe,
            min_popularity=min_popularity,
            hide_explicit=hide_explicit,
            allowed_genres=allowed_genres,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )
        return recs

    # MODE 2 helper: build user profile from seeds
    def build_user_profile(self, seed_track_ids):
        """
        seed_track_ids: list of track_id values from the dataframe.
        Returns a user embedding vector in vibe space, or None if no matches.
        """
        if not self.trackid_to_idx:
            raise ValueError("track_id column not available / mapping not built.")

        idxs = []
        for tid in seed_track_ids:
            if tid in self.trackid_to_idx:
                idxs.append(self.trackid_to_idx[tid])

        if len(idxs) == 0:
            return None

        user_vec = self.X[idxs].mean(axis=0)
        return user_vec

    def recommend_for_profile(
        self,
        user_vec: np.ndarray,
        top_k: int = 20,
        lambda_vibe: float = 0.7,
        shrink_to_global: float = 0.7,
        min_popularity: int = None,
        hide_explicit: bool = False,
        allowed_genres=None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """
        Recommend for a given user embedding (from seeds).
        shrink_to_global: α in [0,1] — how much to trust user vector vs global mean.
        """
        global_mean = self.X.mean(axis=0)
        alpha = shrink_to_global
        target_vec = (1 - alpha) * global_mean + alpha * user_vec

        recs = self._score_tracks(
            target_vec,
            top_k=top_k,
            lambda_vibe=lambda_vibe,
            min_popularity=min_popularity,
            hide_explicit=hide_explicit,
            allowed_genres=allowed_genres,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )
        return recs

    # MODE 2: Surprise me
    def surprise_me(
        self,
        seed_track_ids=None,
        persona_name=None,
        top_k: int = 20,
        lambda_vibe: float = 0.7,
        min_popularity: int = None,
        hide_explicit: bool = False,
        allowed_genres=None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """
        Mode 2:
        - If seed_track_ids provided: build user profile and recommend.
        - Else if persona_name provided: use that persona's sliders.
        - Else: randomly pick a persona.
        """
        # Case 1: user-based profile from seeds
        if seed_track_ids:
            user_vec = self.build_user_profile(seed_track_ids)
            if user_vec is not None:
                return self.recommend_for_profile(
                    user_vec,
                    top_k=top_k,
                    lambda_vibe=lambda_vibe,
                    shrink_to_global=0.7,
                    min_popularity=min_popularity,
                    hide_explicit=hide_explicit,
                    allowed_genres=allowed_genres,
                    diversity=diversity,
                    diversity_threshold=diversity_threshold,
                )

        # Case 2: persona-based
        if persona_name is None:
            persona_name = np.random.choice(list(PERSONAS.keys()))
        sliders = PERSONAS[persona_name]

        target_vec = self._sliders_to_target_vec(sliders)
        recs = self._score_tracks(
            target_vec,
            top_k=top_k,
            lambda_vibe=lambda_vibe,
            min_popularity=min_popularity,
            hide_explicit=hide_explicit,
            allowed_genres=allowed_genres,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )
        recs["persona_used"] = persona_name
        return recs
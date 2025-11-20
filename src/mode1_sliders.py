import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------
# Config: vibe dimensions
# -----------------------

VIBE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",           # NEW: tempo is now part of the vibe space
    "acousticness",
    "instrumentalness",
    "speechiness",
]

# Optional: per-dimension weights for weighted cosine
DEFAULT_WEIGHTS = {
    "danceability": 1.0,
    "energy": 1.2,
    "valence": 1.2,
    "tempo": 1.0,
    "acousticness": 1.0,
    "instrumentalness": 1.0,
    "speechiness": 0.8,
}


class SliderVibeRecommender:
    """
    Mode 1: Slider-based vibe recommender.

    - Uses 7 vibe features:
      [danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness]
    - Core similarity: weighted cosine similarity in standardized feature space.
    - Hybrid scoring: lambda_vibe * vibe_similarity + (1 - lambda_vibe) * popularity.
    - Supports filters: explicit, min_popularity, allowed_genres.
    - Optional diversity-aware re-ranking to avoid near-duplicates.
    """

    def __init__(self, df_tracks: pd.DataFrame,
                 vibe_cols=None,
                 weight_dict=None):
        # Clean copy of the dataframe, stable integer index
        self.df = df_tracks.reset_index(drop=True)

        # Which columns define the vibe space?
        self.vibe_cols = vibe_cols if vibe_cols is not None else VIBE_COLS

        # Per-dimension weights
        self.weight_dict = weight_dict if weight_dict is not None else DEFAULT_WEIGHTS
        self.weight_vec = np.array([self.weight_dict[c] for c in self.vibe_cols])
        self.weight_sqrt = np.sqrt(self.weight_vec)

        # Sanity check: all vibe columns exist
        for c in self.vibe_cols:
            if c not in self.df.columns:
                raise ValueError(f"Column {c} not found in dataframe")

        # Store raw matrix (N, D)
        X_raw = self.df[self.vibe_cols].values.astype(float)

        # Standardize to mean 0, std 1 (handles tempo scale vs [0,1] features)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_raw)  # shape: (N_tracks, D)

        # Precompute weighted representation for cosine similarity
        # Weighted cosine is equivalent to scaling each dim by sqrt(weight)
        self.X_w = self.X * self.weight_sqrt  # shape: (N, D)
        self.X_w_norms = np.linalg.norm(self.X_w, axis=1) + 1e-8

        # Track_id lookup (not strictly needed for Mode 1,
        # but kept in case you want to reuse later)
        if "track_id" in self.df.columns:
            self.trackid_to_idx = {
                tid: idx for idx, tid in enumerate(self.df["track_id"].values)
            }
        else:
            self.trackid_to_idx = {}

        # Precompute tempo range for slider mapping
        if "tempo" in self.vibe_cols:
            tempo_vals = self.df["tempo"].values.astype(float)
            self.tempo_min = float(np.min(tempo_vals))
            self.tempo_max = float(np.max(tempo_vals))
        else:
            self.tempo_min = None
            self.tempo_max = None

    # ------------- internal helpers -------------

    def _sliders_to_target_vec(self, sliders: dict) -> np.ndarray:
        """
        sliders: dict with keys in self.vibe_cols, values in [0, 100].

        For non-tempo features:
            - interpret slider as 0–1 directly (Spotify audio features are already 0–1).
        For tempo:
            - map slider 0–100 to [tempo_min, tempo_max] from the dataset.

        Then apply the same StandardScaler to get a target vector in standardized vibe space.
        """
        target_raw = []

        for col in self.vibe_cols:
            s = sliders.get(col, 50)  # default mid-point
            s = max(0.0, min(100.0, float(s)))

            if col == "tempo" and self.tempo_min is not None and self.tempo_max is not None:
                # Linear interpolation across the dataset's tempo range
                tempo_val = self.tempo_min + (s / 100.0) * (self.tempo_max - self.tempo_min)
                target_raw.append(tempo_val)
            else:
                # Other Spotify features are in [0,1], so map slider 0–100 → 0–1
                v01 = s / 100.0
                target_raw.append(v01)

        target_raw = np.array(target_raw).reshape(1, -1)          # shape (1, D)
        target_scaled = self.scaler.transform(target_raw)[0]      # 1D vector (D,)
        return target_scaled

    def _vibe_cosine_similarity(self, target_vec: np.ndarray) -> np.ndarray:
        """
        Compute weighted cosine similarity between each track and target_vec.

        Steps:
        - Apply sqrt(weights) to both X and target.
        - Compute cosine similarity.
        Returns an array of similarities in [-1, 1].
        """
        target_w = target_vec * self.weight_sqrt            # (D,)
        target_norm = np.linalg.norm(target_w) + 1e-8
        dots = self.X_w @ target_w                          # (N,)
        sims = dots / (self.X_w_norms * target_norm)        # (N,)
        return sims

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
        Core ranking function for Mode 1 (sliders).

        - target_vec: point in standardized vibe space
        - lambda_vibe: weight on vibe similarity vs. popularity
        - min_popularity: drop tracks with popularity < this (if not None)
        - hide_explicit: drop explicit tracks if True
        - allowed_genres: list of allowed genres or None
        - diversity: if True, greedily enforce diversity using cosine similarity
        - diversity_threshold: max allowed cosine sim between any two recommended tracks
        """

        N = self.X.shape[0]

        # 1) Vibe similarity (weighted cosine)
        sims = self._vibe_cosine_similarity(target_vec)  # in [-1, 1]

        # Map to [0,1] so it blends nicely with popularity
        vibe_sim = (sims + 1.0) / 2.0  # now in [0, 1]

        # 2) Popularity normalization
        if "popularity" in self.df.columns:
            pop = self.df["popularity"].values.astype(float)
            pop_norm = pop / 100.0
        else:
            pop_norm = np.zeros(N)

        # 3) Hybrid score: lambda_vibe * vibe_sim + (1 - lambda_vibe) * popularity
        score_full = lambda_vibe * vibe_sim + (1.0 - lambda_vibe) * pop_norm

        # 4) Filters: explicit, popularity, genre
        candidate_idx = np.arange(N)

        # Explicit filter
        if hide_explicit and "explicit" in self.df.columns:
            explicit_mask = ~self.df["explicit"].astype(bool).values
            candidate_idx = candidate_idx[explicit_mask]

        # Popularity threshold
        if min_popularity is not None and "popularity" in self.df.columns:
            pop_mask = self.df["popularity"].values[candidate_idx] >= min_popularity
            candidate_idx = candidate_idx[pop_mask]

        # Genre filter
        if allowed_genres is not None and "track_genre" in self.df.columns:
            allowed_set = set(allowed_genres)
            genres = self.df["track_genre"].values[candidate_idx]
            genre_mask = np.array([g in allowed_set for g in genres])
            candidate_idx = candidate_idx[genre_mask]

        if len(candidate_idx) == 0:
            # no candidates after filtering
            return self.df.head(0).copy()

        # 5) Sort candidates by score (descending)
        candidate_scores = score_full[candidate_idx]
        order = np.argsort(-candidate_scores)
        candidate_idx = candidate_idx[order]

        # 6) Diversity-aware greedy selection (using weighted cosine)
        if diversity:
            selected = []

            for idx in candidate_idx:
                if not selected:
                    selected.append(idx)
                else:
                    # compute max cosine sim with already selected tracks (in weighted space)
                    sims_to_selected = []
                    for s_idx in selected:
                        num = np.dot(self.X_w[idx], self.X_w[s_idx])
                        den = (self.X_w_norms[idx] * self.X_w_norms[s_idx]) + 1e-8
                        sims_to_selected.append(num / den)

                    max_sim = max(sims_to_selected)
                    if max_sim < diversity_threshold:
                        selected.append(idx)

                if len(selected) >= top_k:
                    break

            final_idx = np.array(selected, dtype=int)
        else:
            final_idx = candidate_idx[:top_k]

        # 7) Build result dataframe with aligned scores
        result = self.df.iloc[final_idx].copy()
        result["vibe_score"] = score_full[final_idx]
        result["vibe_similarity"] = vibe_sim[final_idx]
        result["popularity_norm"] = pop_norm[final_idx]

        return result

    # ------------- public API -------------

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

        sliders: dict like
            {
                "danceability": 0–100,
                "energy": 0–100,
                "valence": 0–100,
                "tempo": 0–100,
                "acousticness": 0–100,
                "instrumentalness": 0–100,
                "speechiness": 0–100,
            }

        Returns a dataframe of top_k recommended tracks, including:
            - vibe_score (hybrid score)
            - vibe_similarity (pure content-based similarity, 0–1)
            - popularity_norm (0–1 popularity signal)
        """
        target_vec = self._sliders_to_target_vec(sliders)
        recs = self._score_tracks(
            target_vec=target_vec,
            top_k=top_k,
            lambda_vibe=lambda_vibe,
            min_popularity=min_popularity,
            hide_explicit=hide_explicit,
            allowed_genres=allowed_genres,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )
        return recs

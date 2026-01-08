# Vibe Recommender 🎧

A content-based music recommender that matches users on **vibe**, not just genre or collaborative signals.

The system lives in a 7D audio feature space and supports three interaction modes:

1. **Dial in a vibe** – user steers via 7 sliders (Mode 1)  
2. **Start from a song** – seed from a track in the library (Mode 2A)  
3. **Vibe Roulette** – time-of-day persona spin with controlled exploration (Mode 2B)

---

## 🔴 Live demo

You can try the Vibe Recommender in your browser here:

➡️ **Live app:** https://vibe-recommender.streamlit.app/  
➡️ **Project website (story + visuals):** https://jessicabat.github.io/vibe-recommender/

If you prefer running it locally, please scroll down to the Running locally section for instructions.

---

## 📦 Data & Features

- Source: Spotify-style audio features from a ~114k track dataset (Kaggle).
- Each track is represented as a 7D vector:

  - `danceability`
  - `energy`
  - `valence` (positivity)
  - `tempo`
  - `acousticness`
  - `instrumentalness`
  - `speechiness`

- Preprocessing:
  - `StandardScaler` over all 7 features.
  - Per-dimension weights (e.g. slightly higher for `energy`, `valence`).

---

## 🧠 Core model: VibeEngine

Implemented in `vibe_engine.py`.

### Similarity

- Represent each track as a standardized vector **x ∈ ℝ⁷**.
- Apply per-feature weights via elementwise scaling.
- Use **weighted cosine similarity**:

$$
\text{sim}(x, y)
= \frac{(w^{1/2} \odot x) \cdot (w^{1/2} \odot y)}
{\lVert w^{1/2} \odot x \rVert \, \lVert w^{1/2} \odot y \rVert}
$$

- Map similarity from [-1, 1] → [0, 1] so it can be blended with popularity.

### Scoring

- **Hybrid ranking**:

$$
score = \lambda \cdot vibe_{sim} + (1 - \lambda) \cdot popularity_{norm}
$$

- `vibe_sim` = weighted cosine similarity with the target vibe.
- `popularity_norm` = track popularity / 100.
- λ ∈ [0,1] is user-tunable in the UI (“vibe vs popularity”).

### Diversity

- Greedy diversity-aware selection:
  - Sort candidates by `score`.
  - Iterate through in order; only accept a track if its cosine similarity to all selected tracks is below a threshold (e.g. 0.90).
  - This avoids 10 near-duplicates in a row while staying in the same region of vibe space.

---

## 🎛 Mode 1 – Dial in a vibe (sliders)

File: `mode1_sliders.py` (UI) + `vibe_engine.py` (core).

- User controls:

  - `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness` on 0–100 sliders.
  - “Vibe vs popularity” slider for λ.
  - Playlist style: tightly focused / balanced / exploratory (controls diversity threshold).

- Mapping:
  - Non-tempo features: slider 0–100 → [0,1].
  - Tempo: slider 0–100 → [min_tempo, max_tempo] learned from dataset.
  - Sliders → raw feature vector → standardized using the same `StandardScaler` as the library.

- Engine call:
  - `engine.recommend_by_sliders(sliders, top_k=10, ...)`
  - Returns a DataFrame with `vibe_score`, `vibe_similarity`, `popularity_norm`.

---

## 🎵 Mode 2A – Start from a song (seed-based)

File: `mode2a_seed_from_song.py`.

- Flow:

  1. User searches the library by track name / artist.
  2. Selects a seed track.
  3. The seed’s standardized vector becomes the **target** in vibe space.
  4. Engine runs the same cosine + hybrid ranking to find nearest neighbors.
  5. Top track is “Now playing”; the rest form the queue.

- Implementation details:

  - `Mode2ASeedFromSong.search_tracks(query)` – simple substring match over `track_name`, `artists`.
  - `Mode2ASeedFromSong.recommend_from_seed(...)`:
    - Resolve seed row (`track_id` or df index).
    - Use `engine.X[seed_idx]` as `target_vec`.
    - Ask engine for `top_k + 1` tracks, then drop the seed itself if present.
  - Explanation helper:
    - `explain_recommendation(seed_idx, rec_df_index, top_n_features=3)`:
      - Compare normalized vectors for seed and rec.
      - Find features with smallest absolute difference.
      - Return a short human-readable explanation (e.g. “instrumental vibe, acoustic feel, positivity”).

---

## 🎲 Mode 2B – Vibe Roulette (time-of-day personas)

File: `mode2b_vibe_roulette.py`.

- No user input beyond a single “spin” button.
- Context:

  - `weekday` vs `weekend` via `datetime.weekday()`.
  - Coarse time-of-day buckets: `morning`, `afternoon`, `evening`, `late_night`.

- Persona design:

  - Each `(day_type, time_bucket)` maps to one or more personas, e.g.:

    - Weekday morning → **Soft Sunrise Focus**
    - Weekday afternoon → **Flow State Focus**
    - Weekend evening → **Night Out Pre-Game**
    - Weekend late night → **Neon City Ride** / **Midnight Lo-Fi Drift**

  - Each persona has:
    - A 7D slider profile (0–100) tuned using dataset summary stats.
    - A short tagline and tag set for UX copy (used in Streamlit).

- Recommendation + exploration:

  1. Persona sliders → engine.recommend_by_sliders(...) with a relatively large `top_k` (candidate pool).
  2. Consider the top `explore_k` candidates.
  3. Sample “Now playing” via a temperature-controlled distribution over scores
     (higher temperature → more exploration).
  4. Fill the queue with the remaining top-ranked tracks.

---

## 🖥 App & architecture

- Core logic:
  - `src/vibe_engine.py` – shared engine (similarity, scoring, diversity).
  - `src/mode1_sliders.py` – slider-based recommender wrapper.
  - `src/mode2a_seed_from_song.py` – seed-from-song wrapper.
  - `src/mode2b_vibe_roulette.py` – time-of-day persona wrapper.
- UI:
  - `src/app_streamlit.py` – Streamlit app with:
    - Central “mode picker” hero (3 mode cards).
    - Shared playlist/player component:
      - “Now Playing” card with Spotify embed.
      - “Previous” and “Skip” controls.
      - Inline “Play” buttons for each track in the queue.

---

## 🚀 Running locally

```bash
git clone https://github.com/jessicabat/vibe-recommender.git
cd vibe-recommender

# Create and activate a virtualenv / conda env (recommended)
pip install -r requirements.txt

# Make sure data/spotify_tracks.csv exists:
#   - columns should include the 7 vibe features, popularity, track_id, artists, track_name, track_genre, explicit

streamlit run src/app_streamlit.py
# Then open the URL shown in your terminal

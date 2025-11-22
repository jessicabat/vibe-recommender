import datetime
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from vibe_engine import VibeEngine
from mode2a_seed_from_song import Mode2ASeedFromSong
from mode2b_vibe_roulette import Mode2BVibeRoulette


# =========================
# 0. CONFIG
# =========================

DATA_PATH = "data/spotify_tracks.csv"


# =========================
# 1. CACHED LOADERS
# =========================

@st.cache_data
def load_tracks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_engine_and_modes(path: str):
    df = load_tracks(path)
    engine = VibeEngine(df)
    mode2a = Mode2ASeedFromSong(engine)
    mode2b = Mode2BVibeRoulette(engine)
    return engine, mode2a, mode2b


# =========================
# 2. UI HELPERS
# =========================

def render_spotify_embed(track_id: Optional[str]) -> None:
    """Render a Spotify player if track_id is present."""
    if not track_id or pd.isna(track_id):
        return
    track_url = f"https://open.spotify.com/track/{track_id}"
    st.markdown(f"[Open in Spotify]({track_url})")

    # Small embedded player
    embed_html = f"""
    <iframe style="border-radius:12px"
            src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
            width="100%" height="80" frameborder="0"
            allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture">
    </iframe>
    """
    st.markdown(embed_html, unsafe_allow_html=True)


def render_now_playing_card(
    row: pd.Series,
    title: str = "Now playing",
    explanation: Optional[str] = None,
) -> None:
    """Nice little card for the top track."""
    track_name = row.get("track_name", "Unknown track")
    artist = row.get("artists", "Unknown artist")
    genre = row.get("track_genre", None)
    vibe_score = row.get("vibe_score", None)
    vibe_sim = row.get("vibe_similarity", None)
    pop_norm = row.get("popularity_norm", None)
    track_id = row.get("track_id", None)

    st.subheader(f"🎧 {title}")
    st.markdown(f"**{track_name}** — {artist}")
    if genre:
        st.caption(f"Genre: {genre}")

    # Small metrics row
    cols = st.columns(3)
    if vibe_sim is not None:
        cols[0].metric("Vibe similarity", f"{vibe_sim:.2f}")
    if pop_norm is not None:
        cols[1].metric("Popularity (0–1)", f"{pop_norm:.2f}")
    if vibe_score is not None:
        cols[2].metric("Hybrid score", f"{vibe_score:.2f}")

    # Spotify link + player
    render_spotify_embed(track_id)

    if explanation:
        st.markdown("")
        st.info(explanation)


def render_up_next_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.write("No tracks in queue.")
        return

    st.subheader("🎵 Up Next")

    cols = [c for c in ["track_name", "artists", "track_genre",
                        "vibe_score", "vibe_similarity"] if c in df.columns]
    if not cols:
        cols = df.columns.tolist()

    st.dataframe(df[cols].reset_index(drop=True))


# =========================
# 3. MODE 1: SLIDERS
# =========================

def page_mode1_sliders(engine: VibeEngine):
    st.markdown("### Mode 1 — *I know my vibe*")
    st.write(
        "Drag the sliders to describe your current mood, and I'll dig through "
        "114k tracks to match that vibe."
    )

    with st.form("mode1_form"):
        col1, col2 = st.columns(2)

        with col1:
            dance = st.slider("Danceability 💃", 0, 100, 80)
            energy = st.slider("Energy ⚡", 0, 100, 75)
            valence = st.slider("Positivity (Valence) 😃", 0, 100, 70)
            tempo = st.slider("Tempo 🥁 (slow ←→ fast)", 0, 100, 65)

        with col2:
            acoustic = st.slider("Acousticness 🎸", 0, 100, 20)
            instr = st.slider("Instrumentalness 🎼", 0, 100, 10)
            speech = st.slider("Speechiness 🗣️", 0, 100, 15)

        st.markdown("---")
        st.markdown("**Fine-tune the recommendation behavior**")

        hide_explicit = st.checkbox("Hide explicit tracks", value=True)

        lambda_vibe = st.slider(
            "Vibe vs Popularity",
            0.0, 1.0, 0.8, 0.05,
            help="0 = just popularity, 1 = pure vibe match",
        )

        diversity_mode = st.radio(
            "Playlist style",
            options=["Tightly focused", "Balanced", "Exploratory"],
            index=1,
            horizontal=True,
        )

        if diversity_mode == "Tightly focused":
            diversity_threshold = 0.95
        elif diversity_mode == "Balanced":
            diversity_threshold = 0.90
        else:
            diversity_threshold = 0.85

        submitted = st.form_submit_button("✨ Generate my vibe playlist")

    if not submitted:
        return

    sliders = {
        "danceability": dance,
        "energy": energy,
        "valence": valence,
        "tempo": tempo,
        "acousticness": acoustic,
        "instrumentalness": instr,
        "speechiness": speech,
    }

    with st.spinner("Finding tracks that match your vibe..."):
        recs = engine.recommend_by_sliders(
            sliders=sliders,
            top_k=10,
            lambda_vibe=lambda_vibe,
            hide_explicit=hide_explicit,
            diversity=True,
            diversity_threshold=diversity_threshold,
        )

    if recs.empty:
        st.error("No recommendations found. Try relaxing some filters.")
        return

    now_playing = recs.iloc[0]
    up_next = recs.iloc[1:].copy()

    # Optional explanation based on sliders
    explanation = (
        "This track is close to the vibe you dialed in — high on "
        f"{'danceability' if dance > 70 else ''}"
    )

    render_now_playing_card(now_playing, title="Now playing", explanation=None)
    render_up_next_table(up_next)


# =========================
# 4. MODE 2A: SEED FROM SONG
# =========================

def page_mode2a_seed_from_song(mode2a: Mode2ASeedFromSong):
    st.markdown("### Mode 2A — *Start from a song*")
    st.write(
        "Tell me a song you like. I’ll search inside the library, "
        "you pick one, and I’ll build a vibe-matching playlist from it."
    )

    # Keep search results in session so they survive reruns
    if "mode2a_results" not in st.session_state:
        st.session_state["mode2a_results"] = None

    query = st.text_input("Type a song or artist name", value="Feel Good Inc")

    if st.button("🔍 Search library"):
        if not query.strip():
            st.warning("Please enter a song or artist name.")
        else:
            results = mode2a.search_tracks(query=query, max_results=15)
            st.session_state["mode2a_results"] = results

    results = st.session_state["mode2a_results"]

    # If no results yet, stop here
    if results is None or results.empty:
        if query and results is not None and results.empty:
            st.error("No matches found. Try a different spelling or song.")
        return

    st.markdown("#### Search results")

    display_df = results[["track_name", "artists", "track_genre", "track_id"]].copy()
    st.dataframe(
        display_df.reset_index().rename(columns={"index": "df_index"})
    )

    # Let the user pick a row by dataframe index
    idx_list = results.index.tolist()
    chosen_idx = st.selectbox(
        "Pick a seed track",
        options=idx_list,
        format_func=lambda i: f"{results.loc[i, 'track_name']} — {results.loc[i, 'artists']}",
    )

    if st.button("🎧 Use this as my seed"):
        _run_seed_flow(mode2a, df_index=int(chosen_idx))



# =========================
# 5. MODE 2B: VIBE ROULETTE
# =========================

def page_mode2b_vibe_roulette(mode2b: Mode2BVibeRoulette):
    st.markdown("### Mode 2B — *Vibe Roulette*")
    st.write(
        "One click. I look at the current time, pick a vibe persona "
        "for you, and spin up a playlist."
    )

    # Exploration toggle → maps to explore_k & temperature inside spin()
    exploration_choice = st.radio(
        "How adventurous should I be?",
        options=["Predictable", "Balanced", "Chaotic good"],
        index=1,
        horizontal=True,
    )

    if exploration_choice == "Predictable":
        explore_k = 5
        temperature = 0.3
    elif exploration_choice == "Balanced":
        explore_k = 15
        temperature = 0.7
    else:
        explore_k = 30
        temperature = 1.2

    if st.button("🎰 Spin the Vibe Roulette"):
        with st.spinner("Spinning up your vibe..."):
            now_playing, up_next, meta = mode2b.spin(
                top_k=10,
                hide_explicit=True,
                explore_k=explore_k,
                temperature=temperature,
            )

        persona = meta.get("persona_name", "Unknown persona")
        time_bucket = meta.get("time_bucket", "")
        weekday_bucket = meta.get("weekday_bucket", "")
        sliders = meta.get("sliders_used", {})

        # Story block
        now = datetime.datetime.now()
        st.markdown("---")
        st.markdown("#### Your vibe spin")
        st.markdown(
            f"🕒 It’s **{now.strftime('%A %I:%M %p')}** — "
            f"I’m sensing a **{persona}** mood "
            f"({weekday_bucket}, {time_bucket.lower()})."
        )

        # Optional: show the slider profile for nerdy users
        with st.expander("See the persona vibe profile I used"):
            st.json(sliders)

        render_now_playing_card(
            now_playing,
            title="Now playing",
            explanation=f"This track sits right in the pocket of the **{persona}** vibe.",
        )
        render_up_next_table(up_next)


# =========================
# 6. MAIN APP
# =========================

def main():
    st.set_page_config(
        page_title="Vibe Recommender",
        page_icon="🎶",
        layout="wide",
    )

    st.title("Vibe Recommender 🎧")
    st.caption(
        "A content-based music recommender that matches you on *vibe* — "
        "not just genre or popularity."
    )

    # Load engine + mode helpers
    try:
        engine, mode2a, mode2b = load_engine_and_modes(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find data file at `{DATA_PATH}`. Please update DATA_PATH.")
        return

    # Top-level mode selector
    mode = st.sidebar.radio(
        "Choose a mode",
        options=[
            "Mode 1 — I know my vibe",
            "Mode 2A — Start from a song",
            "Mode 2B — Vibe Roulette",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with cosine similarity, hybrid scoring, and personas.")

    if mode.startswith("Mode 1"):
        page_mode1_sliders(engine)
    elif mode.startswith("Mode 2A"):
        page_mode2a_seed_from_song(mode2a)
    else:
        page_mode2b_vibe_roulette(mode2b)


if __name__ == "__main__":
    main()

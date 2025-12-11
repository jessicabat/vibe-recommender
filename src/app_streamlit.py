import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st

from vibe_engine_full import VibeEngine
from mode2a_seed_from_song_full import Mode2ASeedFromSong
from mode2b_vibe_roulette_full import Mode2BVibeRoulette, TimeOfDayPersona


DATA_PATH = "data/spotify_tracks.csv"

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


# UI HELPERS

def render_spotify_embed(track_id: Optional[str]) -> None:
    """Render a Spotify player if track_id is present."""
    if not track_id or pd.isna(track_id):
        return
    track_url = f"https://open.spotify.com/track/{track_id}"
    st.markdown(f"[Open in Spotify]({track_url})")

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

    st.markdown(
        """
        <div class="vibe-nowplaying-card">
        """,
        unsafe_allow_html=True,
    )

    st.subheader(f"🎧 {title}")
    st.markdown(f"**{track_name}** — {artist}")
    if genre:
        st.caption(f"Genre: {genre}")

    cols = st.columns(3)
    if vibe_sim is not None:
        cols[0].metric("Vibe similarity", f"{vibe_sim:.2f}")
    if pop_norm is not None:
        cols[1].metric("Popularity (0–1)", f"{pop_norm:.2f}")
    if vibe_score is not None:
        cols[2].metric("Hybrid score", f"{vibe_score:.2f}")

    render_spotify_embed(track_id)

    if explanation:
        st.markdown("")
        st.info(explanation)

    st.markdown("</div>", unsafe_allow_html=True)


def render_playlist_with_controls(
    mode_key: str,
    explanation_provider: Optional[Callable[[pd.Series], Optional[str]]] = None,
) -> None:
    """
    Shared player UI:
      - reads playlist + current_idx from st.session_state for `mode_key`
      - renders Now playing card
      - Previous / Skip buttons
      - Up Next as mini cards with inline "Jump here" buttons
    """
    playlist: Optional[pd.DataFrame] = st.session_state.get(f"{mode_key}_playlist")
    if playlist is None or playlist.empty:
        return

    current_idx: int = st.session_state.get(f"{mode_key}_current_idx", 0)
    current_idx = max(0, min(current_idx, len(playlist) - 1))
    st.session_state[f"{mode_key}_current_idx"] = current_idx

    now_playing = playlist.iloc[current_idx]

    explanation = None
    if explanation_provider is not None:
        try:
            explanation = explanation_provider(now_playing)
        except Exception as e:
            explanation = f"(Could not generate explanation: {e})"

    st.markdown("---")
    st.markdown("#### Playlist")

    render_now_playing_card(
        now_playing,
        title="Now playing",
        explanation=explanation,
    )

    # Transport controls
    col_prev, col_skip = st.columns([1, 1])
    with col_prev:
        disabled_prev = current_idx == 0
        if st.button("⏮ Previous", key=f"{mode_key}_prev", disabled=disabled_prev):
            if current_idx > 0:
                st.session_state[f"{mode_key}_current_idx"] = current_idx - 1
                st.rerun()

    with col_skip:
        disabled_skip = current_idx >= len(playlist) - 1
        if st.button("⏭ Skip", key=f"{mode_key}_skip", disabled=disabled_skip):
            if current_idx < len(playlist) - 1:
                st.session_state[f"{mode_key}_current_idx"] = current_idx + 1
                st.rerun()

    # Up Next
    up_next_start = current_idx + 1
    st.markdown("### 🎵 Up Next")

    if up_next_start >= len(playlist):
        st.caption("Queue is empty. Skip backwards or generate a new playlist.")
        return

    for pos in range(up_next_start, len(playlist)):
        row = playlist.iloc[pos]
        display_pos = pos + 1

        title = row.get("track_name", "Unknown track")
        artist = row.get("artists", "Unknown artist")
        genre = row.get("track_genre", None)
        vibe_sim = row.get("vibe_similarity", None)

        info_col, btn_col = st.columns([0.82, 0.18])

        with info_col:
            meta_bits = []
            if genre:
                meta_bits.append(genre)
            if vibe_sim is not None:
                meta_bits.append(f"vibe sim {vibe_sim:.2f}")

            meta_text = " • ".join(meta_bits) if meta_bits else ""

            card_html = f"""
            <div class="vibe-upnext-card">
                <div class="vibe-upnext-index">#{display_pos}</div>
                <div class="vibe-upnext-title">{title}</div>
                <div class="vibe-upnext-artist">{artist}</div>
                <div class="vibe-upnext-meta">{meta_text}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        with btn_col:
            if st.button("▶ Jump here", key=f"{mode_key}_play_{pos}"):
                st.session_state[f"{mode_key}_current_idx"] = pos
                st.rerun()


# MODE 1: SLIDERS

def page_mode1_sliders(engine: VibeEngine):
    st.markdown("### Dial in a vibe 🎚️")
    st.write(
        "Drag the sliders to describe your current mood, and I’ll sift through "
        "114k tracks to match that vibe."
    )

    with st.form("mode1_form"):
        col1, col2 = st.columns(2)

        with col1:
            dance = st.slider("Danceability 💃", 0, 100, 80)
            st.caption("How easy it is to move to the track — 0 = stiff, 100 = super groovy.")

            energy = st.slider("Energy ⚡", 0, 100, 75)
            st.caption("Overall intensity — 0 = mellow / sleepy, 100 = loud / high-intensity.")

            valence = st.slider("Positivity (Valence) 😃", 0, 100, 70)
            st.caption("Emotional positivity — 0 = sad / moody, 100 = bright / euphoric.")

            tempo = st.slider("Tempo 🥁 (slow ←→ fast)", 0, 100, 65)
            st.caption("Speed of the song — 0 = very slow, 100 = very fast BPM.")

        with col2:
            acoustic = st.slider("Acousticness 🎸", 0, 100, 20)
            st.caption("How acoustic vs electronic — 0 = fully electronic, 100 = unplugged / organic.")

            instr = st.slider("Instrumentalness 🎼", 0, 100, 10)
            st.caption(
                "How likely the track has no vocals — 0 = mostly singing/rap, "
                "100 = mostly instruments only."
            )

            speech = st.slider("Speechiness 🗣️", 0, 100, 15)
            st.caption(
                "How much of the audio is spoken words — 0 = mostly musical/sung, "
                "100 = talky / rap / spoken-word."
            )

        st.markdown("---")
        st.markdown("**Fine-tune how I rank songs**")

        hide_explicit = st.checkbox("Hide explicit tracks", value=True)

        lambda_vibe = st.slider(
            "Vibe vs Popularity",
            0.0, 1.0, 0.8, 0.05,
        )
        st.caption("0 = just what’s popular, 1 = pure vibe match based on audio features.")

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

    if submitted:
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

        st.session_state["mode1_playlist"] = recs
        st.session_state["mode1_current_idx"] = 0

    render_playlist_with_controls(mode_key="mode1")


# MODE 2A: SEED FROM SONG

def page_mode2a_seed_from_song(mode2a: Mode2ASeedFromSong):
    st.markdown("### Start from a song 🎵")
    st.write(
        "Tell me a song you like. I’ll search inside the library, "
        "you pick one, and I’ll build a vibe-matching playlist from it."
    )

    if "mode2a_results" not in st.session_state:
        st.session_state["mode2a_results"] = None
    if "mode2a_seed_idx" not in st.session_state:
        st.session_state["mode2a_seed_idx"] = None

    with st.form("mode2a_search_form"):
        query = st.text_input(
            "Type a song or artist name",
            value="",
            placeholder="Feel Good Inc",
        )
        submitted_search = st.form_submit_button("🔍 Search library")

    if submitted_search:
        if not query.strip():
            st.warning("Please enter a song or artist name.")
        else:
            results = mode2a.search_tracks(query=query, max_results=15)
            st.session_state["mode2a_results"] = results

    results = st.session_state["mode2a_results"]

    if results is None or results.empty:
        if submitted_search and query and results is not None and results.empty:
            st.error("No matches found. Try a different spelling or song.")
        if st.session_state.get("mode2a_playlist") is not None:
            def explanation_provider(row: pd.Series) -> Optional[str]:
                seed_idx = st.session_state.get("mode2a_seed_idx")
                if seed_idx is None:
                    return None
                return mode2a.explain_recommendation(
                    seed_idx=int(seed_idx),
                    rec_df_index=int(row.name),
                    top_n_features=3,
                )

            seed_idx = st.session_state.get("mode2a_seed_idx")
            if seed_idx is not None:
                st.markdown("---")
                st.markdown("#### Seed track")
                seed_row = mode2a.df.loc[int(seed_idx)]
                st.write(seed_row[["track_name", "artists", "track_genre", "track_id"]])

            render_playlist_with_controls("mode2a", explanation_provider=explanation_provider)
        return

    st.markdown("#### Search results")

    display_df = results[["track_name", "artists", "track_genre", "track_id"]].copy()
    st.dataframe(
        display_df.reset_index().rename(columns={"index": "df_index"})
    )

    idx_list = results.index.tolist()
    chosen_idx = st.selectbox(
        "Pick a seed track",
        options=idx_list,
        format_func=lambda i: f"{results.loc[i, 'track_name']} — {results.loc[i, 'artists']}",
    )

    if st.button("🎧 Use this as my seed"):
        df_index = int(chosen_idx)
        with st.spinner("Building a vibe-matching playlist from your seed..."):
            now_playing, up_next, seed_idx = mode2a.recommend_from_seed(
                df_index=df_index,
                top_k=10,
                lambda_vibe=0.8,
                hide_explicit=True,
            )

        playlist = pd.concat([now_playing.to_frame().T, up_next])
        st.session_state["mode2a_playlist"] = playlist
        st.session_state["mode2a_current_idx"] = 0
        st.session_state["mode2a_seed_idx"] = int(seed_idx)
        st.success("Got it! Scroll down to see your playlist.")

    if st.session_state.get("mode2a_playlist") is not None:
        def explanation_provider(row: pd.Series) -> Optional[str]:
            seed_idx = st.session_state.get("mode2a_seed_idx")
            if seed_idx is None:
                return None
            try:
                return mode2a.explain_recommendation(
                    seed_idx=int(seed_idx),
                    rec_df_index=int(row.name),
                    top_n_features=3,
                )
            except Exception as e:
                return f"(Could not generate explanation: {e})"

        seed_idx = st.session_state.get("mode2a_seed_idx")
        if seed_idx is not None:
            st.markdown("---")
            st.markdown("#### Seed track")
            seed_row = mode2a.df.loc[int(seed_idx)]
            st.write(seed_row[["track_name", "artists", "track_genre", "track_id"]])

        render_playlist_with_controls("mode2a", explanation_provider=explanation_provider)


# MODE 2B: VIBE ROULETTE

def page_mode2b_vibe_roulette(mode2b: Mode2BVibeRoulette):
    st.markdown("### Vibe Roulette 🎲")
    st.write(
        "One click. I look at the current time, pick a vibe persona "
        "for you, and spin up a playlist."
    )

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

        playlist = pd.concat([now_playing.to_frame().T, up_next])
        st.session_state["mode2b_playlist"] = playlist
        st.session_state["mode2b_current_idx"] = 0
        st.session_state["mode2b_meta"] = meta

    meta = st.session_state.get("mode2b_meta")
    if meta is not None and st.session_state.get("mode2b_playlist") is not None:
        dt = meta.get("timestamp")
        if isinstance(dt, datetime.datetime):
            time_str = dt.strftime("%A %I:%M %p")
        else:
            time_str = datetime.datetime.now().strftime("%A %I:%M %p")

        persona = meta.get("persona_name", "Unknown persona")
        day_type = meta.get("day_type", meta.get("weekday_bucket", ""))
        time_bucket = meta.get("time_bucket", "")
        sliders = meta.get("persona_sliders", meta.get("sliders_used", {}))

        st.markdown("---")
        st.markdown("#### Your vibe spin")
        st.markdown(
            f"🕒 It’s **{time_str}** — "
            f"I’m sensing a **{persona}** mood "
            f"({day_type}, {time_bucket.replace('_', ' ')})."
        )

        with st.expander("See the persona vibe profile I used"):
            st.json(sliders)

        def explanation_provider(row: pd.Series) -> Optional[str]:
            return mode2b._explain_match(
                persona=TimeOfDayPersona(
                    name=meta.get("persona_name", ""),
                    sliders=sliders,
                    tagline=meta.get("persona_tagline", ""),
                    emoji=meta.get("persona_emoji", ""),
                    tags=tuple(meta.get("persona_tags", ())),
                ),
                now_playing=row,
                top_n_features=2,
            )

        render_playlist_with_controls("mode2b", explanation_provider=explanation_provider)


# MAIN APP

def main():
    st.set_page_config(
        page_title="Vibe Recommender",
        page_icon="🎶",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #181826 0, #050509 40%);
        }
        .vibe-header-logo {
            border-radius: 50%;
        }
        .vibe-mode-container {
            height: 100%;
        }
        
        .vibe-mode-card {
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(145deg, rgba(18,18,28,0.96), rgba(10,10,16,0.96));
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.6);

            height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .vibe-mode-container {
            height: 100%;
        }

        .vibe-mode-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.15rem;
        }
        .vibe-mode-caption {
            font-size: 0.9rem;
            color: #c3c3d7;
            margin-bottom: 0.4rem;
        }
        .vibe-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-bottom: 0.4rem;
        }
        .vibe-pill {
            font-size: 0.78rem;
            padding: 0.16rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
        }
        .vibe-nowplaying-card {
            border-radius: 18px;
            padding: 0.9rem 1rem 0.8rem;
            border: 1px solid rgba(255,255,255,0.12);
            background: radial-gradient(circle at 0% 0%, rgba(66,255,158,0.16), transparent 50%),
                        rgba(10,10,16,0.96);
            margin-bottom: 0.6rem;
        }
        .vibe-upnext-card {
            border-radius: 12px;
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.35rem;
            border: 1px solid #222;
            background: #0c0c0f;
        }
        .vibe-upnext-index {
            font-size: 0.85rem;
            color: #888;
        }
        .vibe-upnext-title {
            font-weight: 600;
            font-size: 0.98rem;
        }
        .vibe-upnext-artist {
            font-size: 0.9rem;
            color: #bbb;
        }
        .vibe-upnext-meta {
            font-size: 0.78rem;
            color: #888;
            margin-top: 0.15rem;
        }
        /* Global button styling (mode cards, controls, etc.) */
        .stButton>button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #1db954, #13a848);
            color: #050509;
            font-weight: 600;
            padding: 0.35rem 0.9rem;
            cursor: pointer;
        }
        .stButton>button:hover {
            filter: brightness(1.08);
            box-shadow: 0 8px 18px rgba(0,0,0,0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_logo_col, header_text_col = st.columns([1, 5])

    with header_logo_col:
        st.write("")
        st.write("")
        st.image("assets/pill_nav.png", width=100)

    with header_text_col:
        st.title("Vibe Recommender 🎧")
        st.caption(
            "A content-based music recommender that matches you on *vibe* — "
            "not just genre or popularity."
        )
        st.caption("Under the hood: cosine similarity, hybrid scoring, and time-of-day personas.")

    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = None

    try:
        engine, mode2a, mode2b = load_engine_and_modes(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find data file at `{DATA_PATH}`. Please update DATA_PATH.")
        return

    st.markdown("## Choose how you want to set the vibe")
    st.markdown(
        "Pick a starting point below. You can always come back and try another one."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="vibe-mode-container">
              <div class="vibe-mode-card">
                <div>
                  <div class="vibe-mode-title">🎚️ Dial in a vibe</div>
                  <div class="vibe-mode-caption">
                    You know how you want the music to <em>feel</em>.
                    Use sliders like energy, tempo, and acousticness to sketch your mood.
                  </div>
                  <div class="vibe-pill-row">
                    <span class="vibe-pill">7 sliders</span>
                    <span class="vibe-pill">vibe vs popularity</span>
                    <span class="vibe-pill">diversity-aware queue</span>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("🎚️ Start with sliders", key="hero_mode1"):
            st.session_state["selected_mode"] = "mode1"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="vibe-mode-container">
              <div class="vibe-mode-card">
                <div>
                  <div class="vibe-mode-title">🎵 Start from a song</div>
                  <div class="vibe-mode-caption">
                    You have one song in mind. I’ll find it in the library and build a
                    vibe-matching playlist around it.
                  </div>
                  <div class="vibe-pill-row">
                    <span class="vibe-pill">search library</span>
                    <span class="vibe-pill">seed-based</span>
                    <span class="vibe-pill">“why this track?” explanations</span>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🎵 Build from this song", key="hero_mode2a"):
            st.session_state["selected_mode"] = "mode2a"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="vibe-mode-container">
              <div class="vibe-mode-card">
                <div>
                  <div class="vibe-mode-title">🎲 Vibe Roulette</div>
                  <div class="vibe-mode-caption">
                    You’re indecisive or just curious. One click, I pick a persona
                    from the current time of day and spin up a playlist.
                  </div>
                  <div class="vibe-pill-row">
                    <span class="vibe-pill">time-of-day personas</span>
                    <span class="vibe-pill">soft exploration</span>
                    <span class="vibe-pill">one-click UX</span>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Spin this vibe", key="hero_mode2b"):
            st.session_state["selected_mode"] = "mode2b"
            st.rerun()

    st.markdown("---")

    selected_mode = st.session_state.get("selected_mode")

    if selected_mode == "mode1":
        st.markdown("#### Mode: Dial in a vibe")
        page_mode1_sliders(engine)
    elif selected_mode == "mode2a":
        st.markdown("#### Mode: Start from a song")
        page_mode2a_seed_from_song(mode2a)
    elif selected_mode == "mode2b":
        st.markdown("#### Mode: Vibe Roulette")
        page_mode2b_vibe_roulette(mode2b)
    else:
        st.info("Pick a mode above to get started.")


if __name__ == "__main__":
    main()

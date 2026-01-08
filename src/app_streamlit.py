import datetime
from typing import Optional, Callable, Any
from textwrap import dedent
import numpy as np
import pandas as pd
import streamlit as st

from vibe_engine import VibeEngine
from mode2a_seed_from_song import Mode2ASeedFromSong
from mode2b_vibe_roulette import Mode2BVibeRoulette, TimeOfDayPersona


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

    col_prev, col_skip = st.columns([1, 1])
    with col_prev:
        disabled_prev = current_idx == 0
        if st.button("⏮ Previous", key=f"{mode_key}_prev", disabled=disabled_prev, use_container_width=True):
            if current_idx > 0:
                st.session_state[f"{mode_key}_current_idx"] = current_idx - 1
                st.rerun()

    with col_skip:
        disabled_skip = current_idx >= len(playlist) - 1
        if st.button("⏭ Skip", key=f"{mode_key}_skip", disabled=disabled_skip, use_container_width=True):
            if current_idx < len(playlist) - 1:
                st.session_state[f"{mode_key}_current_idx"] = current_idx + 1
                st.rerun()

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


def page_mode2a_seed_from_song(mode2a: Mode2ASeedFromSong):
    st.markdown("### Start from a song 🎵")
    st.write(
        "Tell me a song you like. I’ll search inside the library. "
        "You pick one, and I’ll build a vibe-matching playlist from it."
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
            help="Start typing any song or artist from the library.",
        )
        search_submitted = st.form_submit_button("🔍 Search library")

    if search_submitted:
        if not query.strip():
            st.warning("Please enter a song or artist name.")
        else:
            results = mode2a.search_tracks(query=query, max_results=15)
            st.session_state["mode2a_results"] = results

    results = st.session_state["mode2a_results"]

    if results is None or results.empty:
        if results is not None and results.empty:
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
        .vibe-link-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 0.25rem 0 0.75rem;
        }
        a.vibe-link-btn {
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.52rem 0.95rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.14);
            color: #eaeaf5;
            font-weight: 650;
            letter-spacing: 0.2px;
        }
        a.vibe-link-btn:hover {
            background: rgba(255,255,255,0.11);
            border-color: rgba(66,255,158,0.35);
            box-shadow: 0 8px 18px rgba(0,0,0,0.5);
        }
        .vibe-link-sub {
            color: #bdbdd3;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }

        /* "How to use" -> Quick start card */
        .vibe-help-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(145deg, rgba(18,18,28,0.92), rgba(10,10,16,0.92));
            box-shadow: 0 10px 26px rgba(0,0,0,0.55);
            margin: 0.25rem 0 1.0rem;
        }
        .vibe-help-title {
            font-weight: 750;
            font-size: 1.05rem;
            margin-bottom: 0.15rem;
        }
        .vibe-help-caption {
            color: #c3c3d7;
            font-size: 0.92rem;
            margin-bottom: 0.75rem;
        }
        .help-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
        }
        .help-step {
            padding: 0.75rem 0.85rem;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
        }
        .help-step-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .help-badge {
            width: 1.55rem;
            height: 1.55rem;
            border-radius: 999px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            font-size: 0.85rem;
            background: rgba(66,255,158,0.14);
            border: 1px solid rgba(66,255,158,0.22);
            color: #dfffe9;
        }
        .help-step-body {
            color: #c3c3d7;
            font-size: 0.92rem;
            line-height: 1.25rem;
        }
        @media (max-width: 900px) {
            .help-grid { grid-template-columns: 1fr; }
        }
        .vibe-link-col {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            align-items: flex-end;
            justify-content: flex-start;
            margin-top: 0.25rem;
        }
        a.vibe-link-btn {
            min-width: 210px;
            justify-content: center;
        }

        html, body {
            font-size: 18px; /* increase base size */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header with logo + title
    header_logo_col, header_text_col, header_links_col = st.columns([1, 5, 2])
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
        st.caption(
            "Built with Python, Streamlit, and a 7D audio feature space "
            "(danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness)."
        )

    with header_links_col:
        st.markdown(
            dedent("""
            <div class="vibe-link-col">
            <a class="vibe-link-btn" href="https://jessicabat.github.io/vibe-recommender/" target="_blank" rel="noopener noreferrer">
                🌐 Project website <span style="opacity:0.75;">↗</span>
            </a>
            <a class="vibe-link-btn" href="https://github.com/jessicabat/vibe-recommender" target="_blank" rel="noopener noreferrer">
                💻 GitHub repo <span style="opacity:0.75;">↗</span>
            </a>
            </div>
            """),
            unsafe_allow_html=True,
        )


    st.markdown("---")

    # Load engine + modes (with a nice error if data is missing)
    try:
        engine, mode2a, mode2b = load_engine_and_modes(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find data file at `{DATA_PATH}`. Please update DATA_PATH.")
        return

    st.markdown(
        dedent("""
        <div class="vibe-help-card">
        <div class="vibe-help-title">Quick start</div>
        <div class="vibe-help-caption">
            Pick a starting point, generate a playlist, then explore by skipping around.
            This demo is built to feel simple on the surface — with smart ranking underneath.
        </div>

        <div class="help-grid">
            <div class="help-step">
            <div class="help-step-kicker"><span class="help-badge">1</span>Choose a mode</div>
            <div class="help-step-body">
                <b>Sliders</b> if you know the mood,
                <b>Start from a song</b> if you have an anchor track,
                or <b>Roulette</b> if you want a one-click surprise.</div></div>
            <div class="help-step">
            <div class="help-step-kicker"><span class="help-badge">2</span>Generate recommendations</div>
            <div class="help-step-body">
                You’ll get a ranked queue designed to balance
                “matches the vibe” with “doesn’t feel repetitive.”</div></div>
            <div class="help-step">
            <div class="help-step-kicker"><span class="help-badge">3</span>Explore & refine</div>
            <div class="help-step-body">
                Try a different vibe, tweak one slider, or switch modes —
                the point is to feel how the system responds.</div></div></div>
        </div>
        """),
        unsafe_allow_html=True,
    )



    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = None

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
        if st.button("🎚️ Start with sliders", key="hero_mode1", use_container_width=True):
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
        if st.button("🎵 Build from a song", key="hero_mode2a", use_container_width=True):
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
        if st.button("🎲 Spin the roulette", key="hero_mode2b", use_container_width=True):
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

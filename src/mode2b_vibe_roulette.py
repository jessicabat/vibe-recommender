# mode2b_vibe_roulette.py

"""
Mode 2B — Vibe Roulette (Time-of-Day Aware + Storytelling + Spotify Embed)

User flow (1 click):
    - User presses "🎲 Vibe Roulette"
    - We:
        1. Inspect current time and weekday/weekend.
        2. Map that to a (day_type, time_bucket) context.
        3. Randomly choose a persona for that context.
        4. Use the persona's sliders with your Mode 1 engine to get recommendations.
        5. Autoplay the #1 track, queue the rest.
        6. Return a metadata dict describing what happened (persona, time, etc).

Storytelling layers:
    - Persona intro templates (randomized)
    - Mood adjectives from slider config
    - Tiny "vibe card" (emoji + tags)
    - Continuity hook: reference last spin if available
    - Light explanation of why the top track fits the chosen persona
    - Spotify URL + embedded player (in Jupyter, if available)

The engine is expected to be something like SliderVibeRecommender with:
    - df: pd.DataFrame of tracks
    - vibe_cols: list of feature names
    - X: np.ndarray of standardized features (N x D)
    - _sliders_to_target_vec(sliders): maps 0–100 sliders into engine's feature space
    - recommend_by_sliders(sliders, top_k, lambda_vibe, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeOfDayPersona:
    name: str
    sliders: Dict[str, float]  # 0–100
    tagline: str
    emoji: str = ""
    tags: Tuple[str, ...] = ()


# ---------------------------------------------------------------------
# 1. Define personas
# ---------------------------------------------------------------------

# -----------------------------
# Dataset-calibrated personas
# -----------------------------

SOFT_SUNRISE = TimeOfDayPersona(
    name="Soft Sunrise Focus",
    sliders={
        "danceability":     45,  # a bit groovy, not dance-heavy
        "energy":           40,  # soft-mid energy
        "valence":          60,  # positive but not hyper
        "tempo":            40,  # ~25th percentile tempo (~100 BPM)
        "acousticness":     70,  # fairly acoustic
        "instrumentalness": 10,  # mostly vocals, not full-on instrumental
        "speechiness":      10,  # normal pop-level speechiness
    },
    tagline="gentle, warm, slightly acoustic morning focus vibes",
    emoji="🌅☕️",
    tags=("soft & cozy", "acoustic", "focus"),
)

FLOW_STATE = TimeOfDayPersona(
    name="Flow State Focus",
    sliders={
        "danceability":     50,  # moderate groove
        "energy":           55,  # slightly above median energy
        "valence":          45,  # emotionally neutral / balanced
        "tempo":            50,  # median tempo (~120 BPM)
        "acousticness":     55,  # somewhat acoustic, not fully unplugged
        "instrumentalness": 30,  # more instrumental to reduce lyric distraction
        "speechiness":       5,  # very low speechiness
    },
    tagline="steady, low-distraction beats for deep work",
    emoji="📚💻",
    tags=("deep work", "steady", "instrumental"),
)

UNWIND_AFTER_HOURS = TimeOfDayPersona(
    name="Unwind After Hours",
    sliders={
        "danceability":     55,  # gentle groove
        "energy":           50,  # mid energy
        "valence":          55,  # slightly positive
        "tempo":            45,  # a bit slower than median
        "acousticness":     60,  # warm acoustic feel
        "instrumentalness": 35,  # mix of instrumental and vocal
        "speechiness":      10,  # not talk-heavy
    },
    tagline="warm, laid-back tracks for evening decompression",
    emoji="🛋️🌙",
    tags=("chill", "warm", "unwind"),
)

MIDNIGHT_LOFI = TimeOfDayPersona(
    name="Midnight Lo-Fi Drift",
    sliders={
        "danceability":     40,  # more sway than dance
        "energy":           35,  # low-ish energy
        "valence":          35,  # slightly on the moody side
        "tempo":            45,  # chill tempo, not too slow to break flow
        "acousticness":     65,  # cozy / warm timbre
        "instrumentalness": 50,  # strong instrumental presence
        "speechiness":       5,  # minimal vocal chatter
    },
    tagline="hazy, lo-fi textures for late-night scrolling",
    emoji="🌌📼",
    tags=("lo-fi", "hazy", "slow"),
)

LAZY_BRUNCH = TimeOfDayPersona(
    name="Lazy Brunch Breeze",
    sliders={
        "danceability":     55,  # light movement
        "energy":           50,  # comfortable mid energy
        "valence":          65,  # clearly positive
        "tempo":            50,  # easy medium tempo
        "acousticness":     55,  # organic, brunchy feel
        "instrumentalness": 15,  # mostly songs with vocals
        "speechiness":      10,  # casual pop speechiness
    },
    tagline="weekend late morning, sunny, easy-going brunch grooves",
    emoji="🥞🌤️",
    tags=("sunny", "breezy", "brunch"),
)

GOLDEN_HOUR = TimeOfDayPersona(
    name="Golden Hour Groove",
    sliders={
        "danceability":     70,  # clearly danceable
        "energy":           70,  # upbeat but not aggressive
        "valence":          75,  # high positivity
        "tempo":            60,  # slightly above median tempo (~140+ BPM)
        "acousticness":     35,  # more produced / pop-leaning
        "instrumentalness": 10,  # vocal-driven
        "speechiness":      15,  # a bit of talk/rap ok
    },
    tagline="feel-good pop and indie for golden hour walks",
    emoji="🌅✨",
    tags=("glowy & upbeat", "walk vibes", "feel-good"),
)

NIGHT_OUT = TimeOfDayPersona(
    name="Night Out Pre-Game",
    sliders={
        "danceability":     80,  # very danceable (~90th percentile)
        "energy":           85,  # high energy (~75–95th)
        "valence":          70,  # fun, upbeat
        "tempo":            65,  # fast (~150–160 BPM region)
        "acousticness":     10,  # very produced / electronic
        "instrumentalness":  5,  # mostly vocal anthems
        "speechiness":      20,  # rap / party hooks allowed
    },
    tagline="high-energy bangers to get you out the door",
    emoji="🥂💃",
    tags=("party", "high energy", "dancefloor"),
)

NEON_CITY = TimeOfDayPersona(
    name="Neon City Ride",
    sliders={
        "danceability":     75,  # smooth groove
        "energy":           70,  # energetic but not chaotic
        "valence":          55,  # emotionally mixed / bittersweet
        "tempo":            60,  # driving tempo
        "acousticness":     20,  # mostly electronic / polished
        "instrumentalness": 25,  # some instrumental textures
        "speechiness":      10,  # not too talky
    },
    tagline="late-night drive: pulsing, slightly moody, cinematic",
    emoji="🌃🚗",
    tags=("dark & wavy", "fast tempo", "night drive"),
)



# Map (day_type, time_bucket) to possible personas
# day_type: "weekday" or "weekend"

CONTEXT_PERSONAS: Dict[Tuple[str, str], List[TimeOfDayPersona]] = {
    # Weekday
    ("weekday", "morning"):   [SOFT_SUNRISE],
    ("weekday", "afternoon"): [FLOW_STATE],
    ("weekday", "evening"):   [UNWIND_AFTER_HOURS],
    ("weekday", "late_night"): [MIDNIGHT_LOFI],

    # Weekend
    ("weekend", "morning"):   [LAZY_BRUNCH],
    ("weekend", "afternoon"): [GOLDEN_HOUR],
    ("weekend", "evening"):   [NIGHT_OUT],
    ("weekend", "late_night"): [NEON_CITY,MIDNIGHT_LOFI],
}



# ---------------------------------------------------------------------
# 2. Helper functions: context inference
# ---------------------------------------------------------------------

def _get_day_type(dt: datetime) -> str:
    """Return 'weekday' or 'weekend' from a datetime."""
    # Monday=0, Sunday=6
    return "weekend" if dt.weekday() >= 5 else "weekday"


def _get_time_bucket(dt: datetime) -> str:
    """
    Map hour to a coarse bucket:
        morning   : 05-11
        afternoon : 11-17
        evening   : 17-22
        late_night: 22-05
    """
    h = dt.hour
    if 5 <= h < 11:
        return "morning"
    elif 11 <= h < 17:
        return "afternoon"
    elif 17 <= h < 22:
        return "evening"
    else:
        return "late_night"


# ---------------------------------------------------------------------
# 3. Main class: Mode 2B Vibe Roulette
# ---------------------------------------------------------------------

class Mode2BVibeRoulette:
    """
    Mode 2B: 1-click "Vibe Roulette" based on time-of-day and weekday/weekend.

    Usage (in a notebook):

        from mode1_sliders import SliderVibeRecommender
        from mode2b_vibe_roulette import Mode2BVibeRoulette

        df = pd.read_csv("spotify_tracks.csv")
        engine = SliderVibeRecommender(df_tracks=df)
        roulette = Mode2BVibeRoulette(engine)

        now_playing, up_next, meta = roulette.spin(top_k=10, lambda_vibe=0.8)
        roulette.print_spin_result(now_playing, up_next, meta)
    """

    def __init__(self, engine):
        """
        engine: an instance of your slider-based recommender, e.g. SliderVibeRecommender.
        It must implement:
            - df (DataFrame)
            - vibe_cols (list)
            - X (np.ndarray of standardized features)
            - _sliders_to_target_vec(sliders)  # used for light explanation
            - recommend_by_sliders(sliders, top_k, lambda_vibe, ...)
        """
        self.engine = engine
        self.df = engine.df
        self._last_meta: Optional[Dict] = None  # for continuity hooks

    # ---------------- internal helpers ----------------

    def _choose_persona_for_context(
        self,
        day_type: str,
        time_bucket: str,
    ) -> TimeOfDayPersona:
        """Randomly choose a persona for the given (day_type, time_bucket)."""
        key = (day_type, time_bucket)
        if key not in CONTEXT_PERSONAS:
            # Fallback if missing (shouldn't happen)
            key = ("weekday", "afternoon")
        candidates = CONTEXT_PERSONAS[key]
        return random.choice(candidates)

    def _describe_sliders_short(self, sliders: Dict[str, float]) -> str:
        """
        Turn sliders into a small mood description using adjectives like:
        - "soft & cozy", "glowy & upbeat", "dark and wavy", "sparkly & high-tempo"
        """
        phrases = []

        energy = sliders.get("energy", 50)
        dance = sliders.get("danceability", 50)
        valence = sliders.get("valence", 50)
        tempo = sliders.get("tempo", 50)
        acousticness = sliders.get("acousticness", 50)
        instrumentalness = sliders.get("instrumentalness", 50)

        # Energy-driven phrases
        if energy >= 80 and tempo >= 70:
            phrases.append("sparkly & high-tempo")
        elif energy >= 70:
            phrases.append("high energy")
        elif energy <= 35:
            phrases.append("soft & low-key")

        # Danceability
        if dance >= 75:
            phrases.append("very danceable")
        elif dance <= 35:
            phrases.append("more chill than dancey")

        # Valence / mood
        if valence >= 70:
            phrases.append("glowy & upbeat")
        elif valence <= 35:
            phrases.append("moody / introspective")

        # Tempo
        if tempo >= 80:
            phrases.append("fast tempo")
        elif tempo <= 40:
            phrases.append("slower, relaxed tempo")

        # Acoustic / electronic
        if acousticness >= 70:
            phrases.append("soft & cozy acoustic")
        elif acousticness <= 30:
            phrases.append("more electronic / synthetic")

        # Instrumental
        if instrumentalness >= 70:
            phrases.append("mostly instrumental")

        if not phrases:
            return "balanced vibe"

        # Deduplicate and keep just a few highlights
        uniq_phrases = []
        for p in phrases:
            if p not in uniq_phrases:
                uniq_phrases.append(p)

        return " · ".join(uniq_phrases[:3])

    def _explain_match(
        self,
        persona: TimeOfDayPersona,
        now_playing: pd.Series,
        top_n_features: int = 2,
    ) -> str:
        """
        Light explanation: which vibe dimensions are closest between
        the persona sliders and the recommended track in the engine's
        standardized feature space?
        """
        # Need engine's internal mapping and standardized matrix
        if not hasattr(self.engine, "_sliders_to_target_vec"):
            return ""

        try:
            # persona target in standardized space
            target_vec = self.engine._sliders_to_target_vec(persona.sliders)

            # locate track vector in engine.X
            # now_playing.name should be the df index
            if now_playing.name not in self.df.index:
                return ""
            pos = self.df.index.get_loc(now_playing.name)
            track_vec = self.engine.X[pos]

            diff = np.abs(target_vec - track_vec)
            feature_diffs = list(zip(self.engine.vibe_cols, diff))
            feature_diffs.sort(key=lambda x: x[1])  # smallest diff = best match

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

            if not pretty_features:
                return ""

            return "This track lines up strongly on " + ", ".join(pretty_features) + "."
        except Exception:
            # Fail silently; explanation is optional sugar
            return ""

    def _build_persona_intro(
        self,
        meta: Dict,
        now_playing: pd.Series,
    ) -> str:
        """
        Randomized persona intro line using templates and context.
        """
        persona_name = meta.get("persona_name", "Unknown Vibe")
        tagline = meta.get("persona_tagline", "")
        day_type = meta.get("day_type", "weekday")
        time_bucket = meta.get("time_bucket", "time")
        vibe_summary = meta.get("vibe_summary", "")

        dt: Optional[datetime] = meta.get("timestamp")
        if dt is not None:
            time_str = dt.strftime("%A • %I:%M %p").lstrip("0")
        else:
            time_str = "Current time"

        bucket_label = time_bucket.replace("_", " ")

        track_name = now_playing.get("track_name", "this track")
        artist = now_playing.get("artists", "Unknown artist")

        templates = [
            (
                f"It’s {time_str}. The algorithm whispers *{persona_name}*."
                f" Starting with **{track_name}** by {artist} to set the tone."
            ),
            (
                f"Sensors detect {bucket_label} {day_type} energy."
                f" Spinning *{persona_name}* and opening with **{track_name}** by {artist}."
            ),
            (
                f"Reading your {day_type} mood..."
                f" prescribing *{persona_name}* — first dose is **{track_name}** by {artist}."
            ),
        ]

        intro = random.choice(templates)
        if tagline:
            intro += f"\nTagline: {tagline}."
        if vibe_summary:
            intro += f"\nVibe: {vibe_summary}."
        return intro
    
    def _choose_now_playing_with_exploration(
        self,
        recs: pd.DataFrame,
        explore_k: int = 30,
        temperature: float = 0.2,
    ):
        """
        Given a sorted recs dataframe (best first),
        pick a 'now playing' track with a bit of randomness.

        - explore_k: how many top tracks to consider for the spin.
        - temperature: controls how peaky the probabilities are.
                       lower = more greedy (closer to always top-1),
                       higher = more random.
        """
        if recs.empty:
            raise RuntimeError("No recommendations available for this persona/context.")

        # Work on a top slice only
        k = min(explore_k, len(recs))
        top = recs.iloc[:k].copy()

        if "vibe_score" not in top.columns:
            # Fallback: if somehow no vibe_score, just pick uniformly
            probs = np.ones(k) / k
        else:
            scores = top["vibe_score"].values.astype(float)

            # Temperature-softmax over scores
            # (scores are in ~[0, 1], so small temperature still works)
            scaled = scores / max(temperature, 1e-6)
            exps = np.exp(scaled - np.max(scaled))
            probs = exps / (exps.sum() + 1e-8)

        # Randomly choose one index from the top slice
        choice_idx = np.random.choice(k, p=probs)
        now_playing = top.iloc[choice_idx]

        # Build queue: everything else in 'top' except the chosen row
        up_next = top.drop(index=now_playing.name)

        return now_playing, up_next

    # ---------------- public API ----------------

    def spin(
        self,
        top_k: int = 10,
        lambda_vibe: float = 0.8,
        min_popularity: Optional[int] = None,
        hide_explicit: bool = True,
        allowed_genres: Optional[List[str]] = None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
        when: Optional[datetime] = None,
    ) -> Tuple[pd.Series, pd.DataFrame, Dict]:
        """
        Perform one Vibe Roulette spin.

        Returns
        -------
        now_playing : pd.Series
            First recommended track (autoplay candidate).
        up_next : pd.DataFrame
            Remaining queue.
        meta : dict
            Metadata including:
                - timestamp
                - day_type
                - time_bucket
                - persona_name
                - persona_tagline
                - persona_sliders
                - emoji
                - tags
                - vibe_summary
                - previous_meta (for continuity hooks)
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        dt = when or datetime.now()
        day_type = _get_day_type(dt)
        time_bucket = _get_time_bucket(dt)

        persona = self._choose_persona_for_context(day_type, time_bucket)

        # recs = self.engine.recommend_by_sliders(
        #     sliders=persona.sliders,
        #     top_k=top_k,
        #     lambda_vibe=lambda_vibe,
        #     min_popularity=min_popularity,
        #     hide_explicit=hide_explicit,
        #     allowed_genres=allowed_genres,
        #     diversity=diversity,
        #     diversity_threshold=diversity_threshold,
        # )
        recs = self.engine.recommend_by_sliders(
            sliders=persona.sliders,
            top_k=100,              # ask for a decent pool
            lambda_vibe=lambda_vibe,
            hide_explicit=hide_explicit,
            min_popularity=min_popularity,
        )

        if recs.empty:
            raise RuntimeError("Vibe Roulette produced no recommendations. Check filters / data.")

        # now_playing = recs.iloc[0]
        # up_next = recs.iloc[1:].copy()
        now_playing, up_next = self._choose_now_playing_with_exploration(
            recs,
            explore_k=30,           # consider top 30 for randomness
            temperature=0.25,       # tweak this if it feels too random / too greedy
        )

        vibe_summary = self._describe_sliders_short(persona.sliders)

        meta = {
            "timestamp": dt,
            "timestamp_iso": dt.isoformat(timespec="seconds"),
            "day_type": day_type,
            "time_bucket": time_bucket,
            "persona_name": persona.name,
            "persona_tagline": persona.tagline,
            "persona_sliders": persona.sliders,
            "persona_emoji": persona.emoji,
            "persona_tags": persona.tags,
            "vibe_summary": vibe_summary,
            "previous_meta": self._last_meta,
        }

        # update continuity state
        self._last_meta = meta

        return now_playing, up_next, meta

    def print_spin_result(
        self,
        now_playing: pd.Series,
        up_next: pd.DataFrame,
        meta: Dict,
        show_story: bool = True,
        show_explanation: bool = True,
    ) -> None:
        """
        Pretty-print the Vibe Roulette result in a notebook.

        - Story line about time & persona
        - Tiny "vibe card"
        - Continuity hook referencing previous spin
        - "Now playing" with Spotify URL + embedded player
        - "Up Next" table
        """
        try:
            from IPython.display import display, IFrame, HTML  # type: ignore
            have_ipython = True
        except ImportError:
            # Fallback: still works in plain console, just no embed/HTML
            have_ipython = False
            display = None  # type: ignore
            IFrame = None   # type: ignore
            HTML = None     # type: ignore

        track_name = now_playing.get("track_name", "Unknown track")
        artist = now_playing.get("artists", "Unknown artist")
        track_id = now_playing.get("track_id", None)

        dt: Optional[datetime] = meta.get("timestamp")
        if dt is not None:
            time_str = dt.strftime("%A • %I:%M %p").lstrip("0")
        else:
            time_str = "Current time"

        day_type = meta.get("day_type", "weekday")
        time_bucket = meta.get("time_bucket", "time")
        persona_name = meta.get("persona_name", "Unknown Vibe")
        tagline = meta.get("persona_tagline", "")
        vibe_summary = meta.get("vibe_summary", "")
        persona_emoji = meta.get("persona_emoji", "")
        persona_tags = meta.get("persona_tags", ())
        previous_meta = meta.get("previous_meta")

        bucket_label = time_bucket.replace("_", " ")

        if show_story:
            print(f"🎲 Vibe Roulette • {time_str}")
            print(f"Context: {day_type.capitalize()} {bucket_label}")
            print()

            # Tiny vibe card
            tag_str = " · ".join(persona_tags) if persona_tags else vibe_summary
            print(f"{persona_emoji}  Persona: **{persona_name}**")
            if tagline:
                print(f"   {tagline}")
            if tag_str:
                print(f"   {tag_str}")
            print()

            # Continuity hook
            if previous_meta is not None:
                prev_name = previous_meta.get("persona_name", "a different vibe")
                prev_bucket = previous_meta.get("time_bucket", "").replace("_", " ")
                print(
                    f"Last spin felt like *{prev_name}* ({prev_bucket}). "
                    f"Today we’re shifting into *{persona_name}*."
                )
                print()

            # Persona intro line involving actual track
            intro = self._build_persona_intro(meta, now_playing)
            print(intro)
            print()

        # Now playing
        print(f"🎧 Now playing: {track_name} — {artist}")
        if isinstance(track_id, str) and track_id:
            spotify_url = f"https://open.spotify.com/track/{track_id}"
            print(f"🔗 Open in Spotify: {spotify_url}")

            # Embedded player (Jupyter)
            if have_ipython and IFrame is not None:
                embed_url = f"https://open.spotify.com/embed/track/{track_id}"
                print("\n▶️ Embedded player:")
                display(IFrame(src=embed_url, width=320, height=80))

        # Light explanation of why this track fits the persona
        if show_explanation:
            expl = self._explain_match(
                persona=TimeOfDayPersona(
                    name=persona_name,
                    sliders=meta.get("persona_sliders", {}),
                    tagline=tagline,
                    emoji=persona_emoji,
                    tags=tuple(persona_tags),
                ),
                now_playing=now_playing,
                top_n_features=2,
            )
            if expl:
                print("\nℹ️ Why this track?")
                print(expl)

        # Queue
        if not up_next.empty:
            print("\n🎵 Up Next:")
            if have_ipython and display is not None:
                cols = []
                for c in ["track_name", "artists", "track_genre", "vibe_score", "vibe_similarity"]:
                    if c in up_next.columns:
                        cols.append(c)
                if not cols:
                    cols = up_next.columns.tolist()
                display(up_next[cols])
            else:
                # Plain text fallback
                print(up_next[["track_name", "artists"]].head())
        else:
            print("\n(No additional tracks in queue.)")
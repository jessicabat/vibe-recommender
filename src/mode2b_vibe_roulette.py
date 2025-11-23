# mode2b_vibe_roulette.py

"""
Mode 2B — Vibe Roulette (Time-of-Day Aware)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import random
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeOfDayPersona:
    name: str
    sliders: Dict[str, float]
    tagline: str
    emoji: str = ""
    tags: Tuple[str, ...] = ()



SOFT_SUNRISE = TimeOfDayPersona(
    name="Soft Sunrise Focus",
    sliders={
        "danceability":     45,
        "energy":           40,
        "valence":          60,
        "tempo":            40,
        "acousticness":     70,
        "instrumentalness": 10,
        "speechiness":      10,
    },
    tagline="gentle, warm, slightly acoustic morning focus vibes",
    emoji="🌅☕️",
    tags=("soft & cozy", "acoustic", "focus"),
)

FLOW_STATE = TimeOfDayPersona(
    name="Flow State Focus",
    sliders={
        "danceability":     50,
        "energy":           55,
        "valence":          45,
        "tempo":            50,
        "acousticness":     55,
        "instrumentalness": 30,
        "speechiness":       5,
    },
    tagline="steady, low-distraction beats for deep work",
    emoji="📚💻",
    tags=("deep work", "steady", "instrumental"),
)

UNWIND_AFTER_HOURS = TimeOfDayPersona(
    name="Unwind After Hours",
    sliders={
        "danceability":     55,
        "energy":           50,
        "valence":          55,
        "tempo":            45,
        "acousticness":     60,
        "instrumentalness": 35,
        "speechiness":      10,
    },
    tagline="warm, laid-back tracks for evening decompression",
    emoji="🛋️🌙",
    tags=("chill", "warm", "unwind"),
)

MIDNIGHT_LOFI = TimeOfDayPersona(
    name="Midnight Lo-Fi Drift",
    sliders={
        "danceability":     40,
        "energy":           35,
        "valence":          35,
        "tempo":            45,
        "acousticness":     65,
        "instrumentalness": 50,
        "speechiness":       5,
    },
    tagline="hazy, lo-fi textures for late-night scrolling",
    emoji="🌌📼",
    tags=("lo-fi", "hazy", "slow"),
)

LAZY_BRUNCH = TimeOfDayPersona(
    name="Lazy Brunch Breeze",
    sliders={
        "danceability":     55,
        "energy":           50,
        "valence":          65,
        "tempo":            50,
        "acousticness":     55,
        "instrumentalness": 15,
        "speechiness":      10,
    },
    tagline="weekend late morning, sunny, easy-going brunch grooves",
    emoji="🥞🌤️",
    tags=("sunny", "breezy", "brunch"),
)

GOLDEN_HOUR = TimeOfDayPersona(
    name="Golden Hour Groove",
    sliders={
        "danceability":     70,
        "energy":           70,
        "valence":          75,
        "tempo":            60,
        "acousticness":     35,
        "instrumentalness": 10,
        "speechiness":      15,
    },
    tagline="feel-good pop and indie for golden hour walks",
    emoji="🌅✨",
    tags=("glowy & upbeat", "walk vibes", "feel-good"),
)

NIGHT_OUT = TimeOfDayPersona(
    name="Night Out Pre-Game",
    sliders={
        "danceability":     80,
        "energy":           85,
        "valence":          70,
        "tempo":            65,
        "acousticness":     10,
        "instrumentalness":  5,
        "speechiness":      20,
    },
    tagline="high-energy bangers to get you out the door",
    emoji="🥂💃",
    tags=("party", "high energy", "dancefloor"),
)

NEON_CITY = TimeOfDayPersona(
    name="Neon City Ride",
    sliders={
        "danceability":     75,
        "energy":           70,
        "valence":          55,
        "tempo":            60,
        "acousticness":     20,
        "instrumentalness": 25,
        "speechiness":      10,
    },
    tagline="late-night drive: pulsing, slightly moody, cinematic",
    emoji="🌃🚗",
    tags=("dark & wavy", "fast tempo", "night drive"),
)


CONTEXT_PERSONAS: Dict[Tuple[str, str], List[TimeOfDayPersona]] = {
    ("weekday", "morning"):    [SOFT_SUNRISE],
    ("weekday", "afternoon"):  [FLOW_STATE],
    ("weekday", "evening"):    [UNWIND_AFTER_HOURS],
    ("weekday", "late_night"): [MIDNIGHT_LOFI],

    ("weekend", "morning"):    [LAZY_BRUNCH],
    ("weekend", "afternoon"):  [GOLDEN_HOUR],
    ("weekend", "evening"):    [NIGHT_OUT],
    ("weekend", "late_night"): [NEON_CITY, MIDNIGHT_LOFI],
}



def _get_day_type(dt: datetime) -> str:
    return "weekend" if dt.weekday() >= 5 else "weekday"


def _get_time_bucket(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 11:
        return "morning"
    elif 11 <= h < 17:
        return "afternoon"
    elif 17 <= h < 22:
        return "evening"
    else:
        return "late_night"


class Mode2BVibeRoulette:

    def __init__(self, engine, rng_seed: int = 42):

        self.engine = engine
        self.df = engine.df
        self.rng = np.random.default_rng(rng_seed)
        self._last_meta: Optional[Dict[str, Any]] = None


    def _choose_persona_for_context(
        self,
        day_type: str,
        time_bucket: str,
    ) -> TimeOfDayPersona:

        key = (day_type, time_bucket)
        if key not in CONTEXT_PERSONAS:
            key = ("weekday", "afternoon")
        candidates = CONTEXT_PERSONAS[key]
        return random.choice(candidates)

    def _describe_sliders_short(self, sliders: Dict[str, float]) -> str:

        phrases = []

        energy = sliders.get("energy", 50)
        dance = sliders.get("danceability", 50)
        valence = sliders.get("valence", 50)
        tempo = sliders.get("tempo", 50)
        acousticness = sliders.get("acousticness", 50)
        instrumentalness = sliders.get("instrumentalness", 50)

        if energy >= 80 and tempo >= 70:
            phrases.append("sparkly & high-tempo")
        elif energy >= 70:
            phrases.append("high energy")
        elif energy <= 35:
            phrases.append("soft & low-key")

        if dance >= 75:
            phrases.append("very danceable")
        elif dance <= 35:
            phrases.append("more chill than dancey")

        if valence >= 70:
            phrases.append("glowy & upbeat")
        elif valence <= 35:
            phrases.append("moody / introspective")

        if tempo >= 80:
            phrases.append("fast tempo")
        elif tempo <= 40:
            phrases.append("slower, relaxed tempo")

        if acousticness >= 70:
            phrases.append("soft & cozy acoustic")
        elif acousticness <= 30:
            phrases.append("more electronic / synthetic")

        if instrumentalness >= 70:
            phrases.append("mostly instrumental")

        if not phrases:
            return "balanced vibe"

        uniq_phrases: List[str] = []
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

        if not hasattr(self.engine, "_sliders_to_target_vec"):
            return ""

        try:
            target_vec = self.engine._sliders_to_target_vec(persona.sliders)

            if now_playing.name not in self.df.index:
                return ""
            pos = self.df.index.get_loc(now_playing.name)
            track_vec = self.engine.X[pos]

            diff = np.abs(target_vec - track_vec)
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
            if not pretty_features:
                return ""

            return "This track lines up strongly on " + ", ".join(pretty_features) + "."
        except Exception:
            return ""

    def _build_persona_intro(
        self,
        meta: Dict[str, Any],
        now_playing: pd.Series,
    ) -> str:
        """Persona intro line using templates and context."""
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
        temperature: float = 0.7,
    ) -> Tuple[pd.Series, pd.DataFrame]:

        if recs.empty:
            raise RuntimeError("No recommendations available for this persona/context.")

        k = min(explore_k, len(recs))
        top = recs.iloc[:k].copy()

        if "vibe_score" not in top.columns:
            probs = np.ones(k) / k
        else:
            scores = top["vibe_score"].values.astype(float)
            scaled = scores / max(temperature, 1e-6)
            exps = np.exp(scaled - np.max(scaled))
            probs = exps / (exps.sum() + 1e-8)

        choice_idx = self.rng.choice(k, p=probs)
        now_playing = top.iloc[choice_idx]
        up_next = top.drop(index=now_playing.name)

        return now_playing, up_next


    def spin(
        self,
        top_k: int = 10,
        lambda_vibe: float = 0.8,
        hide_explicit: bool = True,
        min_popularity: Optional[int] = None,
        diversity: bool = True,
        diversity_threshold: float = 0.9,
        explore_k: int = 20,
        temperature: float = 0.7,
    ) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
        """
        Perform one Vibe Roulette spin.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        dt = datetime.now()
        day_type = _get_day_type(dt)
        time_bucket = _get_time_bucket(dt)

        persona = self._choose_persona_for_context(day_type, time_bucket)

        candidate_k = max(top_k, explore_k)

        recs = self.engine.recommend_by_sliders(
            sliders=persona.sliders,
            top_k=candidate_k,
            lambda_vibe=lambda_vibe,
            hide_explicit=hide_explicit,
            min_popularity=min_popularity,
            diversity=diversity,
            diversity_threshold=diversity_threshold,
        )

        if recs.empty:
            raise RuntimeError("Vibe Roulette produced no recommendations. Check filters / data.")

        now_playing, up_next = self._choose_now_playing_with_exploration(
            recs, explore_k=explore_k, temperature=temperature
        )

        up_next = up_next.iloc[: max(0, top_k - 1)].copy()

        vibe_summary = self._describe_sliders_short(persona.sliders)

        meta: Dict[str, Any] = {
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

        self._last_meta = meta
        return now_playing, up_next, meta

    def print_spin_result(
        self,
        now_playing: pd.Series,
        up_next: pd.DataFrame,
        meta: Dict[str, Any],
        show_story: bool = True,
        show_explanation: bool = True,
    ) -> None:
        """
        Pretty-print the Vibe Roulette result in a notebook.
        """
        try:
            from IPython.display import display, IFrame
            have_ipython = True
        except ImportError:
            have_ipython = False
            display = None
            IFrame = None

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

            tag_str = " · ".join(persona_tags) if persona_tags else vibe_summary
            print(f"{persona_emoji}  Persona: **{persona_name}**")
            if tagline:
                print(f"   {tagline}")
            if tag_str:
                print(f"   {tag_str}")
            print()

            if previous_meta is not None:
                prev_name = previous_meta.get("persona_name", "a different vibe")
                prev_bucket = previous_meta.get("time_bucket", "").replace("_", " ")
                print(
                    f"Last spin felt like *{prev_name}* ({prev_bucket}). "
                    f"Today we’re shifting into *{persona_name}*."
                )
                print()

            intro = self._build_persona_intro(meta, now_playing)
            print(intro)
            print()

        print(f"🎧 Now playing: {track_name} — {artist}")
        if isinstance(track_id, str) and track_id:
            spotify_url = f"https://open.spotify.com/track/{track_id}"
            print(f"🔗 Open in Spotify: {spotify_url}")

            if have_ipython and IFrame is not None:
                embed_url = f"https://open.spotify.com/embed/track/{track_id}"
                print("\n▶️ Embedded player:")
                display(IFrame(src=embed_url, width=320, height=80))

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
                print(up_next[["track_name", "artists"]].head())
        else:
            print("\n(No additional tracks in queue.)")

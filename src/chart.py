import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIBE_COLS = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness']
class MockPersona:
    def __init__(self, sliders): self.sliders = sliders
SOFT_SUNRISE = MockPersona({'danceability': 40, 'energy': 30, 'tempo': 20, 'valence': 60, 'acousticness': 80, 'instrumentalness': 10})
NIGHT_OUT = MockPersona({'danceability': 90, 'energy': 95, 'tempo': 85, 'valence': 70, 'acousticness': 10, 'instrumentalness': 0})

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def persona_to_series(persona, tempo_min=0.0, tempo_max=243.372):
    vals = []
    for col in VIBE_COLS:
        s = persona.sliders[col]
        if col == "tempo":
            tempo_val = tempo_min + (s / 100.0) * (tempo_max - tempo_min)
            tempo_norm = (tempo_val - tempo_min) / (tempo_max - tempo_min)
            vals.append(tempo_norm)
        else:
            vals.append(s / 100.0)
    return pd.Series(vals, index=VIBE_COLS)

tempo_min = 0.0
tempo_max = 243.372

p1 = persona_to_series(SOFT_SUNRISE, tempo_min, tempo_max)
p2 = persona_to_series(NIGHT_OUT, tempo_min, tempo_max)

df_radar = pd.DataFrame({
    "Soft Sunrise": p1,
    "Night Out": p2,
})

BG_COLOR = "#050509"
GRID_COLOR = "#2a2e40"
TEXT_COLOR = "#E0E0E0"
COLOR_P1 = "#FF9F1C"
COLOR_P2 = "#4CC9F0"

labels = [l.upper() for l in VIBE_COLS]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def make_values(series):
    vals = series.values.tolist()
    vals += vals[:1]
    return vals

values_p1 = make_values(df_radar["Soft Sunrise"])
values_p2 = make_values(df_radar["Night Out"])

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

ax.set_facecolor(BG_COLOR)
plt.gcf().patch.set_facecolor(BG_COLOR)

ax.spines['polar'].set_visible(False)

ax.grid(color=GRID_COLOR, linewidth=1.5, linestyle='-')

plt.xticks(angles[:-1], labels, color=TEXT_COLOR, size=10, weight='bold')

ax.set_rlabel_position(0)

plt.yticks([0.25, 0.50, 0.75, 1.0], [], color="grey", size=7) 
plt.ylim(0, 1.05)

ax.plot(angles, values_p1, linewidth=3, linestyle='-', color=COLOR_P1, label="Soft Sunrise", zorder=10)
ax.fill(angles, values_p1, color=COLOR_P1, alpha=0.25) 

ax.scatter(angles, values_p1, s=60, c=COLOR_P1, zorder=11, edgecolors=BG_COLOR, linewidth=1.5)

ax.plot(angles, values_p2, linewidth=3, linestyle='-', color=COLOR_P2, label="Night Out", zorder=10)
ax.fill(angles, values_p2, color=COLOR_P2, alpha=0.25)
ax.scatter(angles, values_p2, s=60, c=COLOR_P2, zorder=11, edgecolors=BG_COLOR, linewidth=1.5)

plt.title("VIBE PERSONA FINGERPRINT", color='white', size=16, weight='bold', pad=30)

legend = plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=11)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

plt.tight_layout()
plt.savefig("assets/mode1_vibe_radar1.png", dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
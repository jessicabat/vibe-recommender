import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- KEEP YOUR ORIGINAL IMPORTS ---
# from mode2b_vibe_roulette import SOFT_SUNRISE, NIGHT_OUT
# from vibe_engine import VIBE_COLS 

# --- (MOCK DATA FOR DEMONSTRATION - DELETE THIS BLOCK IN YOUR CODE) ---
VIBE_COLS = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness']
class MockPersona:
    def __init__(self, sliders): self.sliders = sliders
SOFT_SUNRISE = MockPersona({'danceability': 40, 'energy': 30, 'tempo': 20, 'valence': 60, 'acousticness': 80, 'instrumentalness': 10})
NIGHT_OUT = MockPersona({'danceability': 90, 'energy': 95, 'tempo': 85, 'valence': 70, 'acousticness': 10, 'instrumentalness': 0})
# ----------------------------------------------------------------------

# --- 0. Global Style Settings ---
# This makes fonts look cleaner and modern
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 1. Build Dataframe (Your original logic) ---
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
    "Soft Sunrise": p1, # Shortened names for cleaner legend
    "Night Out": p2,
})

# --- 2. Plotting Setup ---

# Define Colors (Neon Palette)
BG_COLOR = "#050509"
GRID_COLOR = "#2a2e40" # Slightly lighter than BG
TEXT_COLOR = "#E0E0E0"
COLOR_P1 = "#FF9F1C"   # Vibrant Orange (Sunrise)
COLOR_P2 = "#4CC9F0"   # Electric Cyan (Night Out)

labels = [l.upper() for l in VIBE_COLS] # Uppercase looks more "Dashboard"
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def make_values(series):
    vals = series.values.tolist()
    vals += vals[:1]
    return vals

values_p1 = make_values(df_radar["Soft Sunrise"])
values_p2 = make_values(df_radar["Night Out"])

# Create Figure
plt.figure(figsize=(8, 8)) # Increased size slightly for quality
ax = plt.subplot(111, polar=True)

# Set Backgrounds
ax.set_facecolor(BG_COLOR)
plt.gcf().patch.set_facecolor(BG_COLOR)

# --- 3. Customizing the Grid (The "Tech" Look) ---
# Remove the outer circle frame (spine)
ax.spines['polar'].set_visible(False)

# Custom grid lines
ax.grid(color=GRID_COLOR, linewidth=1.5, linestyle='-')

# X-Axis (The Labels)
# Add padding so labels don't overlap the chart
plt.xticks(angles[:-1], labels, color=TEXT_COLOR, size=10, weight='bold')

# Y-Axis (The Concentric Circles)
ax.set_rlabel_position(0)
# Hide the standard numbers (0.2, 0.4 etc) as they look messy. 
# We assume the viewer understands the relative scale (0 to 1).
plt.yticks([0.25, 0.50, 0.75, 1.0], [], color="grey", size=7) 
plt.ylim(0, 1.05) # Give a little headroom

# --- 4. Plotting the Data ---

# Plot P1: Soft Sunrise
ax.plot(angles, values_p1, linewidth=3, linestyle='-', color=COLOR_P1, label="Soft Sunrise", zorder=10)
ax.fill(angles, values_p1, color=COLOR_P1, alpha=0.25) 
# Add markers at the points
ax.scatter(angles, values_p1, s=60, c=COLOR_P1, zorder=11, edgecolors=BG_COLOR, linewidth=1.5)

# Plot P2: Night Out
ax.plot(angles, values_p2, linewidth=3, linestyle='-', color=COLOR_P2, label="Night Out", zorder=10)
ax.fill(angles, values_p2, color=COLOR_P2, alpha=0.25)
ax.scatter(angles, values_p2, s=60, c=COLOR_P2, zorder=11, edgecolors=BG_COLOR, linewidth=1.5)

# --- 5. Title and Legend ---

# Add a Title with padding
plt.title("VIBE PERSONA FINGERPRINT", color='white', size=16, weight='bold', pad=30)

# Custom Legend
legend = plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=11)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

plt.tight_layout()
plt.savefig("assets/mode1_vibe_radar1.png", dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")

COLORS = {
    "collection":  "#2980b9",
    "ingestion":   "#8e44ad",
    "storage":     "#27ae60",
    "processing":  "#e67e22",
    "analytics":   "#c0392b",
    "header":      "#2c3e50",
    "arrow":       "#7f8c8d",
}

def draw_stage(ax, x, y, w, h, title, items, color, icon):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor="white",
                                linewidth=2, zorder=3))

    ax.text(x + w/2, y + h - 0.45, f"{icon}  {title}",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=4)

    ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.75, y + h - 0.75],
            color="white", alpha=0.5, linewidth=1, zorder=4)

    for i, item in enumerate(items):
        iy = y + h - 1.05 - i * 0.52
        ax.add_patch(FancyBboxPatch((x + 0.15, iy - 0.2), w - 0.3, 0.42,
                                    boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor="none",
                                    alpha=0.25, zorder=4))
        ax.text(x + w/2, iy, item,
                ha="center", va="center", fontsize=8.5,
                color="white", zorder=5)

def draw_arrow(ax, x1, x2, y):
    ax.annotate("",
                xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["arrow"],
                                lw=2.5, mutation_scale=20),
                zorder=2)

ax.text(9, 9.5, "Truth Seeker - Big Data Architecture",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=COLORS["header"])
ax.text(9, 9.1, "Konseptual Big Data Sistemi",
        ha="center", va="center", fontsize=10,
        color="#7f8c8d", style="italic")

stage_w = 3.0
stage_h = 7.5
gap     = 0.35
start_x = 0.25
stage_y = 1.0

stages = [
    ("Data Collection",  ["Twitter API", "Politifact.com", "Fact-check sites", "RSS Feeds"],       COLORS["collection"],  "01"),
    ("Data Ingestion",   ["Apache Kafka", "Message Queue", "Batch Import", "Data Validation"],      COLORS["ingestion"],   "02"),
    ("Storage",          ["HDFS / S3", "Data Lake", ".csv / .xlsx", "Raw + Processed"],             COLORS["storage"],     "03"),
    ("Processing",       ["Apache Spark", "Preprocessing", "Feature Engineering", "Normalization"], COLORS["processing"],  "04"),
    ("Analytics",        ["ML Models", "Logistic Reg.", "Random Forest", "XGBoost"],                COLORS["analytics"],   "05"),
]

boxes_x = []
for i, (title, items, color, icon) in enumerate(stages):
    x = start_x + i * (stage_w + gap)
    boxes_x.append(x)
    draw_stage(ax, x, stage_y, stage_w, stage_h, title, items, color, icon)

arrow_y = stage_y + stage_h / 2
for i in range(len(boxes_x) - 1):
    draw_arrow(ax, boxes_x[i] + stage_w, boxes_x[i + 1], arrow_y)

# bottom legend
legend_items = [
    ("Data Collection",  COLORS["collection"]),
    ("Data Ingestion",   COLORS["ingestion"]),
    ("Storage",          COLORS["storage"]),
    ("Processing",       COLORS["processing"]),
    ("Analytics",        COLORS["analytics"]),
]
lx = 1.5
for label, color in legend_items:
    ax.add_patch(FancyBboxPatch((lx - 0.15, 0.25), 0.25, 0.25,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="none"))
    ax.text(lx + 0.2, 0.37, label, fontsize=8, va="center", color=COLORS["header"])
    lx += 3.35

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "architecture_diagram.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Diagram saved: output/architecture_diagram.png")

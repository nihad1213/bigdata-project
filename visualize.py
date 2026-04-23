import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAIN_PREPROCESSED = os.path.join(OUTPUT_DIR, "main_preprocessed.csv")
FEAT_PREPROCESSED = os.path.join(OUTPUT_DIR, "features_preprocessed.csv")

def generate_visualizations():
    print("Loading preprocessed data for visualization...")
    if not os.path.exists(MAIN_PREPROCESSED) or not os.path.exists(FEAT_PREPROCESSED):
        print("Error: Preprocessed files not found. Run preprocessing.py first!")
        return

    df_main = pd.read_csv(MAIN_PREPROCESSED)
    df_feat = pd.read_csv(FEAT_PREPROCESSED)

    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Data Preprocessing — Insights Overview", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    counts = df_main["BinaryNumTarget"].value_counts().sort_index()
    ax1.bar(["False (0)", "True (1)"], counts.values, color=["#e74c3c", "#2ecc71"])
    ax1.set_title("Target Class Distribution")
    ax1.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + (max(counts.values)*0.02), f"{v:,}", ha="center", fontsize=10)

    ax2 = axes[0, 1]
    ax2.hist(df_feat["engagement_score"], bins=50, color="#3498db", edgecolor="white")
    ax2.set_title("Engagement Score Distribution (Normalized)")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Frequency")

    ax3 = axes[1, 0]
    if "BotScore" in df_feat.columns:
        ax3.hist(df_feat["BotScore"], bins=50, color="#9b59b6", edgecolor="white")
        ax3.set_title("Bot Score Distribution (Normalized)")
        ax3.set_xlabel("Score")
    else:
        ax3.text(0.5, 0.5, "BotScore column missing", ha='center')

    ax4 = axes[1, 1]
    if "ner_diversity" in df_feat.columns:
        ner_counts = df_feat["ner_diversity"].value_counts().sort_index()
        ax4.bar(ner_counts.index.astype(str), ner_counts.values, color="#e67e22")
        ax4.set_title("NER Diversity (Entity Types per Tweet)")
        ax4.set_xlabel("Count of Distinct Entities")
    else:
        ax4.text(0.5, 0.5, "ner_diversity column missing", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, "preprocessing_overview.png")
    plt.savefig(save_path, dpi=150)
    print(f"Success: Plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_visualizations()
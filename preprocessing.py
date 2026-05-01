import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# file paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MAIN_CSV = os.path.join(DATASET_DIR, "Truth_Seeker_Model_Dataset.csv")
FEAT_CSV = os.path.join(DATASET_DIR, "Features_For_Traditional_ML_Techniques.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading datasets...")
df_main = pd.read_csv(MAIN_CSV)
df_feat = pd.read_csv(FEAT_CSV)
print(f"  Main dataset  : {df_main.shape[0]:,} rows x {df_main.shape[1]} columns")
print(f"  Features dataset : {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")

# check missing values
missing_main = df_main.isnull().sum()
missing_feat = df_feat.isnull().sum()

print("\n[Main dataset - missing values]")
if missing_main.sum() == 0:
    print("  No missing values found.")
else:
    print(missing_main[missing_main > 0])

print("\n[Features dataset - missing values]")
if missing_feat.sum() == 0:
    print("  No missing values found.")
else:
    cols_with_nulls = missing_feat[missing_feat > 0]
    print(cols_with_nulls)

# fill missing numeric values with median, text columns with mode
numeric_cols = df_feat.select_dtypes(include="number").columns
for col in numeric_cols:
    if df_feat[col].isnull().sum() > 0:
        median_val = df_feat[col].median()
        df_feat[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median: {median_val:.4f}")

cat_cols = df_feat.select_dtypes(include="object").columns
for col in cat_cols:
    if df_feat[col].isnull().sum() > 0:
        mode_val = df_feat[col].mode()[0]
        df_feat[col].fillna(mode_val, inplace=True)
        print(f"  Filled '{col}' with mode: {mode_val}")

print(f"\n  After cleaning - missing in main dataset: {df_main.isnull().sum().sum()}")
print(f"  After cleaning - missing in features dataset: {df_feat.isnull().sum().sum()}")

# new features
df_main["statement_length"] = df_main["statement"].apply(len)
df_main["tweet_length"] = df_main["tweet"].apply(len)
df_main["keyword_count"] = df_main["manual_keywords"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) else 0
)

# total interactions on a tweet
engagement_cols = ["retweets", "replies", "quotes", "mentions", "favourites"]
available = [c for c in engagement_cols if c in df_feat.columns]
df_feat["engagement_score"] = df_feat[available].sum(axis=1)

# flag accounts with BotScore > 0.5 as bot
if "BotScore" in df_feat.columns:
    df_feat["is_bot"] = (df_feat["BotScore"] > 0.5).astype(int)
    bot_count = df_feat["is_bot"].sum()
    print(f"  is_bot: {bot_count:,} accounts flagged")

# count how many different entity types exist in each tweet
ner_cols = [c for c in df_feat.columns if c.endswith("_percentage")]
df_feat["ner_diversity"] = (df_feat[ner_cols] > 0).sum(axis=1)

print("  New features added: statement_length, tweet_length, keyword_count, engagement_score, is_bot, ner_diversity")

# scale everything to 0-1 range
cols_to_normalize = [
    "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "engagement_score",
    "BotScore", "cred", "normalize_influence"
]
cols_to_normalize = [c for c in cols_to_normalize if c in df_feat.columns]
scaler = MinMaxScaler()
df_feat[cols_to_normalize] = scaler.fit_transform(df_feat[cols_to_normalize])

text_len_cols = ["statement_length", "tweet_length", "keyword_count"]
scaler2 = MinMaxScaler()
df_main[text_len_cols] = scaler2.fit_transform(df_main[text_len_cols])
print(f"  Min-Max scaling done on {len(cols_to_normalize) + 3} columns")

# convert text labels to numbers so models can read them
le = LabelEncoder()
df_main["label_3_encoded"] = le.fit_transform(df_main["3_label_majority_answer"])
classes_3 = list(le.classes_)
print(f"  3-class encoding: {dict(zip(classes_3, le.transform(classes_3)))}")

df_main["label_5_encoded"] = le.fit_transform(df_main["5_label_majority_answer"])
classes_5 = list(le.classes_)
print(f"  5-class encoding: {dict(zip(classes_5, le.transform(classes_5)))}")

# retweets etc. are very skewed, log helps flatten the distribution
df_feat_raw = pd.read_csv(FEAT_CSV, index_col=0)
skewed_cols = ["retweets", "replies", "quotes", "favourites"]
skewed_cols = [c for c in skewed_cols if c in df_feat_raw.columns]

print("\n  Log1p on skewed columns:")
for col in skewed_cols:
    before = df_feat_raw[col].skew()
    df_feat[col + "_log"] = np.log1p(df_feat_raw[col].clip(lower=0))
    after = df_feat[col + "_log"].skew()
    print(f"    {col}: {before:.1f} -> {after:.2f}")

# embeddings column is raw vector, not useful for our models
drop_cols = ["embeddings"]
drop_cols = [c for c in drop_cols if c in df_feat.columns]
if drop_cols:
    df_feat.drop(columns=drop_cols, inplace=True)
    print(f"  Dropped: {drop_cols}")

# save cleaned datasets
main_out = os.path.join(OUTPUT_DIR, "main_preprocessed.csv")
feat_out = os.path.join(OUTPUT_DIR, "features_preprocessed.csv")
df_main.to_csv(main_out)
df_feat.to_csv(feat_out)
print("\n  Saved: main_preprocessed.csv")
print("  Saved: features_preprocessed.csv")

# plots
import matplotlib
matplotlib.use("Agg")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Data Preprocessing - Insights Overview", fontsize=14, fontweight="bold")

counts = df_main["BinaryNumTarget"].value_counts().sort_index()
axes[0, 0].bar(["False (0)", "True (1)"], counts.values, color=["#e74c3c", "#2ecc71"])
axes[0, 0].set_title("Target Class Distribution")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0, 0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)

axes[0, 1].hist(df_feat["engagement_score"], bins=50, color="#3498db", edgecolor="white")
axes[0, 1].set_title("Engagement Score Distribution (Normalized)")
axes[0, 1].set_xlabel("Score")

axes[1, 0].hist(df_feat["BotScore"], bins=50, color="#9b59b6", edgecolor="white")
axes[1, 0].set_title("Bot Score Distribution (Normalized)")
axes[1, 0].set_xlabel("Score")

ner_counts = df_feat["ner_diversity"].value_counts().sort_index()
axes[1, 1].bar(ner_counts.index.astype(str), ner_counts.values, color="#e67e22")
axes[1, 1].set_title("NER Diversity (Entity Types per Tweet)")
axes[1, 1].set_xlabel("Count of Distinct Entities")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "preprocessing_overview.png"), dpi=150)
plt.close()
print("  Saved: preprocessing_overview.png")

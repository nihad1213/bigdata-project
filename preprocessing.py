import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define paths of Dataset
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MAIN_CSV = os.path.join(DATASET_DIR, "Truth_Seeker_Model_Dataset.csv")
FEAT_CSV = os.path.join(DATASET_DIR, "Features_For_Traditional_ML_Techniques.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loading dataset show column and row count
print("Loading Dataset started")

df_main = pd.read_csv(MAIN_CSV)
df_feat = pd.read_csv(FEAT_CSV)
print(f"  Main dataset  : {df_main.shape[0]:,} rows x {df_main.shape[1]} columns")
print(f"  Features dataset : {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")

# Missing Values Handling
missing_main = df_main.isnull().sum()
missing_feat = df_feat.isnull().sum()

print("\n[Main dataset — missing values per column]")

if missing_main.sum() == 0:
    print("  No missing values found.")
else:
    print(missing_main[missing_main > 0])


print("\n[Features dataset — missing values per column]")

if missing_feat.sum() == 0:
    print("  No missing values found.")

else:
    cols_with_nulls = missing_feat[missing_feat > 0]
    print(cols_with_nulls)

# No missing values found in datasets. But due to best practice we will keep that part of code.

# Filling numeric columns
numeric_cols = df_feat.select_dtypes(include="number").columns
for col in numeric_cols:
    if df_feat[col].isnull().sum() > 0:
        median_val = df_feat[col].median()
        df_feat[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median: {median_val:.4f}")



# Filling categorical columns
cat_cols = df_feat.select_dtypes(include="object").columns
for col in cat_cols:
    if df_feat[col].isnull().sum() > 0:
        mode_val = df_feat[col].mode()[0]
        df_feat[col].fillna(mode_val, inplace=True)
        print(f"  Filled '{col}' with mode: {mode_val}")

print(f"\n  After cleaning — missing in main dataset: {df_main.isnull().sum().sum()}")
print(f"\n  After cleaning — missing in features dataset: {df_feat.isnull().sum().sum()}")


# Feature Engineering
# statement_lenght column
df_main["statement_length"] = df_main["statement"].apply(len)
print("  [+] statement_length — character count of each claim")

# tweet_length column
df_main["tweet_length"] = df_main["tweet"].apply(len)
print("  [+] tweet_length — character count of each tweet")

# keyword_count column
df_main["keyword_count"] = df_main["manual_keywords"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) else 0
)
print("  [+] keyword_count — number of manual keywords per statement")

# engagement_score column
engagement_cols = ["retweets", "replies", "quotes", "mentions", "favourites"]
available = [c for c in engagement_cols if c in df_feat.columns]
df_feat["engagement_score"] = df_feat[available].sum(axis=1)
print(f"  [+] engagement_score — sum of: {', '.join(available)}")

# is_bot column
if "BotScore" in df_feat.columns:
    df_feat["is_bot"] = (df_feat["BotScore"] > 0.5).astype(int)
    bot_count = df_feat["is_bot"].sum()
    print(f"  [+] is_bot — flagged {bot_count:,} accounts as likely bots (BotScore > 0.5)")

# NER diversity — how many entity types appear in a tweet (percentage > 0)
ner_cols = [c for c in df_feat.columns if c.endswith("_percentage")]
df_feat["ner_diversity"] = (df_feat[ner_cols] > 0).sum(axis=1)
print(f"  [+] ner_diversity — number of distinct NER categories present per tweet")


print(f"\n  New features added to main dataset  : statement_length, tweet_length, keyword_count")
print(f"  New features added to features dataset: engagement_score, is_bot, ner_diversity")


# Data Normalization

# Columns to normalize — social media stats and engagement
cols_to_normalize = [
    "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "engagement_score",
    "BotScore", "cred", "normalize_influence"

]

cols_to_normalize = [c for c in cols_to_normalize if c in df_feat.columns]
scaler = MinMaxScaler()
df_feat[cols_to_normalize] = scaler.fit_transform(df_feat[cols_to_normalize])
print(f"  Applied Min-Max scaling to {len(cols_to_normalize)} columns:")

for col in cols_to_normalize:
    print(f"    - {col}  ->  min={df_feat[col].min():.4f}, max={df_feat[col].max():.4f}")

# Also normalize engineered text-length features
text_len_cols = ["statement_length", "tweet_length", "keyword_count"]
scaler2 = MinMaxScaler()
df_main[text_len_cols] = scaler2.fit_transform(df_main[text_len_cols])
print(f"\n  Also normalized: {', '.join(text_len_cols)}")
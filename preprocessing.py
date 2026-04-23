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
    print(missing_feat[missing_feat > 0])

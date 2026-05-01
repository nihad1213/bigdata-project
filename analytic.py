import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FEAT_CSV    = os.path.join(DATASET_DIR, "Features_For_Traditional_ML_Techniques.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(FEAT_CSV, index_col=0)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# drop text columns, we only use numeric features for these models
drop_cols = ["majority_target", "statement", "tweet", "embeddings"]
drop_cols  = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

TARGET = "BinaryNumTarget"
df[TARGET] = df[TARGET].astype(float).astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.select_dtypes(include="number")
X = X.fillna(X.median())

print(f"  Features used : {X.shape[1]}")
print(f"  Class distribution -> 0: {(y==0).sum():,}  |  1: {(y==1).sum():,}")

# 80% train, 20% test, stratified to keep class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train size : {len(X_train):,}")
print(f"  Test size  : {len(X_test):,}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric="logloss", verbosity=0),
}

results = {}

for name, model in models.items():
    print(f"\n[{name}]")
    print("  Training...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

print("\nRESULTS COMPARISON")
df_results = pd.DataFrame(results).T
print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))

best_model_name = df_results["F1"].idxmax()
print(f"\n  Best model (F1): {best_model_name} -> {df_results.loc[best_model_name, 'F1']:.4f}")

best_model  = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# feature importance from tree-based models
rf_model  = models["Random Forest"]
xgb_model = models["XGBoost"]

rf_importance  = pd.Series(rf_model.feature_importances_,  index=X.columns)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)

top_rf  = rf_importance.nlargest(10)
top_xgb = xgb_importance.nlargest(10)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("ML Model Analysis - Truth Seeker Dataset", fontsize=14, fontweight="bold")

# model scores side by side
ax1 = axes[0, 0]
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics))
width = 0.25
colors = ["#3498db", "#2ecc71", "#e74c3c"]
for i, (name, vals) in enumerate(results.items()):
    ax1.bar(x + i * width, [vals[m] for m in metrics], width, label=name, color=colors[i])
ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics)
ax1.set_ylim(0.5, 1.0)
ax1.set_title("Model Comparison")
ax1.set_ylabel("Score")
ax1.legend(fontsize=8)

# where the model got it right and wrong
ax2 = axes[0, 1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False (0)", "True (1)"])
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"Confusion Matrix - {best_model_name}")

# which features mattered most
ax3 = axes[1, 0]
top_rf.sort_values().plot(kind="barh", ax=ax3, color="#9b59b6")
ax3.set_title("Top 10 Features - Random Forest")
ax3.set_xlabel("Importance")

ax4 = axes[1, 1]
top_xgb.sort_values().plot(kind="barh", ax=ax4, color="#e67e22")
ax4.set_title("Top 10 Features - XGBoost")
ax4.set_xlabel("Importance")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "model_results.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print("\nPlot saved: output/model_results.png")

print(f"\nCLASSIFICATION REPORT - {best_model_name}")
print(classification_report(y_test, y_pred_best, target_names=["False", "True"]))

print("Analysis complete.")

# Fake News Detection Using Machine Learning — Truth Seeker Dataset

**Azerbaijan Technical University — Big Data Technologies**  
**Authors:** Nihad Namatli, Elnur Calalov  
**Group:** M655aS

---

## Project Overview

This project applies Big Data and machine learning techniques to detect misinformation on social media. Using the Truth Seeker dataset, we train and compare three classification models (Logistic Regression, Random Forest, XGBoost) to determine whether a given statement is true or false based on its content and related Twitter activity.

---

## Dataset

| File | Description | Size |
|---|---|---|
| `Truth_Seeker_Model_Dataset.csv` | Main dataset with statements, tweets and labels | 49.93 MB |
| `Features_For_Traditional_ML_Techniques.csv` | Pre-extracted numerical features for ML | 78.56 MB |
| `Truth_Seeker_Model_Dataset_With_TimeStamps 1.xlsx` | Dataset with timestamps for time-series analysis | 18.69 MB |

- **Rows:** 134,198
- **Source:** [UNB CIC TruthSeeker 2023](https://www.unb.ca/cic/datasets/truthseeker-2023.html)
- **Labels:** Binary (True/False), 3-class, 5-class

---

## Requirements

**Python 3.13** is required (Python 3.14 is not yet supported by pandas/numpy).

Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas==2.3.0
numpy==2.3.0
scikit-learn==1.6.1
matplotlib==3.10.3
seaborn==0.13.2
openpyxl==3.1.5
scipy==1.15.3
imbalanced-learn==0.13.0
xgboost==3.2.0
```

---

## How to Run

Run the full pipeline with a single command:

```bash
py -3.13 main.py
```

This executes all steps in sequence:

| Step | Description |
|---|---|
| 1 | Load datasets |
| 2 | Check and handle missing values |
| 3 | Feature engineering (6 new features) |
| 4 | Min-Max normalization |
| 5 | Data transformation (label encoding, log1p) |
| 6 | Generate preprocessing charts |
| 7 | Train and evaluate ML models |
| 8 | Generate model result charts |
| 9 | Generate Big Data architecture diagram |

---

## Output Files

All results are saved to the `output/` folder:

| File | Description |
|---|---|
| `main_preprocessed.csv` | Cleaned main dataset with engineered features |
| `features_preprocessed.csv` | Normalized and transformed features dataset |
| `preprocessing_overview.png` | 4-panel chart: class distribution, engagement score, bot score, NER diversity |
| `model_results.png` | Model comparison bar chart, confusion matrix, feature importance (RF & XGBoost) |
| `architecture_diagram.png` | Conceptual Big Data architecture (Collection → Ingestion → Storage → Processing → Analytics) |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.5812 | 0.5841 | 0.6410 | 0.6113 |
| Random Forest | 0.6947 | 0.7110 | 0.6834 | 0.6969 |
| **XGBoost** | **0.7108** | **0.7237** | **0.7069** | **0.7152** |

**Best model: XGBoost** with F1 = 0.7152

---

## Project Structure

```
bigdata-project/
├── dataset/
│   ├── Truth_Seeker_Model_Dataset.csv
│   ├── Features_For_Traditional_ML_Techniques.csv
│   └── Truth_Seeker_Model_Dataset_With_TimeStamps 1.xlsx
├── output/
│   ├── main_preprocessed.csv
│   ├── features_preprocessed.csv
│   ├── preprocessing_overview.png
│   ├── model_results.png
│   └── architecture_diagram.png
├── main.py
├── requirements.txt
└── README.md
```

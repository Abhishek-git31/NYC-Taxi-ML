# Predicting Tip Behavior in NYC Yellow Taxi Trips
### Using Supervised Machine Learning

**Course:** PS26 - DSC01 ML-90 | UE Potsdam — Master of Data Science  
**Assignment:** Supervised Learning — Submission 1  
**Task:** Binary Classification (`is_tipped`: 1 if tip_amount > 0, else 0)

---

## Dataset

| Field | Details |
|-------|---------|
| **Dataset** | NYC Yellow Taxi Trip Records — January 2025 |
| **Source** | [NYC TLC Official Portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Direct Download** | [yellow_tripdata_2025-01.parquet](https://d37ci6vzurychx.cloudfront.net/tripdata/yellow_tripdata_2025-01.parquet) |
| **Rows Used** | 10,000 (sampled from full dataset) |
| **Columns** | 20 (original) → 10 features used |
| **Target Variable** | `is_tipped` — 1 if tip_amount > 0, else 0 |
| **Publisher** | NYC Taxi & Limousine Commission (TLC) |

---

## Research Questions

| Notebook | Research Question |
|----------|-------------------|
| `RQ1_Baseline_Performance.ipynb` | How effectively do baseline models (Logistic Regression, Decision Tree, k-NN) predict tip behavior? |
| `RQ2_Model_Comparison.ipynb` | Which of the 6 candidate models achieves the best predictive performance? |
| `RQ3_Preprocessing_Effects.ipynb` | How do preprocessing strategies (imputation, scaling, SMOTE) affect performance? |
| `RQ4_Feature_Importance.ipynb` | Which features most influence tip prediction? (XGBoost + SHAP) |
| `RQ5_Metric_Sensitivity.ipynb` | How do model rankings change across different evaluation metrics? |
| `RQ6_Robustness.ipynb` | How robust is XGBoost under CV, noise injection, and artificial missingness? |
| `RQ7_Final_Recommendation.ipynb` | Which model is most practically useful for real-world deployment? |

---

## Models Used

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (k-NN)
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score, AUC-ROC
- Confusion Matrix
- SHAP feature importance values
- Cross-validation stability (std dev)

---

## Project Structure

```
nyc_taxi_ml/
├── notebooks/
│   ├── RQ1_Baseline_Performance.ipynb
│   ├── RQ2_Model_Comparison.ipynb
│   ├── RQ3_Preprocessing_Effects.ipynb
│   ├── RQ4_Feature_Importance.ipynb
│   ├── RQ5_Metric_Sensitivity.ipynb
│   ├── RQ6_Robustness.ipynb
│   └── RQ7_Final_Recommendation.ipynb
├── figures/                  # Generated PDF figures (after running notebooks)
├── results/                  # Generated CSV result tables (after running notebooks)
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nyc-taxi-ml.git
cd nyc-taxi-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
Each notebook is self-contained and downloads the dataset automatically.

```bash
jupyter notebook
```

Open any notebook from the `notebooks/` folder and run all cells (Kernel → Restart & Run All).

> **Note:** The dataset is downloaded directly from the NYC TLC server inside each notebook. No manual download is needed. An internet connection is required.

### 4. Outputs
- All figures are saved as PDF in `figures/`
- All result tables are saved as CSV in `results/`

---

## Key Results Summary

| Model | Accuracy | F1-score | AUC |
|-------|----------|----------|-----|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| k-NN | — | — | — |
| Random Forest | — | — | — |
| **XGBoost** | — | — | — |
| SVM | — | — | — |

> _Actual results will appear here after notebooks are executed._

---

## Dependencies

See `requirements.txt`. Key libraries:

```
pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, matplotlib, seaborn, pyarrow
```

---

## Dataset Reference

NYC Taxi and Limousine Commission. (2025). *NYC Yellow Taxi Trip Records — January 2025* [Dataset]. City of New York.  
Available at: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page  
To download the actual parquet file yourself ,you go to that page and click the January 2025 yellow taxi link under 2026 drop down

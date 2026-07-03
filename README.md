# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions. The primary challenge in this dataset is extreme class imbalance, which is addressed here using SMOTE (Synthetic Minority Over-sampling Technique) before training a Random Forest classifier.

## Dataset
Sourced from the Kaggle [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **Total Records:** 284,807 transactions
- **Features:** 31 (Time, Amount, V1-V28, and Class)
- **Data Quality:** 0 missing or null values
- **Original Distribution:** 284,315 normal (Class 0) vs. 492 fraudulent (Class 1)

## Pipeline Overview

1. **Exploratory Data Analysis (EDA):** Analyzed transaction distributions over time and plotted feature correlation heatmaps to understand underlying patterns.
2. **Data Balancing:** Applied SMOTE to synthesize new fraudulent examples. Both classes were balanced to exactly 284,315 samples each to prevent the model from biasing toward the majority class.
3. **Train/Test Split:** Standard 80/20 split (`test_size=0.2`).
4. **Model Training:** Built and trained a `RandomForestClassifier` with constraints (`n_estimators=100`, `max_depth=3`, `random_state=42`) to maintain generalization.

## Model Evaluation

The model was evaluated on the 20% test holdout split. 

| Metric | Score |
|---|---|
| **Accuracy** | 0.9618 |
| **Precision** | 0.9945 |
| **Recall** | 0.9289 |
| **F1 Score** | 0.9606 |

*(Note: Training metrics showed Accuracy ~0.9610 and F1 ~0.9596. The tight alignment between train and test scores indicates the model is not overfitting).*

## Saved Model
The trained classifier is exported and ready for inference. 
- **File:** `Creadit_Card_Fraud_Detection_RFmodel.pkl`
- **Usage:** Load via `joblib.load()`

## Requirements
To run the notebook, install the following dependencies:
- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`
- `matplotlib`, `seaborn`
- `joblib`

*(The environment also includes TensorFlow/Keras if you want to extend the notebook with neural networks later).*

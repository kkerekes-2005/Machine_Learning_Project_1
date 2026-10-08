# German Credit Risk Classification

Machine Learning Project 1 · Spring Semester 2026 · Kinga Kerekes

A binary classification project that predicts whether a loan applicant is a **good** or **bad** credit risk using the [German Credit dataset](https://www.openml.org/d/31) (OpenML ID 31). The project compares linear and non-parametric classifiers, chooses a metric suited to the class imbalance, and improves the best models with cost-sensitive learning.

## Overview

- **Data:** 1,000 applicants, 20 features (7 numerical, 13 categorical), 70% good / 30% bad risk
- **Task:** predict credit risk (good = 0, bad = 1)
- **Primary metric:** F1-score (accuracy is misleading on imbalanced data); AUC is reported as a complementary metric
- **Models:** Logistic Regression, Linear SVM, k-Nearest Neighbors (the Perceptron is discussed and ruled out because the classes are not linearly separable)

## Methodology

1. **Exploratory data analysis:** class distribution, missing values, feature distributions by class, categorical relationships
2. **Preprocessing:** stratified 80/20 train/test split, `StandardScaler` for numerical features, one-hot encoding for categorical features, all inside a single scikit-learn `Pipeline` with `ColumnTransformer` (no data leakage)
3. **Hyperparameter tuning:** `GridSearchCV` with 5-fold stratified cross-validation, scored on F1
4. **Evaluation:** precision, recall, F1, AUC, confusion matrices and ROC curves on the held-out test set
5. **Improvement:** cost-sensitive class weights (`class_weight={0: 1, 1: 3}`) that reflect the dataset's cost structure, where approving a bad applicant is far costlier than rejecting a good one

## Results

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.681 | 0.533 | 0.598 | 0.81 |
| Linear SVM | 0.642 | 0.567 | 0.602 | 0.80 |
| kNN (k=5) | 0.568 | 0.350 | 0.433 | 0.73 |
| **Logistic Regression, weighted (3:1)** | 0.521 | **0.833** | **0.641** | 0.81 |
| Linear SVM, weighted (3:1) | 0.516 | 0.833 | 0.637 | 0.81 |

**Key findings**

- The linear models clearly outperform kNN, whose distance-based approach suffers in the high-dimensional one-hot feature space.
- Both baseline linear models miss roughly half of the bad-risk applicants (low recall).
- Cost-sensitive weighting raises recall from about 0.53 to 0.83 and improves F1 for both models. AUC is unchanged, since weighting moves the operating point rather than the model's underlying discriminative ability.
- Final recommendation: **weighted Logistic Regression**, thanks to its best F1, interpretable coefficients and probability outputs that allow further threshold tuning.

## Repository Contents

```
.
├── Project1_kinga_kerekes.ipynb   # Full analysis and code
├── Project1_kinga_kerekes.pdf     # Written project report
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Project1_kinga_kerekes.ipynb
```

The dataset is downloaded automatically from OpenML via `fetch_openml(data_id=31)`, so an internet connection is needed the first time you run the notebook.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Author

Kinga Kerekes

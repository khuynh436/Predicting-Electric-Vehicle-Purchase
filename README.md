# Predicting Electric Vehicle Interest

Kaggle Playground Series S6E9 — binary classification project predicting the probability that a person will buy an electric vehicle (`Will_Buy_EV`).

**Competition:** [Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)
**Metric:** ROC AUC
**Data:** Synthetic tabular dataset (31 columns), inspired by an EV adoption behavior dataset

## Overview

This repo contains my approach to the competition: exploratory data analysis, feature engineering, model training, and submission generation.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── submissions/
│   └── submission.csv
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — feature types, missing values, target balance, correlations
2. **Feature engineering** — domain-informed features around EV purchase intent (income, commute, charging access, etc.)
3. **Modeling** — gradient boosting models (LightGBM / XGBoost / CatBoost), stratified k-fold cross-validation
4. **Validation** — AUC tracked across folds, checked for train/test distribution shift
5. **Submission** — probability predictions for `Will_Buy_EV`, formatted per `sample_submission.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py
python src/predict.py
```

## Results

| Model | CV AUC | Public LB |
|---|---|---|
| Baseline LightGBM | TBD | TBD |

## License

Data provided under CC BY 4.0 by the competition organizers. Code in this repo is available under the MIT License.

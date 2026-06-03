# CSAT Verbatim Classifier

An ML pipeline that automatically categorises free-text CSAT customer feedback into structured categories with confidence scores — built to solve a real operational problem at scale.

## The problem

Reading through hundreds of verbatim CSAT comments manually is slow, inconsistent and doesn't scale. This classifier automates that process so ops and CX teams can act on the data instead of just collecting it.

## What it does

Each piece of feedback gets classified into:

- **Major category** — Product, System, People, or Process
- **Sub-category** — e.g. Delivery, Communication, Resolution, Refund
- **Confidence score** — so low-certainty predictions can be filtered before reporting

Results feed into a Power BI dashboard for weekly ops and CX review.

## Why ML instead of an LLM

Token costs made a generative AI approach too expensive to run at the volume needed. The ML model gives consistent, repeatable results at scale for a fraction of the cost.

## How it works

```
Raw verbatim text
       ↓
Text cleaning (spaCy lemmatisation, stopword removal)
       ↓
Feature extraction (TF-IDF trigrams + sentiment + text stats)
       ↓
Ensemble classifier (LinearSVC + XGBoost + LightGBM + RandomForest)
       ↓
Category + Sub-category + Confidence score
```

**Key design decisions:**
- **SMOTE** handles class imbalance (some categories have fewer examples)
- **CalibratedClassifierCV** converts SVC decision scores into proper probabilities
- **Soft voting ensemble** averages probabilities across models for more reliable predictions
- **Separate sub-category models** trained per major category for better accuracy

## Project structure

```
csat-verbatim-classifier/
├── train_enhanced.py     # Train the models
├── predict.py            # Run predictions on new data
├── requirements.txt      # Python dependencies
├── data/
│   └── sample_data.xlsx  # Synthetic training data (52 examples)
├── models/               # Saved models (generated after training)
└── output/               # Prediction results
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Train the models
python train_enhanced.py

# 3. Add your verbatims to data/new_verbatims.xlsx
#    (must have a column called "Verbatim")

# 4. Run predictions
python predict.py
```

## Sample output

| Verbatim | Predicted_Category | Confidence | Predicted_Subcategory | Confidence |
|---|---|---|---|---|
| Package arrived damaged | Product | 0.91 | Delivery | 0.87 |
| Agent was rude on the phone | People | 0.88 | Communication | 0.84 |
| Still waiting for my refund | Process | 0.93 | Refund | 0.90 |

## Tech stack

- Python, Scikit-learn, XGBoost, LightGBM
- spaCy (NLP), NLTK VADER (sentiment)
- imbalanced-learn (SMOTE)
- pandas, openpyxl

## Author

Mohammed Azaruddin — Senior BI & Analytics Leader  
[LinkedIn](https://linkedin.com/in/azar141091)

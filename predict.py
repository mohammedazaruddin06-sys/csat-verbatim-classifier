"""
predict.py
Loads trained models and classifies new verbatim feedback.
Outputs a spreadsheet with predicted category, sub-category and confidence scores.
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)

INPUT_FILE  = Path("data/new_verbatims.xlsx")  # column must be called "Verbatim"
TEXT_COL    = "Verbatim"
MODELS_DIR  = Path("models")
OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "predictions.xlsx"

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
sid = SentimentIntensityAnalyzer()


# Must match the cleaning used during training
def spacy_clean(text):
    text = "" if text is None else str(text).strip()
    text = re.sub(r"[^\w\s]", " ", text).lower()
    doc  = nlp(text)
    return " ".join(
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 2
    )


# Must match the feature class used during training
class TextStatFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            text = str(text)
            s  = sid.polarity_scores(text)
            wc = max(len(text.split()), 1)
            feats.append([
                len(text), wc,
                len(set(text.split())),
                len(set(text.split())) / wc,
                text.count("!"), text.count("?"),
                s["compound"], s["pos"], s["neg"], s["neu"],
                sum(1 for c in text if c.isupper()) / max(len(text), 1),
                len([w for w in text.split() if w.isupper()]),
            ])
        return np.array(feats)


def load_model(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun train_enhanced.py first.")
    return load(path)


def predict(df):
    model_major = load_model(MODELS_DIR / "model_major.joblib")
    le_major    = load_model(MODELS_DIR / "label_encoder_major.joblib")

    X = df[TEXT_COL].apply(spacy_clean).values

    major_probs  = model_major.predict_proba(X)
    major_idx    = np.argmax(major_probs, axis=1)
    major_labels = le_major.inverse_transform(major_idx)
    major_conf   = np.max(major_probs, axis=1).round(3)

    sub_labels, sub_confs = [], []

    for i, major in enumerate(major_labels):
        safe = major.lower().replace(" ", "_")
        sub_model_path = MODELS_DIR / f"model_sub_{safe}.joblib"
        sub_le_path    = MODELS_DIR / f"label_encoder_sub_{safe}.joblib"

        if sub_model_path.exists():
            sub_model = load_model(sub_model_path)
            sub_le    = load_model(sub_le_path)
            sub_probs = sub_model.predict_proba([X[i]])
            sub_idx   = np.argmax(sub_probs)
            sub_labels.append(sub_le.inverse_transform([sub_idx])[0])
            sub_confs.append(round(float(np.max(sub_probs)), 3))
        else:
            sub_labels.append("N/A")
            sub_confs.append(None)

    df["Predicted_Category"]     = major_labels
    df["Category_Confidence"]    = major_conf
    df["Predicted_Subcategory"]  = sub_labels
    df["Subcategory_Confidence"] = sub_confs
    return df


def main():
    df = pd.read_excel(INPUT_FILE)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows")

    df = predict(df)
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}\n")
    print(df[["Verbatim", "Predicted_Category", "Category_Confidence",
              "Predicted_Subcategory", "Subcategory_Confidence"]].head(10).to_string())


if __name__ == "__main__":
    main()

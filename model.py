import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import pipeline


class ToxicityClassifier:
    def __init__(self):

        # -------------------------------
        # Baseline placeholders (optional)
        # -------------------------------
        self.baseline_model = None
        self.baseline_vectorizer = None

        # -------------------------------
        # Load Transformer Model
        # -------------------------------
        print("Loading Toxic-BERT model... (first run may take 10–20 seconds)")

        self.transformer_pipe = pipeline(
    "text-classification",
    model="unitary/multilingual-toxic-xlm-roberta",
    device=-1
)

    # -----------------------------------
    # Baseline (TF-IDF) Placeholder
    # -----------------------------------
    def predict_baseline(self, text):
        return {
            "label": "Baseline not implemented",
            "is_toxic": 0,
            "confidence": 0
        }

    # -----------------------------------
    # Transformer Prediction (Stable)
    # -----------------------------------
    def predict_transformer(self, text):

        output = self.transformer_pipe(
            text,
            truncation=True,
            max_length=512,
            top_k=None  # IMPORTANT: returns all class scores safely
        )

        """
        Output format example:
        [
            {'label': 'toxic', 'score': 0.91},
            {'label': 'non-toxic', 'score': 0.09}
        ]
        """

        toxic_score = 0.0

        for item in output:
            if item["label"].lower() == "toxic":
                toxic_score = item["score"]

        # Adjustable threshold
        threshold = 0.5

        is_toxic = 1 if toxic_score >= threshold else 0
        confidence = toxic_score if is_toxic else (1 - toxic_score)

        return {
            "label": "Toxic" if is_toxic else "Non-Toxic",
            "is_toxic": is_toxic,
            "confidence": round(confidence * 100, 2),
            "toxic_score": round(toxic_score * 100, 2)
        }

    # -----------------------------------
    # Main Predict Method
    # -----------------------------------
    def predict(self, text, use_advanced=True):
        if use_advanced:
            return self.predict_transformer(text)
        else:
            return self.predict_baseline(text)


# ---------------------------------------
# Streamlit Model Cache
# ---------------------------------------
@st.cache_resource
def load_model():
    return ToxicityClassifier()

model = load_model()
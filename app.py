import re
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MAX_LENGTH = 200
MODEL_PATH = "news_classifier.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

st.set_page_config(page_title="News Category Classifier")


# ---------------------------------------------------------
# Text cleaning (same as training notebook)
# ---------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


# ---------------------------------------------------------
# Load artifacts (cached so they load only once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


def predict_category(text, model, tokenizer, label_encoder):
    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding="pre"
    )

    probs = model.predict(padded, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))

    predicted_label = label_encoder.inverse_transform(
        [predicted_idx]
    )[0]

    results = dict(
        zip(label_encoder.classes_, probs)
    )

    return predicted_label, results


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("News Category Classifier")

st.write(
    "Paste a news headline and/or short description below, and the model "
    "will predict its category (Politics, Wellness, Entertainment, Travel, or Business)."
)


try:
    model, tokenizer, label_encoder = load_artifacts()

except Exception as e:
    st.error(
        f"Could not load model/tokenizer/label encoder files. "
        f"Make sure '{MODEL_PATH}', '{TOKENIZER_PATH}', and '{LABEL_ENCODER_PATH}' "
        f"are in the same folder as app.py.\n\nError: {e}"
    )

    st.stop()


article_text = st.text_area(
    "Enter news text",
    height=150,
    placeholder="e.g. Apple reported strong quarterly earnings as sales of its latest products helped increase the company's revenue and profits.",
)


if st.button("Predict Category", type="primary"):

    if not article_text.strip():

        st.warning("Please enter some text first.")

    else:

        predicted_label, results = predict_category(
            article_text,
            model,
            tokenizer,
            label_encoder
        )

        st.success(
            f"Predicted Category: {predicted_label}"
        )

        st.subheader("Confidence Scores")

        sorted_results = dict(
            sorted(
                results.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        for category, prob in sorted_results.items():

            st.write(
                f"{category}: {prob:.2%}"
            )

            st.progress(float(prob))
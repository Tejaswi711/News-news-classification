# News Category Classification using LSTM and GloVe

## Project Overview

This project classifies news articles into five categories using Natural Language Processing and Deep Learning.

The five categories are:

- Business
- Entertainment
- Politics
- Travel
- Wellness

## Dataset

The project uses the HuffPost News Category Dataset.

The original dataset contains 42 categories. For this project, five categories were selected and 5,000 articles were sampled from each category, resulting in 25,000 balanced articles.

## NLP Preprocessing

- Combined headline and short description
- Converted text to lowercase
- Removed HTML tags
- Removed URLs
- Removed extra spaces
- Tokenization
- Padding to 200 tokens

## Model

The model architecture is:

GloVe Embedding → LSTM(128) → Dropout(0.3) → Dense(5, Softmax)

GloVe 100-dimensional pretrained embeddings were used.

## Model Performance

Test Accuracy: 84.72%

## Deployment

The trained model was deployed using Streamlit.

Users can enter a news headline or description, and the application predicts the news category with confidence scores.

## Technologies Used

- Python
- TensorFlow
- Keras
- NLP
- LSTM
- GloVe
- Scikit-learn
- Streamlit
- Pandas
- NumPy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from transformers import pipeline

# Load models and preprocessors
vectorizer = joblib.load("countVectorizer.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_xgb.pkl")

# Initialize Hugging Face RoBERTa sentiment analysis pipeline
roberta = pipeline("sentiment-analysis")

# Define class labels (binary example)
classes = ["Negative", "Positive"]

app = FastAPI()

# CORS setup (if frontend and backend are on different ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    # Step 1: Vectorize for XGBoost
    vectorized_text = vectorizer.transform([input.text])

    # Step 2: Scale for XGBoost
    scaled_text = scaler.transform(vectorized_text.toarray())

    # Step 3: Get XGBoost model probabilities
    proba_xgb = model.predict_proba(scaled_text)[0]
    max_index_xgb = np.argmax(proba_xgb)
    predicted_label_xgb = classes[max_index_xgb]
    
    # Step 4: Get RoBERTa prediction
    roberta_result = roberta(input.text)[0]
    predicted_label_roberta = roberta_result['label']
    roberta_score = roberta_result['score']
    
    # Step 5: Handle low confidence in RoBERTa
    if roberta_score < 0.1:  # Threshold adjusted for uncertain cases
        predicted_label_roberta = "Uncertain"
    
    # Step 6: Correctly assign probabilities for RoBERTa based on label
    if predicted_label_roberta == "POSITIVE":
        roberta_pos_score = roberta_score
        roberta_neg_score = 1.0 - roberta_score
    else:
        roberta_neg_score = roberta_score
        roberta_pos_score = 1.0 - roberta_score

    # Step 7: Format response
    probability_dict = {
        "XGBoost_Positive": float(proba_xgb[1]),
        "XGBoost_Negative": float(proba_xgb[0]),
        "RoBERTa_Positive": float(roberta_pos_score) if predicted_label_roberta != "Uncertain" else 0.0,
        "RoBERTa_Negative": float(roberta_neg_score) if predicted_label_roberta != "Uncertain" else 0.0
    }

    return {
        "input_text": input.text,
        "xgboost_prediction": predicted_label_xgb,
        "roberta_prediction": predicted_label_roberta,
        "probabilities": probability_dict
    }

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html") as f:
        return f.read()

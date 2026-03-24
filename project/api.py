import os
import io
import json
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.data_processor import load_and_preprocess_data, make_windows, make_train_test_splits
from src.models import NBeatsBlock
from src.trainer import make_preds

app = FastAPI(title="Bitcoin Time Series Forecaster")

# Constants
WINDOW_SIZE = 7
HORIZON = 1
DATA_PATH = "data/BTC_USD.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

class PredictionResponse(BaseModel):
    historical_dates: list[str]
    historical_prices: list[float]
    future_dates: list[str]
    future_prices: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    model_name: str

@app.get("/api/predict")
def get_prediction(model: str = "LSTM", days: int = 7):
    """
    Generates an iterative future prediction for the next N days.
    """
    try:
        timesteps, prices = load_and_preprocess_data(DATA_PATH)
        
        # Load the selected model
        model_filename_map = {
            "Dense": "model_dense.keras",
            "Conv1D": "model_conv1D.keras",
            "LSTM": "model_lstm.keras",
            "NBEATS": "model_nbeats.keras"
        }
        
        if model not in model_filename_map:
            raise HTTPException(status_code=400, detail="Invalid model name")
            
        model_path = os.path.join(MODEL_DIR, model_filename_map[model])
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Please run the training pipeline first.")
            
        best_model = tf.keras.models.load_model(model_path, custom_objects={"NBeatsBlock": NBeatsBlock})
        
        # Iterative Prediction
        last_window = np.copy(prices[-WINDOW_SIZE:])
        future_preds = []
        
        # Estimate residuals for confidence intervals
        std_resid = prices[-len(prices)//5:].std() * 0.05
        
        for _ in range(days):
            pred_input = tf.expand_dims(last_window, axis=0)
            pred = make_preds(best_model, pred_input)
            
            p_val = float(pred.numpy()) if pred.ndim == 0 else float(pred.numpy()[0])
            future_preds.append(p_val)
            last_window = np.append(last_window[1:], p_val)
            
        last_date = timesteps[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        lower_bound = np.array(future_preds) - 1.96 * std_resid
        upper_bound = np.array(future_preds) + 1.96 * std_resid
        
        # Provide the last 60 days of history for context
        hist_len = min(60, len(timesteps))
        
        return {
            "historical_dates": pd.to_datetime(timesteps[-hist_len:]).strftime("%Y-%m-%d").tolist(),
            "historical_prices": prices[-hist_len:].tolist(),
            "future_dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "future_prices": future_preds,
            "lower_bounds": lower_bound.tolist(),
            "upper_bounds": upper_bound.tolist(),
            "model_name": model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
def get_metrics():
    """
    Returns the evaluation summary of all trained models.
    """
    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path, index_col=0)
        return df.to_dict(orient="index")
    return {"message": "Metrics not available yet."}

# Mount static files at the root
app.mount("/", StaticFiles(directory="public", html=True), name="public")

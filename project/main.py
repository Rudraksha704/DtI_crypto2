import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_processor import load_and_preprocess_data, make_windows, make_train_test_splits
from src.models import build_dense_model, build_conv1d_model, build_lstm_model, build_nbeats_model, NBeatsBlock
from src.trainer import train_model, make_preds, generate_naive_forecast, evaluate_preds
from src.utils import plot_actual_vs_predicted, save_predictions

# Constants
WINDOW_SIZE = 7
HORIZON = 1
BATCH_SIZE = 128
EPOCHS = 100
DATA_PATH = "data/BTC_USD.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

def run_training():
    print("Starting training pipeline...")
    timesteps, prices = load_and_preprocess_data(DATA_PATH)
    windows, labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows, labels, test_split=0.2)
    
    print(f"Data shapes - Train: {train_windows.shape}, Test: {test_windows.shape}")
    
    models = {
        "Dense": build_dense_model(horizon=HORIZON, neurons=128),
        "Conv1D": build_conv1d_model(horizon=HORIZON, filters=128, kernel_size=5),
        "LSTM": build_lstm_model(horizon=HORIZON, window_size=WINDOW_SIZE, neurons=128),
        # N-BEATS can be slow/heavy, we'll configure a slightly reduced scale
        "NBEATS": build_nbeats_model(input_size=WINDOW_SIZE, horizon=HORIZON, n_neurons=128, n_layers=4, n_stacks=10)
    }
    
    results = {}
    test_dates = timesteps[-len(test_windows):]
    
    # 1. Naive Baseline
    print("\n--- Training Naive Baseline ---")
    naive_forecast = generate_naive_forecast(test_labels)
    # naive forecast shifts by 1, so the first pred corresponds to test_labels[1:]
    naive_results = evaluate_preds(tf.squeeze(test_labels[1:]), tf.squeeze(naive_forecast))
    results["Naive"] = naive_results
    plot_actual_vs_predicted(test_dates[1:], tf.squeeze(test_labels[1:]), tf.squeeze(naive_forecast), "Naive", OUTPUT_DIR)
    
    # 2. Train Deep Learning Models
    for name, model in models.items():
        print(f"\n--- Training {name} Model ---")
        train_model(model, train_windows, train_labels, test_windows, test_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Load best and evaluate
        best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{model.name}.keras"), custom_objects={"NBeatsBlock": NBeatsBlock})
        preds = make_preds(best_model, test_windows)
        
        eval_metrics = evaluate_preds(tf.squeeze(test_labels), preds)
        results[name] = eval_metrics
        
        plot_actual_vs_predicted(test_dates, tf.squeeze(test_labels), preds, name, OUTPUT_DIR)
        
        # Calculate residual standard deviation for prediction intervals on the test set
        test_labels_float32 = tf.cast(tf.squeeze(test_labels), dtype=tf.float32)
        preds_float32 = tf.cast(preds, dtype=tf.float32)
        residuals = test_labels_float32 - preds_float32
        std_resid = np.std(residuals)
        lower_bound = preds - 1.96 * std_resid
        upper_bound = preds + 1.96 * std_resid
        
        save_predictions(test_dates, test_labels, preds, name, bounds=(lower_bound, upper_bound), save_dir=OUTPUT_DIR)

    # 3. Print Tabular Results
    print("\n================ EVALUATION SUMMARY ================")
    results_df = pd.DataFrame(results).T
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"))
    print("====================================================")


def run_prediction(future_steps=7):
    print(f"Starting prediction pipeline for next {future_steps} days...")
    timesteps, prices = load_and_preprocess_data(DATA_PATH)
    
    # We will use the Dense model as the default best model for demonstration purposes
    # Ensure it's trained first
    model_path = os.path.join(MODEL_DIR, "model_dense.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run with --train first.")
        return
        
    best_model = tf.keras.models.load_model(model_path, custom_objects={"NBeatsBlock": NBeatsBlock})
    
    # Predict into the future iteratively
    last_window = np.copy(prices[-WINDOW_SIZE:])
    future_preds = []
    
    # We also need residuals standard deviation to estimate intervals
    # We'll recompute quickly or use a conservative average
    std_resid = prices[-len(prices)//5:].std() * 0.05 # rough heuristic if we don't save residuals
    
    for _ in range(future_steps):
        pred_input = tf.expand_dims(last_window, axis=0)
        pred = make_preds(best_model, pred_input)
        if pred.ndim == 0:
            pred_val = float(pred.numpy())
        else:
            pred_val = float(pred.numpy()[0])
            
        future_preds.append(pred_val)
        
        # update window
        last_window = np.append(last_window[1:], pred_val)
    
    future_preds = np.array(future_preds)
    last_date = timesteps[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    
    lower_bound = future_preds - 1.96 * std_resid
    upper_bound = future_preds + 1.96 * std_resid
    
    df_future = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_preds,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound
    })
    
    out_file = os.path.join(OUTPUT_DIR, "future_forecast.csv")
    df_future.to_csv(out_file, index=False)
    print(f"Saved forward forecast to {out_file}")
    
    # Plot Future Forecast
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.plot(timesteps[-30:], prices[-30:], label="Recent History")
    plt.plot(future_dates, future_preds, label="Forecast", color="orange")
    plt.fill_between(future_dates, lower_bound, upper_bound, color="orange", alpha=0.2, label="95% Interval")
    plt.legend()
    plt.title(f"Future {future_steps} Days Forecast")
    
    plot_file = os.path.join(OUTPUT_DIR, "future_forecast.png")
    plt.savefig(plot_file)
    print(f"Saved forecast plot to {plot_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline for all models")
    parser.add_argument("--predict", action="store_true", help="Run prediction into the future using the best model")
    
    args = parser.parse_args()
    
    if args.train:
        run_training()
    elif args.predict:
        run_prediction(future_steps=7)
    else:
        print("Please specify an action: --train or --predict")

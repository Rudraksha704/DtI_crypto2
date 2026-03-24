import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
    """
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def plot_actual_vs_predicted(timesteps, actual, predicted, model_name, save_dir="outputs"):
    """
    Plots the final test dataset actual vs predicted and saves the PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plot_time_series(timesteps, actual, label="Actual Test Data")
    plot_time_series(timesteps, predicted, format="-", label=f"{model_name} Predictions")
    
    filepath = os.path.join(save_dir, f"{model_name}_actual_vs_predicted.png")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to {filepath}")

def save_predictions(test_dates, actual_prices, predicted_prices, model_name, bounds=None, save_dir="outputs"):
    """
    Saves predictions to a CSV file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if len(predicted_prices.shape) > 1 and predicted_prices.shape[1] == 1:
        predicted_prices = predicted_prices[:, 0]
    if len(actual_prices.shape) > 1 and actual_prices.shape[1] == 1:
        actual_prices = actual_prices[:, 0]

    data = {
        "Date": test_dates,
        "Actual_Price": actual_prices,
        "Predicted_Price": predicted_prices
    }
    
    if bounds:
        data["Lower_Bound"] = bounds[0]
        data["Upper_Bound"] = bounds[1]
        
    df = pd.DataFrame(data)
    
    filepath = os.path.join(save_dir, f"{model_name}_predictions.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved predictions to {filepath}")

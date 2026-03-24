import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
    """
    Loads Bitcoin historical data, parses dates, and extracts prices and timesteps.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col=["Date"])
    
    # Handle missing values if any
    if df.isnull().values.any():
        df = df.fillna(method="ffill")
        
    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    
    # Sort chronologically just in case
    bitcoin_prices = bitcoin_prices.sort_index()

    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()
    
    return timesteps, prices

def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

    # 2. Create a 2D array of multiple window steps
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T

    # 3. Index on the target array
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits preserving order.
    """
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    
    return train_windows, test_windows, train_labels, test_labels

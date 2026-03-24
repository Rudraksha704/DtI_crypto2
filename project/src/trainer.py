import os
import tensorflow as tf

def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season

def evaluate_preds(y_true, y_pred):
    """
    Calculates various evaluation metrics for Time Series Forecasting.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    mape = tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100.0
    mase = mean_absolute_scaled_error(y_true, y_pred)

    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {
        "mae": float(mae.numpy()),
        "mse": float(mse.numpy()),
        "rmse": float(rmse.numpy()),
        "mape": float(mape.numpy()),
        "mase": float(mase.numpy())
    }

def create_model_checkpoint(model_name, save_path="models"):
    """
    Creates a ModelCheckpoint callback to save the best model.
    """
    os.makedirs(save_path, exist_ok=True)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, model_name + ".keras"),
        verbose=0,
        save_best_only=True
    )

def train_model(model, train_windows, train_labels, test_windows, test_labels, epochs=100, batch_size=128):
    """
    Compiles and fits the model. Saves the best version to disk.
    """
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
    
    # Needs a build step or simply fit (which builds automatically)
    history = model.fit(
        train_windows,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_windows, test_labels),
        callbacks=[create_model_checkpoint(model_name=model.name)]
    )
    return history

def make_preds(model, input_data):
    """
    Uses model to make predictions on input_data.
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)

def generate_naive_forecast(y_test):
    """
    Generates a naive forecast (predicts yesterday's value for today).
    """
    naive_forecast = y_test[:-1]
    return naive_forecast

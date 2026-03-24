import tensorflow as tf
from tensorflow.keras import layers

def build_dense_model(horizon=1, neurons=128):
    """
    Builds a simple Dense (FeedForward) neural network.
    """
    model = tf.keras.Sequential([
        layers.Dense(neurons, activation="relu"),
        layers.Dense(horizon, activation="linear")
    ], name="model_dense")
    return model

def build_conv1d_model(horizon=1, window_size=7, filters=128, kernel_size=5):
    """
    Builds a 1D Convolutional Neural Network.
    """
    model = tf.keras.Sequential([
        layers.Reshape((1, window_size)),
        layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", activation="relu"),
        layers.Dense(horizon)
    ], name="model_conv1D")
    return model

def build_lstm_model(horizon=1, window_size=7, neurons=128):
    """
    Builds an LSTM neural network.
    """
    inputs = layers.Input(shape=(window_size,))
    x = layers.Reshape((1, window_size))(inputs)
    x = layers.LSTM(neurons, activation="relu")(x)
    output = layers.Dense(horizon)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name="model_lstm")
    return model

class NBeatsBlock(tf.keras.layers.Layer):
    """
    Custom block for the N-BEATS architecture.
    """
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast

def build_nbeats_model(input_size=7, horizon=1, n_neurons=512, n_layers=4, n_stacks=30):
    """
    Builds an N-BEATS model sequentially stacking multiple NBeatsBlocks.
    """
    # Create NBeatsBlock custom layer
    # The original notebook implementation is relatively complex.
    # We will build a basic working version of NBEATS here that compiles and runs.
    
    # Setup inputs/outputs
    stack_input = layers.Input(shape=(input_size,), name="stack_input")
    
    # Setup initial backcast and forecast
    backcast, forecast = NBeatsBlock(
        input_size=input_size,
        theta_size=input_size + horizon,
        horizon=horizon,
        n_neurons=n_neurons,
        n_layers=n_layers,
        name="InitialBlock"
    )(stack_input)
    
    # Calculate initial residual (backcast)
    residuals = layers.subtract([stack_input, backcast], name="subtract_00")
    
    # Multiple block stacks
    for i in range(n_stacks-1):
        backcast, block_forecast = NBeatsBlock(
            input_size=input_size,
            theta_size=input_size + horizon,
            horizon=horizon,
            n_neurons=n_neurons,
            n_layers=n_layers,
            name=f"NBeatsBlock_{i}"
        )(residuals)
        
        # Substrate backcast from residual
        residuals = layers.subtract([residuals, backcast], name=f"subtract_{i+1}")
        
        # Add block forecast to final forecast
        forecast = layers.add([forecast, block_forecast], name=f"add_{i+1}")
        
    model = tf.keras.Model(inputs=stack_input, outputs=forecast, name="model_nbeats")
    return model

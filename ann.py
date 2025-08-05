'''Este arquivo contém funções para construir modelos de aprendizado de máquina usando Keras.'''

import keras

def build_model_mlp(hidden_layers_units: tuple, window_size: int = 15, n_steps: int = 15, activation: str = 'relu') -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size,)),
        keras.layers.Dense(hidden_layers_units[0], activation=activation),
        keras.layers.Dense(hidden_layers_units[1], activation=activation),
        keras.layers.Dense(n_steps)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def build_model_lstm(units: int = 15, window_size: int = 15, n_steps: int = 15, activation: str = 'tanh') -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        keras.layers.LSTM(units, activation=activation),
        keras.layers.Dense(n_steps)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model
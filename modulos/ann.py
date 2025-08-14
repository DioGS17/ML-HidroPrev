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

def build_model_lstm_mlp(lstm_units: int, lstm_input_n: int, 
                         hidden_layers_units: tuple, mlp_input_n: int,
                         lstm_activation: str = 'tanh', mlp_activation: str = 'relu'):
    # LSTM way
    lstm_input = keras.layers.Input(shape=(lstm_input_n, 1))
    lstm_layer = keras.layers.LSTM(lstm_units, activation=lstm_activation)(lstm_input)

    # MLP way
    mlp_input = keras.layers.Input(shape=(mlp_input_n,))
    mlp_layer1 = keras.layers.Dense(hidden_layers_units[0], activation=mlp_activation)(mlp_input)
    mlp_layer2 = keras.layers.Dense(hidden_layers_units[1], activation=mlp_activation)(mlp_layer1)

    # Concat
    concat = keras.layers.Concatenate()([lstm_layer, mlp_layer2])

    # Saída
    output = keras.layers.Dense(1)(concat)

    model = keras.Model(inputs=[lstm_input, mlp_input], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model
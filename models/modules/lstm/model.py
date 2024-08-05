import tensorflow as tf
import keras

import os
import pickle

tf.config.run_functions_eagerly(True)

class LSTMAutoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(None, self.input_shape)))
        model.add(keras.layers.Masking(mask_value=0.0))
        model.add(
            keras.layers.LSTM(
                160,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
            )
        )
        model.add(keras.layers.Dropout(0.1))
        model.add(
            keras.layers.LSTM(
                160,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
            )
        )
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.input_shape, activation="linear")))
        my_loss = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss=my_loss)
        return model

    def summary(self):
        self.model.summary()

    def save(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self._save_parameters(filepath)
        self._save_weights(filepath)

    def _save_parameters(self, path):
        parameters = {"input_shape": self.input_shape}
        with open(os.path.join(path, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, path):
        self.model.save(os.path.join(path, "weights.h5"))

    def load(self, filepath):
        parameters_path = os.path.join(filepath, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        self.input_shape = parameters["input_shape"]
        self.model = self._build_model()
        self.model.load_weights(os.path.join(filepath, "weights.h5"))

    def train(self, x_train, x_val, epochs=500, batch_size=32):
        history = self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val),
            shuffle=False,
        )
        return history

    def predict(self, x):
        return self.model.predict(x)

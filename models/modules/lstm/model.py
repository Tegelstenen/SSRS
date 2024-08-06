
# TODO: remove defaults

import tensorflow as tf
import keras

import keras_tuner as kt

import os
import pickle
import logging

from utils.log_setup import setup_logging

setup_logging()

class LSTMAutoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.best_hps = None
        self.tuner = kt.BayesianOptimization(
                    self._build_model,
                    objective='val_mean_absolute_error',
                    max_trials=10
                    )
        

    def _build_model(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(None, self.input_shape)))
        model.add(keras.layers.Masking(mask_value=0.0))

        for i in range(hp.Int('num_layers', 2, 20)):
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                activation='tanh'))
            model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0.1, 0.5, step=0.1)))
        
        # last LSTM layer
        hp_units_last = hp.Int("units_last", min_value=32, max_value=512, step=32)
        model.add(
            keras.layers.LSTM(
                units=hp_units_last,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
            )
        )
        
        # TimeDistributed layer
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(self.input_shape, activation="linear")
            )
        )

        # Learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_absolute_error',
            metrics=['mean_absolute_error'])
        
        return model

    def summary(self):
        self.model.summary()

    def save(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self._save_parameters(filepath)
        self._save_weights(filepath)
        self._save_hps(filepath)

    def _save_hps(self, path):
        with open(os.path.join(path, "hps.pkl"), "wb") as f:
            pickle.dump(self.best_hps, f)

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
        
        hps_path = os.path.join(filepath, "hps.pkl")
        with open(hps_path, "rb") as f:
            self.best_hps = pickle.load(f)
        
        self.model = self.tuner.hypermodel.build(self.best_hps)
        
        self.model.load_weights(os.path.join(filepath, "weights.h5"))

    def train(self, x_train, x_val):
        
        logging.model("Defining callback")
        stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

        logging.model("Searching for best hyperparameters")
        self.tuner.search(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=10, #TODO: change this to 50
            shuffle=False,
            callbacks=[stop_early],
        )
        logging.model("Getting best hyperparameters")
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        
        logging.model("Building model with best hyperparameters")
        model = self.tuner.hypermodel.build(self.best_hps)
        
        logging.model("Training hypermodel")
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=50, #TODO: change this to 500
            shuffle=False,
        )
        logging.model("Getting best epoch")
        val_loss_per_epoch = history.history["val_loss"]
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        
        logging.model("Training model with best epoch")
        self.model = self.tuner.hypermodel.build(self.best_hps)
        history = self.model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=best_epoch,
            shuffle=False,
        )
        return history


    def predict(self, x):
        return self.model.predict(x)

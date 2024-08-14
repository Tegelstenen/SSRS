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
        self.tuner = kt.Hyperband(
            self._build_model, 
            objective="val_loss", 
            directory='my_dir',
            project_name='lstm_autoencoder'
        )

    def _build_model(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1])))
        model.add(keras.layers.Masking(mask_value=-1))

        units = hp.Int("units", min_value=32, max_value=512, step=32)
        model.add(keras.layers.LSTM(units,return_sequences=True, dropout=0.5))
        model.add(keras.layers.LSTM(units//2, return_sequences=False, dropout=0.5))

        model.add(keras.layers.RepeatVector(self.input_shape[0]))

        model.add(keras.layers.LSTM(units//2, return_sequences=True, dropout=0.5))
        model.add(keras.layers.LSTM(units, return_sequences=True))

        model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.input_shape[1])))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
            ),
            loss="mse",
        )

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

    def train(self, train_dataset, val_dataset):
        logging.model("Defining callback")
        stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

        logging.model("Searching for best hyperparameters")
        self.tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            shuffle=False,
            callbacks=[stop_early],
        )
        logging.model("Getting best hyperparameters")
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        logging.model("Building model with best hyperparameters")
        model = self.tuner.hypermodel.build(self.best_hps)

        logging.model("Training hypermodel")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=500,
            shuffle=False,
            callbacks=[stop_early],
        )
        logging.model("Getting best epoch")
        val_loss_per_epoch = history.history["val_loss"]
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

        logging.model("Training model with best epoch")
        self.model = self.tuner.hypermodel.build(self.best_hps)
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=best_epoch,
            shuffle=False,
        )
        return history

    def predict(self, x):
        x = x.astype('float32')
        return self.model.predict(x)

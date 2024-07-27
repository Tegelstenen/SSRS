# TODO:
# - remove all tunings dicst and make it tunable instead

from numpy import array, ndarray, full, ceil, mean, vstack, zeros
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Masking
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import ParameterGrid, KFold

import logging
import concurrent.futures
from functools import partial

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging

setup_logging()


class LSTMAutoencoder(BaseEstimator, RegressorMixin):
    def __init__(self, full_length_of_dataset: int, features: int) -> None:
        self.config = ConfigManager()
        self.features = features
        self.full_length_of_dataset = full_length_of_dataset

        self.timesteps = None  # standard value until tuned
        self.epochs = None  # standard value until tuned
        self.learning_rate = None

        # self.timesteps = 100 # standard value until tuned
        # self.epochs = 10 # standard value until tuned
        # self.learning_rate = 0.0001 # standard value until tuned

        self.model = self._create_model(self.timesteps, self.learning_rate)

    # ********* METHODS *********
    def fit(self, data: ndarray) -> Sequential:
        self.model = self._create_model(self.timesteps, self.learning_rate)
        batched_data = self._batch_data(data)
        train_data, test_data = train_test_split(
            batched_data, test_size=0.33, random_state=42
        )
        self.model.fit(
            train_data,
            train_data,
            epochs=self.epochs,
            verbose=0,
            validation_data=(test_data, test_data),
        )
        logging.model("Model fitted")
        return self.model

    def predict(self, data: ndarray) -> array:
        self.model = self._create_model(self.timesteps, self.learning_rate)
        data = self._batch_data(data)
        reconstruction = self.model.predict(data, verbose=0)
        original_format = self._re_shape(reconstruction)
        logging.model("Predictions done")
        return original_format

    def save(self, path: str) -> None:
        try:
            new_tunings = {
                "timesteps": self.timesteps,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            }
            self.config.update_param("TUNED_PARAMETERS", new_tunings)
            self.model.save_weights(path)
            logging.model("Model saved")
        except Exception as e:
            raise Exception(f"Error saving model weights: {e}")

    def load(self, path: str) -> "LSTMAutoencoder":
        try:
            tunings = self.config.get("TUNED_PARAMETERS")
            self.model.load_weights(path)
            self.timesteps = tunings["timesteps"]
            self.learning_rate = tunings["learning_rate"]
            self.epochs = tunings["epochs"]
            self.batch_size = tunings["batch_size"]
            logging.model("Model loaded")
            return self
        except Exception as e:
            raise Exception(f"Error loading model weights: {e}")

    def tune_model(self, data: ndarray) -> Sequential:
        best_params, _ = self._grid_search(data)
        for param, value in best_params.items():
            setattr(self, param, value)

    # ********* HELPERS *********
    def _batch_data(self, data: ndarray) -> ndarray:
        if self.timesteps is None:
            raise ValueError("timesteps must be set before batching data")
        
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_complete_batches = num_samples // self.timesteps
        batches = data[:num_complete_batches * self.timesteps].reshape(-1, self.timesteps, num_features)
        remaining_samples = num_samples % self.timesteps
        if remaining_samples > 0:
            last_batch = zeros((self.timesteps, num_features))
            last_batch[:remaining_samples] = data[-remaining_samples:]
            batches = vstack((batches, last_batch.reshape(1, self.timesteps, num_features)))
        
        return batches

    def _re_shape(self, data: ndarray) -> ndarray:
        flattened_data = data.reshape(-1, self.features)
        original_data = flattened_data[: self.full_length_of_dataset, :]
        logging.model("Re-shaping data done")
        return original_data

    def _create_model(self, timesteps, learning_rate):
        model = Sequential()
        model.add(Input(shape=(timesteps, self.features)))
        model.add(Masking(mask_value=-100))
        model.add(
            LSTM(
                64,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                return_sequences=True,
            )
        )
        model.add(Dropout(0.2))
        model.add(
            LSTM(
                32,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                return_sequences=True,
            )
        )
        model.add(Dropout(0.2))
        model.add(
            LSTM(
                16,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                return_sequences=True,
            )
        )
        model.add(TimeDistributed(Dense(self.features, activation="sigmoid")))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def _grid_search(self, data: ndarray):
        grid = self._get_param_grid()
        best_params = self._get_params_dict()
        best_score = float("inf")

        # Create a partial function for _evaluate_params
        evaluate_params = partial(self._evaluate_params, data=data)

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all parameter combinations for evaluation
            future_to_params = {
                executor.submit(evaluate_params, params): params for params in grid
            }

            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    mean_score = future.result()
                    logging.model(f"Finished tuning with:")
                    logging.model(f"\t {params}")
                    logging.model(f"\t {mean_score}")
                    best_params, best_score = self._get_new_best_params(
                        best_score, mean_score, best_params, params
                    )
                except Exception as exc:
                    logging.error(f"Params {params} generated an exception: {exc}")

        logging.model(f"Final params:")
        logging.model(f"\t {best_params}")
        logging.model(f"Final score:")
        logging.model(f"\t {best_score}")
        return best_params, best_score

    def _evaluate_params(self, params: dict, data: ndarray) -> float:
        self.timesteps = params["timesteps"]
        batch_data = self._batch_data(data)
        param_id = f"{params['timesteps']}_{params['learning_rate']}_{params['epochs']}_{params['batch_size']}"
        split = self._get_kfold_splits(batch_data)
        scores = []
        for iteration, (train_index, test_index) in enumerate(split, 1):
            logging.model(f"param_id: {param_id}, Kfold iteration: {iteration}")
            score = self._get_score(batch_data, params, train_index, test_index)
            scores.append(score)
        mean_score = mean(scores)
        return mean_score

    # ********* GETTERS *********
    def _get_batch_start_stop(self, i: int):
        start = i * self.timesteps
        stop = min((i + 1) * self.timesteps, self.full_length_of_dataset)
        return start, stop

    def _get_num_batches(self) -> int:
        num_batches = int(ceil(self.full_length_of_dataset / self.timesteps))
        return num_batches

    def _get_timesteps(self) -> int:
        return self.timesteps

    def _get_param_grid(self) -> ParameterGrid:
        grid = self.config.get("PARAM_GRID")
        return ParameterGrid(grid)

    def _get_params_dict(self) -> dict:
        grid = self.config.get("PARAM_GRID")
        params_dict = {key: None for key in grid.keys()}
        return params_dict

    def _get_kfold_splits(self, data: ndarray):
        kf = KFold(n_splits=5)
        splits = kf.split(data)
        return splits

    def _get_score(
        self, data: ndarray, params: dict, train_index: int, test_index: int
    ) -> float:
        train_data, test_data = data[train_index], data[test_index]
        model = self._create_model(params["timesteps"], params["learning_rate"])
        model.fit(
            train_data,
            train_data,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
        )
        score = model.evaluate(test_data, test_data, verbose=0)
        return score

    def _get_new_best_params(
        self, best_score: float, score: float, best_params: dict, params: dict
    ) -> tuple[dict, float]:
        if score < best_score:
            best_score = score
            best_params = params
        return best_params, best_score

    def get_config(self) -> list:
        return [self.batch_size, self.epochs, self.learning_rate, self.timesteps]
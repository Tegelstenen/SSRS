import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

import os

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from tests.test_helper import TestHelperFunctions

config = ConfigManager()


class TestLSTMAutoencoder:

    def setup_method(self):
        self.dataframe, self.encoded, self.sequences, self.scaled_data = TestHelperFunctions.create_test_data()
        self.features = self.scaled_data.shape[1]
        self.full_length_of_dataset = self.scaled_data.shape[0]
        self.autoencoder = LSTMAutoencoder(self.full_length_of_dataset, self.features)
        self.batched_data = self.autoencoder._batch_data(self.scaled_data)

    def test_batch_data_content(self):
        min_value = np.min(self.batched_data)
        max_value = np.max(self.batched_data)
        assert min_value == -100 and max_value == 1

        for i, batch in enumerate(self.batched_data):
            if i < len(self.batched_data) - 1:
                assert np.all(batch <= 1) and np.all(batch >= 0)
            elif i == len(self.batched_data) - 1:
                assert np.any(batch == -100)
            else:
                assert np.all(batch != -100)
                
    def test_get_num_batches(self):
        autoencoder = LSTMAutoencoder(367, 20)
        num_batches = autoencoder._get_num_batches()
        assert num_batches == 37
        
    def test_get_batch_start_stop(self):
        autoencoder = LSTMAutoencoder(367, 20)
        start, stop = autoencoder._get_batch_start_stop(22)
        assert start == 220 and stop == 230
        autoencoder = LSTMAutoencoder(367, 20)
        start, stop = autoencoder._get_batch_start_stop(36)
        assert start == 360 and stop == 367

    def test_batch_data_shape(self):
        num_of_batches = np.ceil(self.full_length_of_dataset / self.autoencoder._get_timesteps())

        assert self.batched_data.shape == (num_of_batches, self.autoencoder._get_timesteps(), self.features)

    def test_re_shape(self):
        reshaped_data = self.autoencoder._re_shape(self.batched_data)
        assert reshaped_data.shape == (self.full_length_of_dataset, self.features)

    def test_model_definition(self):
        model = self.autoencoder._define_model()
        assert model is not None

    def test_fit(self):
        model = self.autoencoder.fit(self.scaled_data)
        assert model is not None

    def test_predict(self):
        predictions = self.autoencoder.predict(self.scaled_data)
        assert predictions.shape == (self.full_length_of_dataset, self.features)
        assert np.all(predictions <= 1) and np.all(predictions >= 0)

    def test_save_and_load(self):
        path = "test_model.weights.h5"
        self.autoencoder.save(path)
        loaded_autoencoder = LSTMAutoencoder(self.full_length_of_dataset, self.features)
        loaded_autoencoder.load(path)
        assert loaded_autoencoder.model is not None
        os.remove(path)

    def test_get_param_grid(self):
        param_grid = config.get("PARAM_GRID")
        combinations = 1
        for _, values in param_grid.items():
            combinations *= len(values)
        grid = self.autoencoder._get_param_grid()
        assert len(grid) == combinations
        assert  isinstance(grid[0], dict)
        assert grid[0].keys() == param_grid.keys()
        
    def test_get_params_dict(self):
        params_dict = self.autoencoder._get_params_dict()
        assert isinstance(params_dict, dict)
        assert len(params_dict) == len(config.get("PARAM_GRID"))
        assert all(key in params_dict for key in config.get("PARAM_GRID").keys())
        
    def test_get_score(self):
        params = {"timesteps": 10, "learning_rate": 0.0001, "epochs": 10, "batch_size": 32}
        splits = self.autoencoder._get_kfold_splits(self.batched_data)
        train_index, test_index = next(splits)
        score = self.autoencoder._get_score(self.batched_data, params, train_index, test_index)
        assert isinstance(score, float)
        assert score >= 0

    def test_get_new_best_params(self):
        params_1 = {"timesteps": 10, "learning_rate": 0.0001, "epochs": 10, "batch_size": 32}
        params_2 = {"timesteps": 10, "learning_rate": 0.0001, "epochs": 10, "batch_size": 32}
        score_1 = 0
        score_2 = 1
        best_params, best_score = self.autoencoder._get_new_best_params(score_1, score_2, params_1, params_2)
        assert best_params == params_1
        assert best_score == 0
        params_1 = {"timesteps": 10, "learning_rate": 0.0001, "epochs": 10, "batch_size": 32}
        params_2 = {"timesteps": 10, "learning_rate": 0.0001, "epochs": 10, "batch_size": 32}
        score_1 = 1
        score_2 = 0
        best_params, best_score = self.autoencoder._get_new_best_params(score_1, score_2, params_1, params_2)
        assert best_params == params_2
        assert best_score == 0

    def test_grid_search(self):
        best_params, best_score = self.autoencoder._grid_search(self.batched_data)
        assert best_params is not None
        assert best_score is not None
        assert best_params.keys() == config.get("PARAM_GRID").keys()
        

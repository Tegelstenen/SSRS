import numpy as np

import os

from modules.lstm.train import Trainer, Tuner
from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from tests.test_helper import TestHelperFunctions

config = ConfigManager()


class TestTrain:

    def setup_method(self):
        np.random.seed(42)
        self.trainer = Trainer()
        self.tuner = Tuner()
        _, _, _, self.data = TestHelperFunctions.create_test_data()
        
    def test_train(self):
        model = self.trainer._train(self.data)
        assert model is not None
        assert isinstance(model, LSTMAutoencoder)
        
    def test_save(self):        
        model = self.trainer._train(self.data)
        self.trainer._save_model(model)
        assert os.path.exists("../models/model_info/.weights.h5")
        os.remove("../models/model_info/.weights.h5")
    
class TestTune:
    def setup_method(self):
        np.random.seed(42)
        self.trainer = Trainer()
        
        _, _, _, self.data = TestHelperFunctions.create_test_data()
    
    def test_get_start_stop(self):
        remaining_rows = 1000
        total_rows = 10000
        min_slice = 100
        for _ in range(1000):
            start, stop = Tuner._get_start_stop(remaining_rows, total_rows, min_slice)
            assert 0 <= start <= (total_rows - min_slice), f"Start {start} is out of valid range"
            assert min_slice <= stop <= total_rows, f"Stop {stop} is out of valid range"
            slice_size = stop - start
            assert min_slice <= slice_size <= remaining_rows, f"Slice size {slice_size} is out of valid range"
        
    def test_get_tuning_data(self):
        tuning_data = Tuner._get_tuning_data(self.data)
        assert tuning_data.shape[0] == 10000
        
    def test_slice_data(self):
        tuning_data = Tuner._slice_data(self.data, 100, 10000)
        assert tuning_data.shape[0] == 10000
        
    def test_remove_excess_rows(self):
        tuning_data = Tuner._remove_excess_rows(self.data, 10000)
        assert tuning_data.shape[0] == 10000
        
        tuning_data = Tuner._remove_excess_rows(self.data, 100)
        assert tuning_data.shape[0] == 100
        
        to_add = np.random.rand(100000, tuning_data.shape[1])
        tuning_data = np.concatenate((tuning_data, to_add))
        tuning_data = Tuner._remove_excess_rows(tuning_data, 100)
        assert tuning_data.shape[0] == 100
        
    def test_get_best_params(self):
        best_params, best_score = Tuner.get_best_params(self.data)
        params = config.get("PARAM_GRID")
        
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert best_params['timesteps'] in params['timesteps']
        assert best_params['learning_rate'] in params['learning_rate']
        assert best_params['epochs'] in params['epochs']
        assert best_params['batch_size'] in params['batch_size']
        for key, value in best_params.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
        assert best_params.keys() == params.keys()
        
        assert best_score is not None
        assert isinstance(best_score, (int, float))
        
    
    
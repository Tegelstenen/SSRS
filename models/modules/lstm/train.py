# TODO:
# - will certain groupings improve the model given that min max normalizations will be
#   dependent on, for instance, rainy windy days etc
from numpy import random, ndarray, concatenate, delete

import os
import logging

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from utils.helper import load_data, instantiate_model
from utils.log_setup import setup_logging

setup_logging()
config = ConfigManager()

class Trainer:
    def __init__(self) -> None:
        config = ConfigManager()
        self.data_path = f"{config.get('BASE_DIR')}/models/data/data.csv"

    def _train(self, data: ndarray, tune=True) -> LSTMAutoencoder:
        if tune:
            model = Tuner.tuned_model(data)
            model.fit(data)
        else:
            model = instantiate_model(data)
            model.fit(data)
        return model
    
    def _save_model(self, model: LSTMAutoencoder) -> None:
        base_dir = config.get("BASE_DIR")
        model_dir = os.path.join(base_dir, "models/model_info")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, ".weights.h5")
        model.save(model_path)
    
    def _get_best_params(self, data: ndarray) -> dict:
        best_params, best_score = self.get_best_params(data)
        return best_params, best_score
    
    def run(self):
        data = load_data(self.data_path)
        model = self._train(data)
        self._save_model(model)
        
class Tuner:
    # TODO: Make it select random trips instead
    @staticmethod
    def _get_tuning_data(data: ndarray) -> ndarray:
        target_rows = 10000
        min_slice = 100
        tuning_data = Tuner._slice_data(data, min_slice, target_rows)        
        tuning_data = Tuner._remove_excess_rows(tuning_data, target_rows)
        return tuning_data
    
    @staticmethod
    def _remove_excess_rows(data: ndarray, target_rows: int) -> ndarray:
        if data.shape[0] > target_rows:
            data = data[:target_rows]
        return data
    
    @staticmethod
    def _slice_data(data: ndarray, min_slice: int, target_rows: int) -> list:
        slices = []
        remaining_rows = target_rows
        total_rows = data.shape[0]
        while remaining_rows > 0 and total_rows > min_slice:
            start, stop = Tuner._get_start_stop(remaining_rows, total_rows, min_slice)
            slices.append(data[start:stop])
            data = delete(data, slice(start, stop), axis=0)
            slice_size = stop - start
            remaining_rows -= slice_size
            total_rows -= slice_size
        
        tuning_data = concatenate(slices)
        return tuning_data
    
    @staticmethod
    def _get_start_stop(remaining_rows: int, total_rows: int, min_slice: int) -> tuple:
        if remaining_rows < min_slice:
            slice_size = remaining_rows
            start = 0
        else:
            max_slice = min(remaining_rows, total_rows - min_slice)
            slice_size = random.randint(min_slice, max_slice)
            earliest_start = 0
            latest_start = total_rows - slice_size
            start = random.randint(earliest_start, latest_start)
        stop = start + slice_size
        return start, stop

    @staticmethod
    def tuned_model(data: ndarray) -> LSTMAutoencoder:
        tuning_data = Tuner._get_tuning_data(data)
        model = instantiate_model(tuning_data)
        model.tune_model(tuning_data)
        return model
        
    
    



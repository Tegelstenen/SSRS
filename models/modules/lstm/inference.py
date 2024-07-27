# TODO:
# Finish this shit to return the daily error things
# IDEA look at if the .evaluate function is better?
# Maybe add some tests

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import logging

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from utils.helper import load_data, instantiate_model
from utils.log_setup import setup_logging

setup_logging()

class LSTMInferencer:
    def __init__(self) -> None:
        self.config = ConfigManager()
        self.model_info_path = "models/model_info/.weights.h5"
        self.data_path = "models/data/data.csv"
        self.base_dir = self.config.get("BASE_DIR")
        self.features = self.config.get("MODEL_FEATURES")
        

    def _load_model(self, data: np.ndarray) -> LSTMAutoencoder:
        full_path = f"{self.base_dir}/{self.model_info_path}"
        model = instantiate_model(data)
        loaded_model = model.load(full_path)
        logging.infer("Model loaded")
        return loaded_model

    def _calculate_error(self, data: np.ndarray, predictions: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
        error = predictions
        error_df = pd.DataFrame(error, index=index, columns=[self.features])
        print(error_df)
        error = data
        error_df = pd.DataFrame(error, index=index, columns=[self.features])
        logging.infer("Error calculated")
        print(error_df)

    def run(self) -> np.ndarray:
        data, index = load_data(f"{self.base_dir}/{self.data_path}", for_inference=True)
        model = self._load_model(data)
        predictions = model.predict(data)
        errors = self._calculate_error(data, predictions, index)


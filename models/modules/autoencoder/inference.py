import pandas as pd
import numpy as np

import os

from utils.config_manager import ConfigManager
from modules.autoencoder.train import Trainer
from modules.autoencoder.autoencoder import Autoencoder
from utils.helper import load_data, instantiate_model



class Inferencer:
    def __init__(self):
        self.config = ConfigManager()
        self.trainer = Trainer()
        
        self.FEATURES = self.config.get("MODEL_FEATURES")

    def predict(self, raw_data, model):
        reconstructed_raw = model.reconstruct(raw_data)[0]
        return reconstructed_raw

    def _get_indeces(self):
        df = pd.read_csv("models/data/data.csv")
        
        df.drop(columns=self.FEATURES, inplace=True)
        return df
    
    @staticmethod
    def clean_instance(data):
        data[:, 0] = np.where(data[:, 0] < 0.5, 0, 1)
        return data

    def daily_mae(self, absolute_errors, index_data, features):
        errors_df = index_data.join(pd.DataFrame(absolute_errors, columns=features, index=index_data.index))
        grouped = errors_df.groupby(['node_name', 'date'])
        mean_errors_per_group = grouped.apply(lambda group: self.trainer.mean_error(group[features].values, features), include_groups=False)
        mean_errors_per_group = mean_errors_per_group.reset_index(level=[0, 1])
        
        return mean_errors_per_group

    def run(self):
        data_path = "models/data/data.csv"
        data, indeces = load_data(data_path, for_inference=True)
        autoencoder = Autoencoder.load("models/model_info")

        reconstructed_raw = self.predict(data, autoencoder)
        reconstructed_raw = self.clean_instance(reconstructed_raw)

        absolute_errors = self.trainer.absolute_error(data, reconstructed_raw)
        index_data = self._get_indeces()

        daily_error = self.daily_mae(absolute_errors, indeces, self.FEATURES)
        app_data_dir = self.config.get("APP_DATA_DIR")
        save_path = os.path.join(app_data_dir, "errors.csv")
        daily_error.to_csv(save_path, index=False)
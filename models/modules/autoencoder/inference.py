import pandas as pd
import numpy as np

import os

from utils.config_manager import ConfigManager
from modules.autoencoder.train import Trainer
from modules.autoencoder.autoencoder import Autoencoder

class Inferencer:
    def __init__(self):
        self.config = ConfigManager()
        self.trainer = Trainer()

    def load_data(self, path):
        dataframe = pd.read_csv(path)
        dataframe.dropna(inplace=True)
        model_features = self.config.get("MODEL_FEATURES")
        raw_data = dataframe[model_features].values
        indices = dataframe.drop(columns=model_features).columns
        index_data = dataframe[indices]
        raw_data = self.trainer.scale_data(raw_data)
        return index_data, raw_data, model_features 

    def predict(self, raw_data, model):
        reconstructed_raw = model.reconstruct(raw_data)[0]
        return reconstructed_raw

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
        index_data, raw_data, features = self.load_data(data_path)
        autoencoder = Autoencoder.load("models/model_info")

        reconstructed_raw = self.predict(raw_data, autoencoder)
        reconstructed_raw = self.clean_instance(reconstructed_raw)

        absolute_errors = self.trainer.absolute_error(raw_data, reconstructed_raw)

        daily_error = self.daily_mae(absolute_errors, index_data, features)
        app_data_dir = self.config.get("APP_DATA_DIR")
        save_path = os.path.join(app_data_dir, "errors.csv")
        daily_error.to_csv(save_path, index=False)
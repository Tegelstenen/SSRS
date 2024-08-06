import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from modules.autoencoder.autoencoder import Autoencoder
from utils.config_manager import ConfigManager
from utils.helper import load_data, instantiate_model


class Trainer:
    def __init__(self):
        self.config = ConfigManager()
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.epochs = 10
        

    def scale_data(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def train(self, data):
        model = instantiate_model(data)
        model.fit(data)
        return model

    @staticmethod
    def absolute_error(true, reconstructed):
        residuals = np.abs(true - reconstructed)
        return residuals

    @staticmethod
    def mean_error(error, features):
        mean_errors = np.mean(error, axis=0)

        mean_errors_df = pd.DataFrame(
            {
                **{
                    feature: [mean_error]
                    for feature, mean_error in zip(features, mean_errors)
                }
            }
        )

        return mean_errors_df

    def run(self):
        data_path = "models/data/data.csv"
        data = load_data(data_path)
        model = self.train(data)
        model.save("/models/tunings")

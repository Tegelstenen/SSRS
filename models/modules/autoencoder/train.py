# TODO:
# - will certain groupings improve the model given that min max normalizations will be
#   dependent on, for instance, rainy windy days etc

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from modules.autoencoder.autoencoder import Autoencoder
from utils.config_manager import ConfigManager


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

    def load_data(self, path):
        dataframe = pd.read_csv(path)
        dataframe.dropna(inplace=True)
        model_features = self.config.get("MODEL_FEATURES")
        raw_data = dataframe[model_features].values
        scaled_data = self.scale_data(raw_data)
        x_train, x_test = train_test_split(scaled_data, test_size=0.3, random_state=42)
        return x_train, x_test

    def train(self, x_train, x_test):
        autoencoder = Autoencoder(
            input_shape=x_train.shape[1:], hidden_layers=[128, 64], latent_space_dim=2
        )
        autoencoder.summary()
        autoencoder.compile(self.learning_rate)
        autoencoder.train(x_train, x_test, self.batch_size, self.epochs)
        return autoencoder

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
        x_train, x_test = self.load_data(data_path)
        autoencoder = self.train(x_train, x_test)
        autoencoder.save("models/model_info")

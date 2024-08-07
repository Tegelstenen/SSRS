import numpy as np

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from utils.lstm_utils import load_data, get_padded_sequence, numeric_features

config = ConfigManager()

class LSTMInferencer:
    def __init__(self):
        self.data_path = "/models/data/data.csv"
        self.model_path = "/models/tunings"
        self.output_path = "/dashboard/data/errors.csv"
    
    

    def run(self):
        df, _ = load_data(self.data_path, for_inference=True) # TODO: load the scaler
        sequences_padded, _ = get_padded_sequence(df, for_inference=True)
        input_shape = (sequences_padded.shape[1], sequences_padded.shape[2])
        autoencoder = LSTMAutoencoder(input_shape=input_shape)
        autoencoder.load(self.model_path)
        reconstructions = self._get_reconstructions(sequences_padded, autoencoder)
        sequences_padded, reconstructions = self._reshape(sequences_padded, reconstructions)
        reconstructions, original = self._remove_paddings(reconstructions, sequences_padded)
        residuals = self._get_residuals(original, reconstructions)
        df = self._get_daily_mse(df, residuals)
        df.to_csv(self.output_path)

    def _get_daily_mse(self, df, residuals):
        df[numeric_features] = residuals
        df = (df
              .groupby(["node_name", "date"])[numeric_features]
              .mean()
              .reset_index()
              )
        return df
        

    def _reshape(self, sequences_padded, reconstructions):
        reconstructions = np.reshape(reconstructions, (-1, reconstructions.shape[-1]))
        sequences_padded = np.reshape(sequences_padded, (-1, sequences_padded.shape[-1]))
        return sequences_padded,reconstructions

    def _remove_paddings(self, reconstructions, sequences_padded):
        # Remove all rows that contain -1 from reconstructions and sequences_padded
        mask = ~np.any(sequences_padded == -1, axis=1)
        reconstructions = reconstructions[mask]
        sequences_padded = sequences_padded[mask]
        return reconstructions, sequences_padded
    
    def _get_reconstructions(self, sequences_padded, autoencoder):
        reconstructions = autoencoder.predict(sequences_padded)
        return reconstructions
    
    def _get_residuals(self, sequences_padded, reconstructions):
        squared_errors = np.square(sequences_padded - reconstructions)
        return squared_errors
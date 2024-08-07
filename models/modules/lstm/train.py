import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import keras

import os
import logging
import pickle

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager

config = ConfigManager()

class Trainer:
    def __init__(self):
        self.data_path = "/models/data/data.csv"
        self.model_path = "/models/tunings"
        self.model_features = config.get("MODEL_FEATURES")
        self.scaling_features = config.get("ENGINE_FEATURES") + config.get("GEO_FEATURES")
        self.indices = config.get("INDEX") + ["TRIP_ID"]
        if "signal_instance" in self.indices:
            self.indices.remove("signal_instance")
        self.numeric_features = self.scaling_features + ["signal_instance"]

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        df = self._remove_outliers(df, self.scaling_features)
        df_numerical = self._get_numerical_values(df, self.scaling_features)
        scaler.fit(df_numerical)
        df_numerical = scaler.transform(df_numerical)
        df.loc[:, self.scaling_features] = pd.DataFrame(df_numerical, columns=self.scaling_features, index=df.index)
        return df, scaler

    def _get_numerical_values(self, df: pd.DataFrame, features: list):
        df_numerical = df[features].values
        return df_numerical

    def _remove_outliers(self, df: pd.DataFrame, features: list):
        df_engine = df[features]
        mask = (np.abs(stats.zscore(df_engine)) < 3).all(axis=1)
        df = df[mask]
        return df

    # TODO: move to utils a bunch of stuff
    def load_data(self, path):
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        df, scaler = self._scale_data(df)
        df = self._encode_signal_instance(df)
        df = df[self.model_features + self.indices]
        return df, scaler

    def _split_data_frame(self, df):
        df_numeric = df[self.numeric_features]
        df_indices = df[self.indices]
        return df_numeric, df_indices

    def _encode_signal_instance(self, df):
        df = df.copy()
        df.loc[:, 'signal_instance'] = df['signal_instance'].apply(lambda x: 1 if x == 'SB' else 0 if x == 'P' else "NAN")
        return df

    def _to_sequence(self, data: pd.DataFrame, indices: pd.DataFrame, group_cols=['node_name', 'TRIP_ID']):
        sequences = []
        grouped = indices.groupby(group_cols)
        for _, group in grouped:
            trip_data = data.loc[group.index, :].values
            sequences.append(trip_data)
        return sequences


    def save_scaler(self, scaler, path):
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def run(self):
        self._create_dirs()

        logging.train("Loading data...")
        df, scaler = self.load_data(self.data_path)

        logging.train("Splitting data into training and testing sets...")
        train_data, train_indeces, test_data, test_indeces = self._test_train_split_data(df)
        
        logging.train("Converting data to sequences...")
        train_sequences_padded, test_sequences_padded = self._get_padded_splits(train_data, train_indeces, test_data, test_indeces)

        logging.train("Initializing the autoencoder...")
        autoencoder = LSTMAutoencoder(input_shape=train_sequences_padded.shape[2])
        
        logging.train("Training the autoencoder...")
        history = autoencoder.train(train_sequences_padded, test_sequences_padded)

        logging.train(f"Saving the trained model to {self.model_path}...")
        autoencoder.save(self.model_path)
        
        logging.train(f"Saving the scalers to {self.model_path}...")
        self.save_scaler(scaler, self.model_path)
        
        logging.train("Training complete.")

    def _get_padded_splits(self, train_data, train_indeces, test_data, test_indeces):
        train_sequences = self._to_sequence(train_data, train_indeces)
        test_sequences = self._to_sequence(test_data, test_indeces)
        train_sequences_padded = keras.utils.pad_sequences(train_sequences, padding='post', dtype='float32', value=-1)
        test_sequences_padded = keras.utils.pad_sequences(test_sequences, padding='post', dtype='float32', value=-1)
        return train_sequences_padded,test_sequences_padded

    def _test_train_split_data(self, df):
        df['date'] = pd.to_datetime(df['date'])
        
        # Try different quantiles to ensure non-empty splits
        for quantile in [0.8, 0.7, 0.6, 0.5]:
            split_date = df['date'].quantile(quantile)
            train_data = df[df['date'] <= split_date]
            test_data = df[df['date'] > split_date]
            
            if not train_data.empty and not test_data.empty:
                break
        else:
            raise ValueError("Train or test split is empty. Adjust the split logic.")
        
        train_data, train_indeces = self._split_data_frame(train_data)
        test_data, test_indeces = self._split_data_frame(test_data)
        return train_data, train_indeces, test_data, test_indeces
    
    
    def _test_train_split_data(self, df):
        df['date'] = pd.to_datetime(df['date'])
        
        # Try different quantiles to ensure non-empty splits
        for quantile in [0.8, 0.7, 0.6, 0.5]:
            split_date = df['date'].quantile(quantile)
            train_data = df[df['date'] <= split_date]
            test_data = df[df['date'] > split_date]
            
            if not train_data.empty and not test_data.empty:
                break
        else:
            raise ValueError("Train or test split is empty. Adjust the split logic.")
        
        train_data, train_indeces = self._split_data_frame(train_data)
        test_data, test_indeces = self._split_data_frame(test_data)
        return train_data, train_indeces, test_data, test_indeces

    def _create_dirs(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
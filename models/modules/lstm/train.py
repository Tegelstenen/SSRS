import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.lstm.model import LSTMAutoencoder
import os
import logging
import pickle
import argparse
from utils.config_manager import ConfigManager

class Trainer:
    def __init__(self):
        self.data_path = "models/data/data.csv"
        self.model_path = "models/model_info"
        self.config = ConfigManager()
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.epochs = 500
        self.model_features = self.config.get("MODEL_FEATURES")
        
        self.alignment_features = ['alignment_factor']
        self.angle_features = ['COG', 'wind_angle']
        self.numeric_features = [f for f in self.model_features if f not in self.alignment_features and f not in self.angle_features and f not in ['signal_instance']]
        if 'wind_velocity' not in self.numeric_features:
            self.numeric_features.append('wind_velocity')

    def validate_data(self, df):
        nan_columns = df.columns[df.isnull().any()].tolist()
        if nan_columns:
            print(f"Columns with NaN values: {nan_columns}")
            for col in nan_columns:
                print(f"NaN count in {col}: {df[col].isnull().sum()}")
        assert not df.isnull().values.any(), "Data contains NaN values"
        assert not (df == np.inf).values.any(), "Data contains Inf values"
        assert not (df == -np.inf).values.any(), "Data contains -Inf values"
        return True

    def handle_missing_data(self, df):
        df = df.dropna()
        self.validate_data(df)
        return df


    def load_data(self, path):
        df = pd.read_csv(path)
        df = df.dropna(subset=['RPM'])
        df = self.handle_missing_data(df)

        logging.train("Scaling data")
        robust_scaler = RobustScaler()
        df[self.numeric_features] = robust_scaler.fit_transform(df[self.numeric_features])
        
        alignment_scaler = StandardScaler()
        df[self.alignment_features] = alignment_scaler.fit_transform(df[self.alignment_features])
        
        logging.train("Creating angle features")
        for feature in self.angle_features:
            if feature in df.columns:
                df[feature + '_sin'] = np.sin(np.radians(df[feature]))
                df[feature + '_cos'] = np.cos(np.radians(df[feature]))
                df.drop(columns=[feature], inplace=True)

        logging.train("Making signal instance binary")
        df['signal_instance'] = df['signal_instance'].apply(lambda x: 1 if x == 'SB' else 0 if x == 'P' else x)
        
        logging.train("Creating features")
        features = self.numeric_features + [f"{feat}_sin" for feat in self.angle_features] + [f"{feat}_cos" for feat in self.angle_features] + self.alignment_features + ['signal_instance']

        self.validate_data(df[features])
        return df, features, robust_scaler, alignment_scaler

    def to_sequence(self, data, features, trip_id_col='TRIP_ID'):
        sequences = []
        indices = []
        trip_ids = data[trip_id_col].unique()
        for trip_id in trip_ids:
            trip_data = data[data[trip_id_col] == trip_id][features].values
            sequences.append(trip_data)
            indices.append(data[data[trip_id_col] == trip_id].index)
        return sequences, indices

    def save_scalers(self, robust_scaler, alignment_scaler, path):
        with open(os.path.join(path, 'robust_scaler.pkl'), 'wb') as f:
            pickle.dump(robust_scaler, f)
        with open(os.path.join(path, 'alignment_scaler.pkl'), 'wb') as f:
            pickle.dump(alignment_scaler, f)

    def run(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        logging.basicConfig(level=logging.train)
        logging.train("Loading data...")

        try:
            data, features, robust_scaler, alignment_scaler = self.load_data(self.data_path)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return

        logging.train("Splitting data into training and testing sets...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        logging.train("Converting data to sequences...")
        train_sequences, train_indices = self.to_sequence(train_data, features)
        test_sequences, test_indices = self.to_sequence(test_data, features)
        train_sequences_padded = pad_sequences(train_sequences, padding='post', dtype='float32')
        test_sequences_padded = pad_sequences(test_sequences, padding='post', dtype='float32')

        logging.train("Initializing the autoencoder...")
        autoencoder = LSTMAutoencoder(input_shape=train_sequences_padded.shape[2])
        autoencoder.summary()
        
        logging.train("Training the autoencoder...")
        history = autoencoder.train(train_sequences_padded, test_sequences_padded, epochs=self.epochs, batch_size=self.batch_size)

        logging.train(f"Saving the trained model to {self.model_path}...")
        autoencoder.save(self.model_path)
        
        logging.train(f"Saving the scalers to {self.model_path}...")
        self.save_scalers(robust_scaler, alignment_scaler, self.model_path)
        
        logging.train("Training complete.")
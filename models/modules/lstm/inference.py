import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.lstm.model import LSTMAutoencoder
import os
import pickle
from utils.config_manager import ConfigManager

class LSTMInferencer:
    def __init__(self):
        self.data_path = "models/data"
        self.model_path = "models/model_info"
        self.output_path = "dashboard/data"
        self.config = ConfigManager()
        self.model_features = self.config.get("MODEL_FEATURES")
        self.event_measurements = self.config.get("EVENT_MEASUREMENTS")
        self.alignment_features = ['alignment_factor']
        self.angle_features = ['COG', 'wind_angle']
        self.numeric_features = [f for f in self.model_features if f not in self.event_measurements and f not in self.alignment_features and f not in self.angle_features]
        if 'wind_velocity' not in self.numeric_features:
            self.numeric_features.append('wind_velocity')

    def load_data(self, path):
        df = pd.read_csv(path)
        df = df.dropna(subset=['RPM'])

        # Extract only the date part from the time column
        df['date'] = pd.to_datetime(df['time']).dt.date

        # Normalize numeric features using RobustScaler
        with open(os.path.join(self.model_path, 'robust_scaler.pkl'), 'rb') as f:
            robust_scaler = pickle.load(f)
        df[self.numeric_features] = robust_scaler.transform(df[self.numeric_features])

        # Normalize alignment features using StandardScaler
        with open(os.path.join(self.model_path, 'alignment_scaler.pkl'), 'rb') as f:
            alignment_scaler = pickle.load(f)
        df[self.alignment_features] = alignment_scaler.transform(df[self.alignment_features])

        # Normalize angle features using sine and cosine transformation
        for feature in self.angle_features:
            if feature in df.columns:
                df[feature + '_sin'] = np.sin(np.radians(df[feature]))
                df[feature + '_cos'] = np.cos(np.radians(df[feature]))
                df.drop(columns=[feature], inplace=True)

        # Scale signal_instance
        df['signal_instance'] = df['signal_instance'].apply(lambda x: 1 if x == 'SB' else 0 if x == 'P' else x)

        # Other features
        features = self.numeric_features + [f"{feat}_sin" for feat in self.angle_features] + [f"{feat}_cos" for feat in self.angle_features] + self.alignment_features + ['signal_instance'] + self.event_measurements

        return df, features

    def to_sequence(self, data, features, trip_id_col='TRIP_ID'):
        sequences = []
        trip_ids = data[trip_id_col].unique()
        node_names = []
        dates = []
        for trip_id in trip_ids:
            trip_data = data[data[trip_id_col] == trip_id][features].values
            sequences.append(trip_data)
            node_names.append(data[data[trip_id_col] == trip_id]['node_name'].iloc[0])
            dates.append(data[data[trip_id_col] == trip_id]['date'].iloc[0])
        return sequences, trip_ids, node_names, dates

    def calculate_reconstruction_errors(self, original_sequences, reconstructed_sequences):
        errors = np.square(original_sequences - reconstructed_sequences)
        return errors

    def aggregate_errors(self, errors, trip_ids, features, node_names, dates):
        aggregated_data = []
        for i, trip_id in enumerate(trip_ids):
            trip_errors = errors[i]
            aggregated_row = {
                'node_name': node_names[i],
                'date': dates[i],
                'trip_id': trip_id,
                'overall_mse': trip_errors.mean()
            }
            for feature_index, feature in enumerate(features):
                aggregated_row[feature] = trip_errors[:, feature_index].mean() if trip_errors.ndim == 2 else trip_errors[feature_index].mean()

            aggregated_data.append(aggregated_row)

        return pd.DataFrame(aggregated_data)

    def run(self):
        data, features = self.load_data(self.data_path)
        sequences, trip_ids, node_names, dates = self.to_sequence(data, features)
        sequences_padded = pad_sequences(sequences, padding='post', dtype='float32')

        autoencoder = LSTMAutoencoder(input_shape=sequences_padded.shape[2])
        autoencoder.load(self.model_path)

        reconstructions = autoencoder.predict(sequences_padded)
        errors = self.calculate_reconstruction_errors(sequences_padded, reconstructions)

        aggregated_data = self.aggregate_errors(errors, trip_ids, features, node_names, dates)
        aggregated_data.dropna(inplace=True)
        aggregated_data.to_csv(self.output_path, index=False)

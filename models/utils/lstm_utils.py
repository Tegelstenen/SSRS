import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf

import pickle

from utils.config_manager import ConfigManager

config = ConfigManager()

data_path = "/models/data/data.csv"
model_path = "/models/tunings"
model_features = config.get("MODEL_FEATURES")
scaling_features = config.get("ENGINE_FEATURES") + config.get("GEO_FEATURES")
indices = config.get("INDEX") + ["TRIP_ID"]
if "signal_instance" in indices:
    indices.remove("signal_instance")
numeric_features = scaling_features + ["signal_instance"]

def create_dataset(sequences, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((sequences, sequences))  # Create (input, target) tuples
    dataset = dataset.shuffle(buffer_size=len(sequences))  # Shuffle the sequences
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_padded_sequence(df, for_inference=False):
    train_data, train_indeces, test_data, test_indeces = _test_train_split_data(df, for_inference=for_inference) 
    train_sequences_padded, test_sequences_padded = _get_padded_splits(train_data, train_indeces, test_data, test_indeces)
    return train_sequences_padded, test_sequences_padded
    

def _get_numerical_values(df: pd.DataFrame, features: list):
    df_numerical = df[features].values
    return df_numerical

def _remove_outliers(df: pd.DataFrame, features: list):
    df_engine = df[features]
    mask = (np.abs(stats.zscore(df_engine)) < 3).all(axis=1)
    df = df[mask]
    return df


def load_data(path, for_inference=False):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    
    if for_inference:
        with open('models/tunings/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df, _ = _scale_data(df, scaler)
    else:
        df = _remove_outliers(df, scaling_features)
        train_data, train_indeces, test_data, test_indeces = _test_train_split_data(df, for_inference)
        train_data, scaler = _scale_data(train_data, None)
        test_data, _ = _scale_data(test_data, scaler)
        train_data = pd.concat([train_data, train_indeces], axis=1)
        test_data = pd.concat([test_data, test_indeces], axis=1)
        df = pd.concat([train_data, test_data], axis=0)
    
    df = _encode_signal_instance(df)
    df = df[numeric_features + indices]
    return df, scaler

def _scale_data(df: pd.DataFrame, predefined_scaler: MinMaxScaler) -> pd.DataFrame:
    df_numerical = _get_numerical_values(df, scaling_features)
    
    if predefined_scaler is None:    
        scaler = MinMaxScaler() 
        scaler.fit(df_numerical)
    else:
        scaler = predefined_scaler
    
    df_numerical = scaler.transform(df_numerical)
    df.loc[:, scaling_features] = pd.DataFrame(df_numerical, columns=scaling_features, index=df.index)
    return df, scaler

def _split_data_frame(df):
    df_numeric = df[numeric_features]
    df_indices = df[indices]
    return df_numeric, df_indices

def _encode_signal_instance(df):
    df = df.copy()
    df.loc[:, 'signal_instance'] = df['signal_instance'].apply(lambda x: 1 if x == 'SB' else 0 if x == 'P' else "NAN")
    return df

def _to_sequence(data: pd.DataFrame, indices: pd.DataFrame, group_cols=['node_name', 'TRIP_ID']):
    sequences = []
    grouped = indices.groupby(group_cols)
    for _, group in grouped:
        trip_data = data.loc[group.index, :].values
        for i in range(0, len(trip_data), 30):
            sequence = trip_data[i:i+30]
            sequences.append(sequence)
    return sequences

        
def _get_padded_splits(train_data, train_indeces, test_data, test_indeces):
    train_sequences = _to_sequence(train_data, train_indeces)
    test_sequences = _to_sequence(test_data, test_indeces) if test_data is not None else []
    
    max_seq_length = max(
        max(len(seq) for seq in train_sequences) if train_sequences else 0,
        max(len(seq) for seq in test_sequences) if test_sequences else 0
    )
    
    train_sequences_padded = keras.utils.pad_sequences(train_sequences, maxlen=max_seq_length, padding='post', dtype='float32', value=-1)
    test_sequences_padded = keras.utils.pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', dtype='float32', value=-1) if test_sequences else None
    
    return train_sequences_padded,test_sequences_padded

def _test_train_split_data(df, for_inference=False):
    if not for_inference:
        split_date = '2024-06-01'
        train_data = df[df['date'] <= split_date]
        test_data = df[df['date'] > split_date]    
        
        train_data, train_indeces = _split_data_frame(train_data)
        test_data, test_indeces = _split_data_frame(test_data)
        return train_data, train_indeces, test_data, test_indeces
    else:
        train_data, train_indeces = _split_data_frame(df)
        return train_data, train_indeces, None, None
    

# TODO: Make sure this imoplementation works on aiqu
# TODO: Clean up inside inferer (given that it works inside aiqu)
# TODO: check that this implementation is alright for inference, if not change sequence length to be much smaller and pad to be within a day or so
# TODO: Check if adding differnet layers, like lstm as seen in https://machinelearningmastery.com/lstm-autoencoders/
# TODO: test the entire final df that is used for training such that it is conforming to what i was imagining
# TODO: fix the testings
# TODO: Add testings to like everything if time permits
# TODO: SPLIT DATA BY DATE TO INFER ON LATEST 2 MONTHS

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import keras

from modules.lstm.model import LSTMAutoencoder
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

def run():
    df, scaler = load_data(data_path, for_inference=True) # TODO: load the scaler
    sequences_padded, _ = get_padded_sequence(df, for_inference=True)
    autoencoder = LSTMAutoencoder(input_shape=sequences_padded.shape[2])
    autoencoder.load(model_path)

def get_padded_sequence(df, for_inference=False):
    train_data, train_indeces, test_data, test_indeces = _test_train_split_data(df, for_inference=for_inference) 
    train_sequences_padded, test_sequences_padded = _get_padded_splits(train_data, train_indeces, test_data, test_indeces)
    return train_sequences_padded, test_sequences_padded
    
def _scale_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df = _remove_outliers(df, scaling_features)
    df = _remove_outliers(df, scaling_features)
    df_numerical = _get_numerical_values(df, scaling_features)
    scaler.fit(df_numerical)
    df_numerical = scaler.transform(df_numerical)
    df.loc[:, scaling_features] = pd.DataFrame(df_numerical, columns=scaling_features, index=df.index)
    return df, scaler

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
    df, scaler = _scale_data(df)
    df = _encode_signal_instance(df)
    df = df[model_features + indices]
    if for_inference:
        scaler = None # TODOL replace with loading the scaler and scaling using that instead
        return df, None
    else:
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
        sequences.append(trip_data)
    return sequences

        
def _get_padded_splits(train_data, train_indeces, test_data, test_indeces):
    train_sequences = _to_sequence(train_data, train_indeces)
    train_sequences_padded = keras.utils.pad_sequences(train_sequences, padding='post', dtype='float32', value=-1)
    if test_data is not None and test_indeces is not None:
        test_sequences = _to_sequence(test_data, test_indeces)
        test_sequences_padded = keras.utils.pad_sequences(test_sequences, padding='post', dtype='float32', value=-1)
    else:
        test_sequences = None
        test_sequences_padded = None
    return train_sequences_padded,test_sequences_padded

def _test_train_split_data(df, for_inference=False):
    df['date'] = pd.to_datetime(df['date'])
    
    # Try different quantiles to ensure non-empty splits
    if not for_inference:
        for quantile in [0.8, 0.7, 0.6, 0.5]:
            split_date = df['date'].quantile(quantile)
            train_data = df[df['date'] <= split_date]
            test_data = df[df['date'] > split_date]
            
            if not train_data.empty and not test_data.empty:
                break
        else:
            raise ValueError("Train or test split is empty. Adjust the split logic.")
        train_data, train_indeces = _split_data_frame(train_data)
        test_data, test_indeces = _split_data_frame(test_data)
        return train_data, train_indeces, test_data, test_indeces
    else:
        train_data, train_indeces = _split_data_frame(df)
        return train_data, train_indeces, None, None
        
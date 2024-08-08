import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras

from pprint import pprint

from modules.lstm.inference import LSTMInferencer
from utils.lstm_utils import load_data, get_padded_sequence, numeric_features
from modules.lstm.model import LSTMAutoencoder


@pytest.fixture
def inferer():
    return LSTMInferencer()

def test_get_reconstructions(inferer: LSTMInferencer):
    df, _ = load_data("models/data/data.csv", for_inference=True) # TODO: load the scaler
    df = df.iloc[:10000, :]
    pprint(f"df.shape: {df.shape}")
    pprint(type(df))
    
    sequences_padded, _ = get_padded_sequence(df, for_inference=True)
    autoencoder = LSTMAutoencoder(input_shape=sequences_padded.shape[2])
    autoencoder.load("models/tunings")
    
    reconstructions = inferer._get_reconstructions(sequences_padded, autoencoder)
    pprint(f"reconstructions.shape: {reconstructions.shape}")

    sequences_padded, reconstructions = inferer._reshape(sequences_padded, reconstructions)
    pprint(f"_reshape reconstructions shape: {reconstructions.shape}")
    pprint(f"_reshape original shape: {sequences_padded.shape}")
    
    reconstructions, original = inferer._remove_paddings(reconstructions, sequences_padded)
    pprint(f"_remove_paddings reconstructions shape: {reconstructions.shape}")
    pprint(f"_remove_paddings original shape: {original.shape}")
    assert reconstructions.shape == df.shape
    assert original.shape == df.shape
    
    residuals = inferer._get_residuals(original, reconstructions)    
    assert len(numeric_features) == residuals.shape[1]
    
    pprint(df.columns)
    df = inferer._get_daily_mse(df, residuals)
    pprint(f"df.shape: {df.shape}")
    # Calculate the total number of dates for each node_name
    expected_row_count = df.groupby("node_name")["date"].nunique().sum()
    assert df.shape[0] == expected_row_count, f"Expected {expected_row_count} rows, but got {df.shape[0]}"
    assert 1 == 2
    
    
    
    
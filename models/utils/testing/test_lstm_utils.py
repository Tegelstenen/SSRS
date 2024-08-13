import pytest
import pandas as pd

from models.utils.lstm_utils import load_data, _test_train_split_data, _scale_data, _encode_signal_instance, numeric_features, indices



@pytest.fixture
def data():
    data = pd.read_csv("models/data/data.csv")
    return data

def test_test_train_split_data(data):
    data.dropna(inplace=True)
    train_data, train_indeces, test_data, test_indeces = _test_train_split_data(data)
    assert train_data is not None and not train_data.isna().any().any(), "Should not contain NAs"
    assert train_indeces is not None and not train_indeces.isna().any().any(), "Should not contain NAs"
    assert test_data is not None and not test_data.isna().any().any(), "Should not contain NAs"
    assert test_indeces is not None and not test_indeces.isna().any().any(), "Should not contain NAs"

def test_scale_data(data):
    data.dropna(inplace=True)
    scaled_data, scaler = _scale_data(data, None)
    assert scaled_data is not None
    assert not scaled_data.isna().any().any(), "Should not contain NAs"
    assert scaler is not None

def test_encode_signal_instance(data):
    data.dropna(inplace=True)
    encoded_data = _encode_signal_instance(data)
    assert encoded_data is not None and not encoded_data.isna().any().any(), "Should not contain NAs"
    encoded_data = encoded_data[numeric_features + indices]
    assert encoded_data is not None and not encoded_data.isna().any().any(), "Should not contain NAs after filtering"

def test_split(data):
    df = data
    df.dropna(inplace=True)

    train_data, train_indeces, test_data, test_indeces = _test_train_split_data(df)
    assert train_data is not None and not train_data.isna().any().any(), "Train data should not contain NAs after split"
    assert train_indeces is not None and not train_indeces.isna().any().any(), "Train indices should not contain NAs after split"
    assert train_data.shape[0] == train_indeces.shape[0], "Train data and train indices should have the same number of rows"
    
    assert test_data is not None and not test_data.isna().any().any(), "Test data should not contain NAs after split"
    assert test_indeces is not None and not test_indeces.isna().any().any(), "Test indices should not contain NAs after split"
    assert test_data.shape[0] == test_indeces.shape[0], "Test data and test indices should have the same number of rows"

    assert train_data.shape[1] == test_data.shape[1], "Train data and test data should have the same number of columns"
    assert train_indeces.shape[1] == test_indeces.shape[1], "Train data and test data should have the same number of columns"    
    
def test_scale(data):
    df = data
    df.dropna(inplace=True)
    train_data, train_indeces, test_data, test_indeces = _test_train_split_data(df)
    
    train_data, scaler = _scale_data(train_data, None)
    assert train_data is not None and not train_data.isna().any().any(), "Train data should not contain NAs after scaling"
    
    bfr_scale = test_data
    test_data, _ = _scale_data(test_data, scaler)
    assert test_data is not None and not test_data.isna().any().any(), "Test data should not contain NAs after scaling"

    # Check for NaNs in train_data and train_indeces before concatenation
    assert not train_data.isna().any().any(), "Train data should not contain NAs before concatenation"
    assert not train_indeces.isna().any().any(), "Train indices should not contain NAs before concatenation"
    assert train_indeces.shape[0] == train_data.shape[0], "Train indices should have the same number of rows as train data"

    
    train_data = pd.concat([train_data, train_indeces], axis=1)
    assert train_data is not None and not train_data.isna().any().any(), "Train data should not contain NAs after concatenation with indices"

    test_data = pd.concat([test_data, test_indeces], axis=1)
    assert test_data is not None and not test_data.isna().any().any(), "Test data should not contain NAs after concatenation with indices"

    df = pd.concat([train_data, test_data], axis=0)
    assert df is not None and not df.isna().any().any(), "Combined data should not contain NAs after concatenation"

    df = _encode_signal_instance(df)
    assert df is not None and not df.isna().any().any(), "Encoded data should not contain NAs"
    df = df[numeric_features + indices]
    assert df is not None and not df.isna().any().any(), "Filtered data should not contain NAs"
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras

from pprint import pprint

from modules.lstm.train import Trainer


@pytest.fixture
def padding_setup():
    seq1 = np.random.rand(10, 10)
    seq2 = np.random.rand(8, 10)
    seq3 = np.random.rand(6, 10)
    seq4 = np.random.rand(2, 10)
    sequences = [seq1, seq2, seq3, seq4]
    return sequences

def test_padding(padding_setup):
    sequences = padding_setup
    sequences_padded = keras.utils.pad_sequences(sequences, padding='post', dtype='float32')
    assert sequences_padded.shape == (4, 10, 10), "Padding failed"
    
@pytest.fixture
def trainer():
    return Trainer()

@pytest.fixture
def df():
    chunk_size = 10000 
    # np.random.seed(2)
    skip_rows = np.random.randint(0, 300000)  
    df = pd.read_csv("models/data/data.csv", nrows=chunk_size, skiprows=range(1, skip_rows))    
    assert df.shape[0] == chunk_size, "Data reading failed"
    return df

def test_remove_outliers(trainer: Trainer):
    feat = trainer.scaling_features
    
    matrix = np.random.rand(100, len(feat))
    random_row = np.random.randint(0, 10)
    random_col = np.random.randint(0, len(feat))
    matrix[random_row, random_col] = 1000
    
    df = pd.DataFrame(matrix, columns=feat)
    
    assert df.shape == (100, len(feat)), "Dataframe creation failed"
    assert df.columns.tolist() == feat, "Dataframe creation failed"
    df = trainer._remove_outliers(df, feat)
    assert df.shape == (99, len(feat)), "Outliers removal failed"
    

    
def test_get_continous_model_features(trainer: Trainer, df: pd.DataFrame):
    continous_model_features = trainer.scaling_features
    df_numerical = df[continous_model_features].values
    assert df_numerical.shape[0] == df.shape[0], "Numerical data retrieval failed"
    assert df_numerical.shape[1] == len(continous_model_features), "Numerical data retrieval failed"
    for col in df_numerical:
        assert np.issubdtype(col.dtype, np.number), "Numerical data retrieval failed"
    
    
def test_scale_data(trainer: Trainer, df: pd.DataFrame):
    continous_model_features = trainer.scaling_features
    
    df.dropna(axis=0, inplace=True)
    assert df.shape[0] != 0, "Dropping NaNs failed"
    
    df_scaled, scaler = trainer._scale_data(df)
    
    for col in continous_model_features:
        assert df_scaled[col].min().round(2) >= 0 and df_scaled[col].max().round(2) <= 1, f"Scaling failed for column {col}"
        
    assert df.shape[1] == df_scaled.shape[1], "Scaling failed"

def test_unique_scaling_features(trainer: Trainer):
    scaling_features = trainer.scaling_features
    assert len(scaling_features) == len(set(scaling_features)), "scaling_features contains duplicate column names."

def test_correct_scaling(trainer: Trainer):
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    trainer.scaling_features = ['feature1', 'feature2']
    
    df, scaler = trainer._scale_data(df)
    
    for col in trainer.scaling_features:
        assert df[col].min().round(2) >= 0 and df[col].max().round(2) <= 1, f"Scaling failed for column {col}"
    
def test_encode_signal_instance(trainer: Trainer):
    df = pd.DataFrame({
        'signal_instance': ['SB', 'P', 'SB', 'P', 'SB', 'P']
    })
    df = trainer._encode_signal_instance(df)
    assert df['signal_instance'].max() == 1, "Signal instance is not binary"
    assert df['signal_instance'].min() == 0, "Signal instance is not binary"
    assert df['signal_instance'].unique().tolist() == [1, 0], "Signal instance is not binary"
    
def test_load_data(trainer: Trainer):
    df, scaler = trainer.load_data("models/data/data.csv")
    assert df.shape[0] != 0, "Loading data failed"
    assert df.shape[1] == len(trainer.model_features) + len(trainer.indices), "Loading data failed"
    assert df.columns.tolist() == trainer.model_features + trainer.indices, "Loading data failed"
    assert isinstance(scaler, MinMaxScaler), "Loading data failed"
    
def test_test_train_split_data(trainer: Trainer, df: pd.DataFrame):
    train_data, train_indeces, test_data, test_indeces = trainer._test_train_split_data(df)
    
    assert train_data.shape[0] > test_data.shape[0], "Test train split failed"
    assert train_indeces.shape[0] > test_indeces.shape[0], "Test train split failed"
    
    assert train_data.shape[1] == test_data.shape[1], "Test train split failed"
    assert train_indeces.shape[1] == test_indeces.shape[1], "Test train split failed"

@pytest.fixture
def df_scaled(trainer: Trainer, df: pd.DataFrame):
    df.dropna(inplace=True)
    df, scaler = trainer._scale_data(df)
    df = trainer._encode_signal_instance(df)
    df = df[trainer.model_features + trainer.indices]
    return df

def test_to_sequence(trainer: Trainer, df_scaled: pd.DataFrame):
    train_data, train_indeces, test_data, test_indeces = trainer._test_train_split_data(df_scaled)

    # Debug: pprint the shapes of the split data
    pprint(f"Train data shape: {train_data.shape}")
    pprint(f"Train indices shape: {train_indeces.shape}")
    pprint(f"Test data shape: {test_data.shape}")
    pprint(f"Test indices shape: {test_indeces.shape}")
    pprint(f"Test data columns: {test_data.columns.tolist()}")
    pprint(f"Train data columns: {train_data.columns.tolist()}")
    pprint(f"Test indices columns: {test_indeces.columns.tolist()}")
    pprint(f"Train indices columns: {train_indeces.columns.tolist()}")

    train = trainer._to_sequence(train_data, train_indeces)
    test = trainer._to_sequence(test_data, test_indeces)

    # Debug: pprint the lengths of the sequences
    pprint(f"Number of train sequences: {len(train)}")
    pprint(f"Number of test sequences: {len(test)}")

    # Check that the total number of sequences matches the total number of unique groups
    total_sequences = len(train) + len(test)
    unique_groups = df_scaled.groupby(['node_name', 'TRIP_ID']).ngroups
    pprint(f"Total sequences: {total_sequences}, Unique groups: {unique_groups}")
    assert total_sequences == unique_groups, "Total number of sequences does not match the number of unique groups"

    # Debug: Print unique groups in train and test sets
    train_unique_groups = train_indeces.groupby(['node_name', 'TRIP_ID']).ngroups
    test_unique_groups = test_indeces.groupby(['node_name', 'TRIP_ID']).ngroups
    pprint(f"Train unique groups: {train_unique_groups}")
    pprint(f"Test unique groups: {test_unique_groups}")

    # Check that the split ratio is approximately correct
    split_ratio = 0.2  # Assuming a test size of 20%
    expected_test_size = int(unique_groups * split_ratio)
    expected_train_size = unique_groups - expected_test_size

    # Adjust the tolerance for the split ratio check
    tolerance = max(1, int(0.1 * unique_groups))  # Allow a 10% tolerance or at least 1

    assert abs(len(test) - expected_test_size) <= tolerance, f"Test set size is not as expected: {len(test)} != {expected_test_size}"
    assert abs(len(train) - expected_train_size) <= tolerance, f"Train set size is not as expected: {len(train)} != {expected_train_size}"
    
def test_get_padded_splits(trainer: Trainer, df_scaled: pd.DataFrame):
    train_data, train_indeces, test_data, test_indeces = trainer._test_train_split_data(df_scaled)
    
    # Debug: Print the number of unique groups in train and test sets
    train_unique_groups = train_indeces.groupby(['node_name', 'TRIP_ID']).ngroups
    test_unique_groups = test_indeces.groupby(['node_name', 'TRIP_ID']).ngroups
    pprint(f"Train unique groups: {train_unique_groups}")
    pprint(f"Test unique groups: {test_unique_groups}")
    
    train_sequences = trainer._to_sequence(train_data, train_indeces)
    if not train_sequences:
        raise ValueError("Train sequences are empty")
    
    max_length_sequence = max(train_sequences, key=len)
    max_length_train = len(max_length_sequence)
    pprint(f"Maximum sequence length in train_data: {max_length_train}")

    test_sequences = trainer._to_sequence(test_data, test_indeces)
    if not test_sequences:
        raise ValueError("Test sequences are empty")
    
    max_length_sequence = max(test_sequences, key=len)
    max_length_test = len(max_length_sequence)
    pprint(f"Maximum sequence length in test_data: {max_length_test}")

    train_padded, test_padded = trainer._get_padded_splits(train_data, train_indeces, test_data, test_indeces)
    pprint(f"shape of train_padded: {train_padded.shape}")
    pprint(f"shape of test_padded: {test_padded.shape}")
    assert train_padded.shape[0] == train_indeces["TRIP_ID"].nunique()
    assert test_padded.shape[0] == test_indeces["TRIP_ID"].nunique()
    assert train_padded.shape[1] == max_length_train
    assert test_padded.shape[1] == max_length_test
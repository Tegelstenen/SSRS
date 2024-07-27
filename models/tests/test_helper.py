import numpy as np
import pandas as pd

from utils.helper import (
    load_data,
    instantiate_model,
    _get_sequences,
    _scale,
    _encode_all,
    _encode_binary,
)
from utils.config_manager import ConfigManager

config = ConfigManager()


class TestHelperFunctions:
    def setup_method(self):
        np.random.seed(42)
        self.dataframe, self.encoded, self.sequences, self.scaled_data = TestHelperFunctions.create_test_data()
        self.data_path = f"{config.get('BASE_DIR')}/models/data/data.csv"

    @staticmethod
    def create_test_data(): 
        features = config.get("MODEL_FEATURES")
        cols = len(features) - len(config.get("CATEGORICAL_FEATURES"))
        rows = np.random.randint(10001, 30002)
        
        categorical_binary = pd.DataFrame(np.where(np.random.randint(2, size=rows) == 1, "SB", "P"), columns=["signal_instance"])
        features = [feature for feature in features if feature != "signal_instance"]
        numerical = pd.DataFrame(np.random.rand(rows, cols), columns=features)
        
        dataframe = pd.concat([numerical, categorical_binary], axis=1)
        dataframe.columns = features + ["signal_instance"]
        encoded = _encode_all(dataframe)
        sequences = _get_sequences(encoded)
        scaled_data = _scale(sequences)
        return dataframe, encoded, sequences, scaled_data

    def test_encode(self):
        encoded_binary = _encode_binary(self.dataframe, "signal_instance")
        assert all(encoded_binary["signal_instance"].isin([0, 1]))
        
        assert all(self.encoded["signal_instance"].isin([0, 1]))

    def test_get_sequence(self):
        assert self.sequences is not None

        rows_it_should_return = self.dataframe.shape[0]
        nrows = self.sequences.shape[0]
        assert nrows == rows_it_should_return

        cols_it_should_return = len(config.get("MODEL_FEATURES"))
        ncols = self.sequences.shape[1]
        assert ncols == cols_it_should_return

    def test_scale(self):
        assert self.scaled_data is not None

        assert np.min(self.scaled_data.round(1)) == 0
        assert np.max(self.scaled_data.round(1)) == 1

    def test_load_data(self):
        data = load_data(self.data_path)
        assert data is not None

    def test_instantiate_model(self):
        model = instantiate_model(self.scaled_data)
        assert model is not None
        assert model.full_length_of_dataset == self.scaled_data.shape[0]
        assert model.features == len(config.get("MODEL_FEATURES"))
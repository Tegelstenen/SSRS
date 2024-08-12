import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from utils.config_manager import ConfigManager
from modules.autoencoder.autoencoder import Autoencoder

config = ConfigManager()
model_features = config.get("MODEL_FEATURES")
categorical_features = config.get("CATEGORICAL_FEATURES")
index = config.get("INDEX")

# ----------------------------------#
#        model instansiator         #
# ----------------------------------#
def instantiate_model(data: np.ndarray) -> Autoencoder:
    model = Autoencoder((data.shape[1],), [128, 64], 2)
    return model


# ----------------------------------#
#    load and preprocess dataset    #
# ----------------------------------#
def load_data(path: str, for_inference=False) -> np.ndarray:
    dataframe = pd.read_csv(path)
    dataframe.dropna(inplace=True)
    dataframe.sort_values(by=index, inplace=True)
    dataframe = _encode_all(dataframe)

    data = _get_sequences(dataframe)
    data = _scale(data)
    if for_inference:
        indeces = dataframe.drop(columns=model_features)
        return data, indeces
    return data
    
def _get_sequences(dataframe: pd.DataFrame) -> np.ndarray:
    data = dataframe[model_features].values
    return data

def _scale(data: np.ndarray) -> tuple:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

def _encode_binary(dataframe: pd.DataFrame, feature: str) -> pd.DataFrame:
    encoder = LabelEncoder()
    dataframe[feature] = encoder.fit_transform(dataframe[feature])
    return dataframe

def _encode_all(dataframe: pd.DataFrame) -> pd.DataFrame:
    for feature in categorical_features:
        if dataframe[feature].nunique() == 2:  
            dataframe = _encode_binary(dataframe, feature)
    return dataframe


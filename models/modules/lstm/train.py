import os
import logging
import pickle

from modules.lstm.model import LSTMAutoencoder
from utils.config_manager import ConfigManager
from utils.lstm_utils import load_data, get_padded_sequence, create_dataset

config = ConfigManager()

class Trainer:  
    def __init__(self):
        self.data_path = "/models/data/data.csv"
        self.model_path = "/models/tunings"

    def save_scaler(self, scaler, path):
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def run(self):
        self._create_dirs()
        df, scaler = load_data(self.data_path)
        train_sequences_padded, test_sequences_padded= get_padded_sequence(df)
        input_shape = (train_sequences_padded.shape[1], train_sequences_padded.shape[2])
        autoencoder = LSTMAutoencoder(input_shape=input_shape)
        train_dataset = create_dataset(train_sequences_padded, batch_size=512)
        test_dataset = create_dataset(test_sequences_padded, batch_size=512)
        
        logging.train("Training the autoencoder...")
        history = autoencoder.train(train_dataset, test_dataset)

        logging.train(f"Saving the trained model to {self.model_path}...")
        autoencoder.save(self.model_path)
        
        logging.train(f"Saving the scalers to {self.model_path}...")
        self.save_scaler(scaler, self.model_path)
        
        logging.train("Training complete.")
                    
    def _create_dirs(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
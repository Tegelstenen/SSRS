import pandas as pd
import numpy as np

from pathlib import Path
import logging
import os

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging
from utils.helper import HelperFunctions

class DataSequencer:
    def __init__(self):
        setup_logging()
        self.config = ConfigManager()
        self.directory = HelperFunctions.get_data_folder()
        self.run(self.directory)

    def get_mean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean values at specified intervals for the already pivoted data."""
        sequence_length = self.config.get("SEQUENCE_LENGTH", "5s")
        df["date"] = pd.to_datetime(df["time"]).dt.date
        df["time"] = pd.to_datetime(df["time"]).dt.round(sequence_length).dt.strftime("%H:%M:%S")
        
        if "signal_instance" in df.columns:
            df["signal_instance"] = df["signal_instance"].fillna("unknown")

        index = [col for col in self.config.get("INDEX", []) if col in df.columns]

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['date', 'time']]

        df = df.groupby(index)[numeric_columns].mean().reset_index()

        return df

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from a CSV file, handling empty files."""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logging.warning(f"The file {file_path} is empty.")
                return pd.DataFrame()
            logging.pipeline(f"Loaded processed data from {file_path}")
            return df
        except pd.errors.EmptyDataError:
            logging.warning(f"The file {file_path} is empty or has no columns.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame to a CSV file."""
        df.to_csv(file_path, index=False)
        logging.pipeline(f"Saved processed data to {file_path}")

    def run(self, directory: Path) -> None:
        logging.info(f"Sequencing data into even periods")
        geo_path = os.path.join(directory, "geo_data.csv")
        engine_path = os.path.join(directory, "engine_data.csv")

        geo_df = self.load_data(geo_path)
        engine_df = self.load_data(engine_path)

        if geo_df.empty and engine_df.empty:
            logging.warning("Both geo_data.csv and engine_data.csv are empty. No processing needed.")
            return
        
        if not geo_df.empty:
            geo_df = self.get_mean_values(geo_df)
            self.save_data(geo_df, geo_path)
        
        if not engine_df.empty:
            engine_df = self.get_mean_values(engine_df)
            self.save_data(engine_df, engine_path)
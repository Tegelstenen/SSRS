import pandas as pd

import logging
import os
from pathlib import Path

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging
from utils.helper import HelperFunctions

class DataMerger:
    def __init__(self):
        setup_logging()
        self.config = ConfigManager()
        self.directory = HelperFunctions.get_data_folder()
        self.run(self.directory)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logging.pipeline(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise

    def preprocess_geo_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the time column in geo data."""
        try:
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
            df['time'] = df['time'].dt.strftime('%H:%M:%S')
            logging.pipeline("Geo data time column preprocessed successfully")
            df.drop(columns=['signal_instance'], inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error preprocessing geo data time: {e}")
            raise

    def merge_data(self, engine: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
        """Merge engine and geo data."""
        try:
            geo.drop(columns="TRIP_ID", inplace=True)
            merge_columns = ['node_name', 'date', 'time']
            merged_df = pd.merge(engine, geo, on=merge_columns, how='left', suffixes=('_engine', '_geo'))
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            logging.pipeline("Data merged successfully")
            return merged_df
        except Exception as e:
            logging.error(f"Error merging data: {e}")
            raise

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save dataframe to CSV file."""
        try:
            df.to_csv(file_path, index=False)
            logging.pipeline(f"Data saved successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise

    def run(self, directory: Path) -> None:
        """Main function to orchestrate the data processing pipeline."""
        logging.info(f"Merging data")
        try:
            geo_file = os.path.join(directory, "geo_data.csv")
            engine_file = os.path.join(directory, "engine_data.csv")
            merged_file = os.path.join(directory, "data.csv")
            
            engine = self.load_data(engine_file)
            geo = self.load_data(geo_file)
            
            geo = self.preprocess_geo_time(geo)
            
            merged_df = self.merge_data(engine, geo)
            
            self.save_data(merged_df, merged_file)
            logging.pipeline(f"Merged data saved to {merged_file}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
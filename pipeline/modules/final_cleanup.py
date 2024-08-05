import pandas as pd
from pathlib import Path

import logging
import os

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging
from utils.helper import HelperFunctions

setup_logging()
config = ConfigManager()

class DataCleaner:
    def __init__(self):
        self.directory = HelperFunctions.get_data_folder()
        self.run(self.directory)

    def run(self, directory: Path) -> None:
        df = self._load_Data(directory)    
        timespans = self._get_timespans(df)
        filtered = self._filter_data(df, timespans)
        self._save_data(filtered, directory)

    def _filter_data(self, df: pd.DataFrame, timespans: pd.DataFrame) -> pd.DataFrame:
        filtered = df.merge(timespans[['node_name', 'TRIP_ID']], on=["node_name", "TRIP_ID"])
        return filtered

    def _get_timespans(self, df: pd.DataFrame) -> pd.DataFrame:
        timespans = (df
        .groupby(["node_name", "TRIP_ID"])
        .agg(time_min=('time', 'min'), time_max=('time', 'max'))
        .reset_index()
        .assign(timespan=lambda x: (pd.to_datetime(x['time_max']) - pd.to_datetime(x['time_min'])).dt.total_seconds() / 60)
        .query("timespan > 10")
        )
        return timespans
        
        
    def _save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        df.to_csv(file_path, index=False)
        logging.pipeline(f"Saved processed data to {file_path}")

    def _load_Data(self, directory: str) -> pd.DataFrame:
        data_path = os.path.join(directory, "data.csv")
        df = pd.read_csv(data_path)
        return df
        
        








import pandas as pd
from pathlib import Path

import logging
import re
import difflib
import os

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging
from utils.helper import HelperFunctions

class DataWidener:
    def __init__(self):
        setup_logging()
        self.config = ConfigManager()
        self.directory = HelperFunctions.get_data_folder()
        self.run(self.directory)

    def regex_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        model_features = self.config.get("TO_QUERY", [])
        cleaned_features = {re.sub(r'\d', '', feature.replace('_', '').lower()): feature for feature in model_features}
        
        def find_best_match(alias):
            cleaned_alias = re.sub(r'\d', '', alias.replace('_', '').lower())
            if cleaned_alias in cleaned_features:
                return cleaned_features[cleaned_alias]
            matches = difflib.get_close_matches(cleaned_alias, cleaned_features.keys(), n=1, cutoff=0.6)
            return cleaned_features[matches[0]] if matches else alias

        unique_aliases = df['signal_name_alias'].unique()
        alias_map = {alias: find_best_match(alias) for alias in unique_aliases}
        df['signal_name_alias'] = df['signal_name_alias'].map(alias_map)
        
        return df

    def process_data(self, file_path: Path) -> pd.DataFrame:
        logging.pipeline(f"Processing data from {file_path}")
        chunk_size = self.config.get('CHUNK_SIZE')
        chunks = []
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                logging.pipeline(f"Read chunk with shape: {chunk.shape}")
                logging.pipeline(f"Chunk columns: {chunk.columns.tolist()}")
                
                chunk = self.regex_fix(chunk)
                chunk["date"] = pd.to_datetime(chunk["time"]).dt.date
                chunks.append(chunk)

            if not chunks:
                logging.warning(f"No data read from {file_path}")
                return pd.DataFrame()
            
            df = pd.concat(chunks, ignore_index=True)
            logging.pipeline(f"Concatenated DataFrame shape: {df.shape}")
            logging.pipeline(f"Unique values in signal_name_alias after regex_fix: {df['signal_name_alias'].unique()}")
            
            df = self.pivot_wider(df)
            logging.pipeline(f"DataFrame shape after pivot_wider: {df.shape}")
            logging.pipeline(f"Columns after pivot_wider: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()

    def pivot_wider(self, df: pd.DataFrame) -> pd.DataFrame:
        geo_columns = ['SOG', 'COG', 'LON', 'LAT']
        is_geo_data = any(col in df['signal_name_alias'].unique() for col in geo_columns)
        
        if is_geo_data:
            df = df.drop_duplicates(subset=['node_name', 'signal_instance', 'date', 'time', 'signal_name_alias'])
            df = df.pivot(
                index=['node_name', 'signal_instance', 'date', 'time'],
                columns='signal_name_alias',
                values='value'
            ).reset_index()
        else:
            df = df.pivot_table(
                index=['node_name', 'signal_instance', 'date', 'time'],
                columns='signal_name_alias',
                values='value',
                aggfunc='first'
            ).reset_index()
        
        return df

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        df.to_csv(file_path, index=False)
        logging.pipeline(f"Saved processed data to {file_path}")

    def run(self, directory: Path) -> None:
        logging.info(f"Widening data")
        geo_path = os.path.join(directory, "geo_data.csv")
        engine_path = os.path.join(directory, "engine_data.csv")

        geo_df = self.process_data(geo_path)
        engine_df = self.process_data(engine_path)

        self.save_data(geo_df, geo_path)
        self.save_data(engine_df, engine_path)
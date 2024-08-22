import os
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging

config = ConfigManager()

class RawDataProcessor:
    def __init__(self):
        setup_logging()
        self.directory = config.get("TEMP_DATA_DIR")
        self.column_names: List[str] = config.get('COLUMN_NAMES')
        self.col_types: Dict[str, str] = config.get('COLUMN_TYPES')
        self.run()

    def run(self):
        logging.info(f"Processing raw data")
        logging.pipeline(f"Working from directory: {self.directory}")
        geo_df_list, engine_df_list = self._process_directory()
        self._save_processed_data(geo_df_list, engine_df_list)

    def _is_geo_data(self, df: pd.DataFrame) -> bool:
        return df["signal_instance"].isna().all()

    def _load_csv(self, csv_file: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(csv_file):
            logging.warning(f"File does not exist: {csv_file}")
            return None

        try:
            df = pd.read_csv(
                csv_file,
                names=self.column_names,
                dtype={**self.col_types, 'time': str},  
                skiprows=1,
            )
            
            df['time'] = pd.to_datetime(df['time'], format='ISO8601')
            return df
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
            return None

    def _find_csv_files(self) -> List[str]:
        all_csv_files = [str(path) for path in Path(self.directory).rglob('*.csv')]
        raw_files = [file for file in all_csv_files if Path(file).name not in ['geo_data.csv', 'engine_data.csv']]
        
        if raw_files:
            return raw_files
        elif len(all_csv_files) == 2 and all(file.endswith(('geo_data.csv', 'engine_data.csv')) for file in all_csv_files):
            logging.pipeline("Found only geo_data.csv and engine_data.csv. These will be processed.")
            return all_csv_files
        else:
            logging.warning("No CSV files found to process.")
            return []

    def _process_file(self, csv_file: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        some_df = self._load_csv(csv_file)
        if some_df is not None:
            if self._is_geo_data(some_df):
                return (some_df, None)
            else:
                some_df["signal_instance"] = some_df["signal_instance"].apply(
                    lambda x: "P" if x == "0" else ("SB" if x == "1" else x)
                )
                return (None, some_df)
        return (None, None)

    @staticmethod
    def process_file_wrapper(column_names, col_types, csv_file):
        try:
            df = pd.read_csv(
                csv_file,
                names=column_names,
                dtype={**col_types, 'time': str},  
                skiprows=1,
            )
            
            df['time'] = pd.to_datetime(df['time'], format='ISO8601')
            
            if df["signal_instance"].isna().all():
                return (df, None)
            else:
                df["signal_instance"] = df["signal_instance"].apply(
                    lambda x: "P" if x == "0" else ("SB" if x == "1" else x)
                )
                return (None, df)
        except Exception as exc:
            logging.error(f'{csv_file} generated an exception: {exc}')
            return (None, None)

    def _process_directory(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        csv_files = self._find_csv_files()
        total_files = len(csv_files)
        logging.pipeline(f"Found {total_files} CSV files in {self.directory}")

        geo_df_list = []
        engine_df_list = []
        error_count = 0

        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(self.process_file_wrapper, self.column_names, self.col_types, csv_file): csv_file for csv_file in csv_files}
            for i, future in enumerate(as_completed(future_to_file), 1):
                csv_file = future_to_file[future]
                try:
                    geo_df, engine_df = future.result()
                    if geo_df is not None:
                        geo_df_list.append(geo_df)
                    elif engine_df is not None:
                        engine_df_list.append(engine_df)
                    else:
                        error_count += 1
                except Exception as exc:
                    logging.error(f'{csv_file} generated an exception: {exc}')
                    error_count += 1

                if total_files > 0 and (i % max(1, total_files // 10) == 0 or i == total_files):
                    progress = (i / total_files) * 100
                    logging.pipeline(f"Progress: {progress:.1f}% - Processed {i}/{total_files} files")

        logging.pipeline(f"Directory summary for {self.directory}:")
        logging.pipeline(f"  - Processed {len(geo_df_list)} geo files and {len(engine_df_list)} engine files")
        logging.pipeline(f"  - Encountered errors in {error_count} files")

        return geo_df_list, engine_df_list

    def _save_processed_data(self, geo_df_list: List[pd.DataFrame], engine_df_list: List[pd.DataFrame]):
        logging.pipeline(f"Total geo dataframes: {len(geo_df_list)}")
        logging.pipeline(f"Total engine dataframes: {len(engine_df_list)}")

        if not geo_df_list:
            logging.warning("No geo data found. Skipping geo data concatenation.")
        else:
            geo_df = pd.concat(geo_df_list, ignore_index=True)
            logging.pipeline(f"Concatenated geo data shape: {geo_df.shape}")
            geo_output_path = f"{self.directory}/geo_data.csv"
            geo_df.to_csv(geo_output_path, index=False)
            logging.pipeline(f"Wrote geo data to {geo_output_path}")

        if not engine_df_list:
            logging.warning("No engine data found. Skipping engine data concatenation.")
        else:
            engine_df = pd.concat(engine_df_list, ignore_index=True)
            logging.pipeline(f"Concatenated engine data shape: {engine_df.shape}")
            engine_output_path = f"{self.directory}/engine_data.csv"
            engine_df.to_csv(engine_output_path, index=False)
            logging.pipeline(f"Wrote engine data to {engine_output_path}")
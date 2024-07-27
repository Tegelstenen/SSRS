# TODO:
# - Decide more on cleaning
# - Extend with more features

import argparse
import logging
from pathlib import Path
import os
import shutil

from utils.config_manager import ConfigManager
from utils.helper import HelperFunctions
from utils.log_setup import setup_logging

from modules.db_query import DataBaseQuery
from modules.impute import DataImputer
from modules.merge import DataMerger
from modules.process_raw import RawDataProcessor
from modules.sequence import DataSequencer
from modules.weather import WeatherProcessor
from modules.widen import DataWidener


# Load config.get(uration
config = ConfigManager()
setup_logging()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument("start_date", help="Start date in YYYY-MM-DD format")
    parser.add_argument("end_date", help="End date in YYYY-MM-DD format")
    parser.add_argument("save_path", help="Path to save the processed data")
    parser.add_argument("--only_query", action="store_true", help="Will only call the DB query")
    return parser.parse_args()

def run_steps(start: str, stop: str, only_query=False) -> None:
    DataBaseQuery(start, stop)
    if not only_query:
        RawDataProcessor()
        DataWidener()
        DataSequencer()
        DataImputer()
        DataMerger()
        WeatherProcessor()

def move_data(save_path: str) -> None:
    logging.info(f"Moving data to {save_path}")
    move_path = os.path.join(config.get('BASE_DIR'), save_path)
    data_folder = HelperFunctions.get_data_folder()
    merged_file = os.path.join(data_folder, "data.csv")
    destination_file = os.path.join(move_path, "data.csv")
    os.makedirs(move_path, exist_ok=True)
    
    if not os.path.exists(merged_file):
        logging.error(f"Source file not found: {merged_file}")
        raise FileNotFoundError(f"No such file or directory: '{merged_file}'")
    
    try:
        shutil.move(merged_file, destination_file)
        logging.info(f"Successfully moved data to {destination_file}")
    except Exception as e:
        logging.error(f"Error moving file from {merged_file} to {destination_file}: {str(e)}")
        raise

def remove_data_folder(silence=False) -> None:
    if not silence:
        logging.info(f"Removing raw data folder")
    data_folder = HelperFunctions.get_data_folder()
    try:
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
    except Exception as e:
        logging.error(f"Error removing directory {data_folder}: {str(e)}")

def remove_pycache_dirs(silence=False) -> None:
    if not silence:
        logging.info(f"Removing pycache")
    base_dir = config.get('BASE_DIR')
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path)
            

def clean_directory(silence=False) -> None:
    remove_data_folder(silence)
    remove_pycache_dirs(silence)

def main() -> None:
    try:
        args = get_args()
        if(args.only_query):
            clean_directory(silence=True)
            run_steps(args.start_date, args.end_date, only_query=True)    
        else:
            run_steps(args.start_date, args.end_date)
            move_data(args.save_path)
            clean_directory()
    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
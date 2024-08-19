import argparse
import logging
import os
import shutil
from pprint import pprint

from utils.config_manager import ConfigManager

from utils.log_setup import setup_logging
from modules.db_query import DataBaseQuery, InfluxDBClient
from modules.impute import DataImputer
from modules.merge import DataMerger
from modules.process_raw import RawDataProcessor
from modules.sequence import DataSequencer
from modules.weather import WeatherProcessor
from modules.widen import DataWidener
from modules.final_cleanup import DataCleaner

config = ConfigManager()
setup_logging()

def get_args() -> argparse.Namespace:
    '''This argument set up ensures you either show all tables, get a sample from a table, or make a query'''
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for show_tables
    subparsers.add_parser("show_tables", help="Will show all available tables")  # Removed '--'

    # Subparser for sample
    sample_parser = subparsers.add_parser("sample", help="Prints two rows of the given table")  # Removed '--'
    sample_parser.add_argument("table_name", type=str, help="Name of the table to sample")

    # Subparser for query
    query_parser = subparsers.add_parser("query", help="Run a query with start_date, end_date, and save_path")  # Removed '--'
    query_parser.add_argument("start_date", help="Start date in YYYY-MM-DD format")
    query_parser.add_argument("end_date", help="End date in YYYY-MM-DD format")
    query_parser.add_argument("save_path", help="Path to save the processed data")

    return parser.parse_args()

def run_steps(start: str, stop: str) -> None:
    db = DataBaseQuery(start, stop)
    db.run()

    RawDataProcessor()
    DataWidener()
    DataSequencer()
    DataImputer()
    DataMerger()
    WeatherProcessor()
    DataCleaner()

def move_data(save_path: str) -> None:
    logging.info(f"Moving data to {save_path}")
    move_path = os.path.join(config.get('BASE_DIR'), save_path)
    data_folder = config.get("TEMP_DATA_DIR")
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

def remove_data_folder(e) -> None:
    logging.info(f"Removing raw data folder")
    data_folder = config.get("TEMP_DATA_DIR")
    try:
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
    except Exception as e:
        logging.error(f"Error removing directory {data_folder}: {str(e)}")

def main() -> None:
    try:
        args = get_args()
        if args.command == "show_tables": 
            db = InfluxDBClient()
            df = db.is_connected(True)
            pprint(df.tolist())
        elif args.command == "sample": 
            db = InfluxDBClient()
            query = f"SELECT * FROM '{args.table_name}' LIMIT 2"
            df = db.query(query)
            pprint(df)
        elif args.command == "query": 
            run_steps(args.start_date, args.end_date)
            move_data(args.save_path)
            remove_data_folder()
    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
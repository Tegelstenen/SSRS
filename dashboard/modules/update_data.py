
# TODO: Add loggings
import pandas as pd

import pathlib
from typing import Tuple
from datetime import datetime
import subprocess

import logging

def update_data():
    """
    Update the existing data with new data and run inference.

    This function performs the following steps:
    1. Loads existing data from 'dashboard/data/data.csv'.
    2. Loads new data using the _load_new_data() function.
    3. Concatenates existing and new data.
    4. Saves the updated data to 'dashboard/modules/data.csv' and 'models/data/data.csv'.
    5. Calls the inference script using _call_inference_script().

    The function doesn't return anything but updates CSV files and triggers inference.

    Raises:
        Any exceptions raised by the underlying functions (e.g., file I/O errors)
        will propagate up from this function.
    """
    logging.info("Updating data...")
    existing_data = pd.read_csv("dashboard/data/data.csv")
    logging.info("Loading new data...")
    new_data = _load_new_data(existing_data)
    logging.info("Concatenating data...")
    # Concatenate existing and new data
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Remove duplicate rows based on all columns except TRIP_ID
    columns_for_deduplication = [col for col in updated_data.columns if col != 'TRIP_ID']
    updated_data = updated_data.drop_duplicates(subset=columns_for_deduplication, keep='first')
    
    # Reset the index after removing duplicates
    updated_data = updated_data.reset_index(drop=True)
    logging.info("Saving updated data...")
    updated_data.to_csv("dashboard/data/data.csv", index=False)
    updated_data.to_csv("dashboard/modules/data.csv", index=False)
    updated_data.to_csv("models/data/data.csv", index=False) # for inference
    logging.info("Calling inference script...")
    _call_inference_script()
    pathlib.Path("dashboard/modules/data.csv").unlink()
    
def _call_inference_script():
    """
    Call the LSTM inference script and handle its output.

    This function executes the LSTM inference script using subprocess.run().
    It captures both stdout and stderr, and prints the appropriate output
    based on the script's execution status.

    The inference script is expected to be located at '/models/main.py'
    and is run with the arguments '--model lstm' and '--mode infer'.
    
    Will update the error.csv file in '/dashboard/data'

    Returns:
        None

    Raises:
        No exceptions are explicitly raised, but errors from the subprocess
        execution are printed to stdout.
    """
    command = ["python", "models/main.py", "--model", "lstm" ,"--mode", "infer"]

    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output: {result.stdout}")

def _load_new_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Load newly created data based on the input DataFrame.

    This function performs the following steps:
    1. Calls _create_new_data() to generate new data based on the input.
    2. Reads the newly created data from 'modules/data.csv'.
    3. Removes the temporary CSV file.
    4. Returns the new data as a DataFrame.

    Args:
        data (pd.DataFrame): The existing data used to create new data.

    Returns:
        pd.DataFrame: A DataFrame containing the newly created and loaded data.

    Note:
        This function assumes that _create_new_data() generates a CSV file
        at 'modules/data.csv', which is then read and deleted.
    """
    _create_new_data(data)
    new_data = pd.read_csv("dashboard/modules/data.csv")
    # TODO: Implement file removal logic
    try:
        pathlib.Path("dashboard/modules/data.csv").unlink()
        logging.info("Temporary data file removed successfully.")
    except FileNotFoundError:
        logging.warning("Temporary data file not found. It may have been already removed.")
    except PermissionError:
        logging.error("Permission denied when trying to remove temporary data file.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while removing temporary data file: {str(e)}")
    return new_data

def _get_querying_dates(data: pd.DataFrame) -> Tuple[str, str]:
    """
    Determine the date range for querying new data.

    This function finds the most recent date in the provided DataFrame
    and returns it along with today's date as strings.

    Args:
        data (pd.DataFrame): A DataFrame containing a 'date' column.

    Returns:
        Tuple[str, str]: A tuple containing two date strings:
            - last_date_in_df: The most recent date in the DataFrame (format: 'YYYY-MM-DD').
            - todays_date: Today's date (format: 'YYYY-MM-DD').
    """
    data["date"] = pd.to_datetime(data["date"], format='mixed')
    last_date_in_df = data["date"].max().strftime('%Y-%m-%d')
    todays_date = datetime.now().date().strftime('%Y-%m-%d')
    return last_date_in_df, todays_date

def _call_pipeline_script(start_date, end_date, save_path):
    """
    Call the pipeline script with specified parameters.

    This function executes the pipeline script located at '.././pipeline/main.py'
    with the given start date, end date, and save path. It captures and prints
    the output or error messages from the script execution.

    Args:
        start_date (str): The start date for the data query (format: 'YYYY-MM-DD').
        end_date (str): The end date for the data query (format: 'YYYY-MM-DD').
        save_path (str): The path where the processed data will be saved.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the subprocess call fails.

    Note:
        The function prints the output or error messages to the console.
    """
    command = [
        "python", "pipeline/main.py", 
        start_date, end_date, save_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output: {result.stdout}")

def _create_new_data(data: pd.DataFrame):
    """
    Create new data by calling the pipeline script with updated date range.

    This function determines the date range for new data creation based on the
    provided DataFrame and calls the pipeline script to generate new data.

    Args:
        data (pd.DataFrame): A DataFrame containing existing data with a 'date' column.

    Returns:
        None

    Note:
        This function uses _get_querying_dates to determine the date range and
        _call_pipeline_script to execute the data creation process. The new data
        will be saved in the 'dashboard/modules' directory.
    """
    last_date_in_df, todays_date = _get_querying_dates(data)
    _call_pipeline_script(last_date_in_df, todays_date, "dashboard/modules")
    

if __name__ == "__main__":
    update_data()
# ------------------------------------------------------------------------------
# This script processes raw data from various CSV files,
# parsing and minimally cleaning the data for further analysis.
# ------------------------------------------------------------------------------

# Libraries ----------
import os
import pandas as pd
from datetime import datetime
import argparse
#directory = "Axel\GPSdata\GPStrips"
# Variables ----------
# Define column names for the CSV files
column_names = [
    "time",
    "node_id",
    "node_name",
    "signal_name_alias",
    "value",
    "signal_instance",
]
# Define data types for each column
col_types = {
    "node_id": "object",
    "node_name": "object",
    "signal_name_alias": "object",
    "value": "float64",
    "signal_instance": "object",
}
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process raw data from CSV files.")
parser.add_argument("directory", type=str, help="Directory where raw data is stored")
args = parser.parse_args()
directory = args.directory

# List all boat directories within the trips directory
boats = [
    os.path.join(directory, d)
    for d in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, d))
]
# List all subdirectories within each boat directory
boat_dirs = [
    os.path.join(boat, d)
    for boat in boats
    for d in os.listdir(boat)
    if os.path.isdir(os.path.join(boat, d))
]

def is_geo_data(df):
    col = df["signal_instance"]
    return col.isna().all()

# Function to load a CSV file into a DataFrame
def load_csv(csv_file):

    if not os.path.exists(csv_file):
        print(f"File does not exist: {csv_file}")
        return None

    try:
        # Load the CSV file into a DataFrame with specified column names, data types, and parsing 'time' as datetime
        df = pd.read_csv(
            csv_file,
            names=column_names,
            dtype=col_types,
            parse_dates=["time"],
            skiprows=1,
        )
        # No rounding applied to 'value' column
        return df

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None
    
# Process data algorithm ----------
print("---------- Processing Data ----------")
geo_df_list = []
# List all CSV files within the main directory
csv_files = [
    os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
]

for csv_file in csv_files:
    print(f"Handling {csv_file}")
    some_df = load_csv(csv_file)

    if some_df is not None and not some_df.empty:
        # Check if the DataFrame contains geo data
        if is_geo_data(some_df):
            geo_df_list.append(some_df)
    else:
        print(f"{csv_file} is empty or None")

# Debugging print statements
print(f"geo_df_list length: {len(geo_df_list)}")

# Concatenate all DataFrames
if geo_df_list:
    geo_df = pd.concat(geo_df_list, ignore_index=True)
    # Write to file ----------
    print("---------- Writing to Files ----------")
    geo_df.to_csv(f"{directory}/geo_data.csv", index=False)
else:
    print("No geo data to concatenate.")

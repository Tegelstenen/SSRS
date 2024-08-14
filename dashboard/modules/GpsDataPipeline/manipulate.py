# ------------------------------------------------------------------------------
# This script loads, cleans, and processes geo and engine data by correcting 
# column names, calculating mean values at 1-second intervals, imputing missing 
# values, and saving the manipulated data back to CSV files.
# ------------------------------------------------------------------------------
 
import pandas as pd
import numpy as np

import argparse
from datetime import datetime, timedelta
import os


# Helper functions ----------
def regex_fix(df):
    # Remove prefix before underscore in 'signal_name_alias'
    df["signal_name_alias"] = df["signal_name_alias"].str.replace(
        r"^[^_]*_", "", regex=True
    )
    # Remove prefix before underscore in 'signal_name_alias' again
    df["signal_name_alias"] = df["signal_name_alias"].str.replace(
        r"^[^_]*_", "", regex=True
    )
    return df

def get_mean_values(df):
    # Convert 'time' to date and round 'time' to 10-second intervals
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df["time"] = pd.to_datetime(df["time"]).dt.round("1s").dt.strftime("%H:%M:%S")
    df = (
        df.groupby(
            ["node_name", "signal_name_alias", "date", "time"]
        )
        .agg({"value": "mean"})
        .reset_index()
    )

    # Pivot the table to wide format
    df = df.pivot_table(
        index=["node_name", "date", "time"],
        columns="signal_name_alias",
        values="value",
    ).reset_index()

    # Add a column 'tripID' to identify trips based on time difference
    df['tripID'] = 1  # Initialize tripID with 1
    time_diffs = df['time'].apply(lambda x: timedelta(hours=int(x.split(':')[0]), minutes=int(x.split(':')[1]), seconds=int(x.split(':')[2]))).diff().abs().dt.total_seconds().fillna(0)
    trip_starts = (time_diffs > 300).cumsum()  # 300 seconds = 5 minutes
    df['tripID'] += trip_starts

    return df

def eng_imputation(df):
    # Impute NA's by forward filling and then backward filling
    df = df.groupby("date").apply(lambda group: group.ffill().bfill())
    df = df.reset_index(drop=True)
    return df

def geo_imputation(df):
    unique_trip_ids = df['tripID'].unique()
    for trip_id in unique_trip_ids:
        try:
            trip_df = df[df['tripID'] == trip_id]
            # Apply backward fill followed by forward fill within the same trip_id
            trip_df = trip_df.bfill().ffill()
            # Update the main dataframe with the imputed trip data
            trip_df = trip_df.fillna(0)
            df.update(trip_df)
        except Exception as e:
            try:
                trip_df = trip_df.fillna(0)
                df.update(trip_df)
            except Exception as e:
                print(f"Error during imputation for tripID {trip_id}: {e}")
    return df

def normalize_rpm(df):
    df['RPM'] = df['RPM'].apply(lambda x: 1 if x > 0 else 0)
    return df

def add_deltas(df):
    # Ensure the dataframe is sorted by trip_id and time
    # Initialize the delta columns with NaN values
    df['LAT_Delta'] = np.nan
    df['LON_Delta'] = np.nan
    df['SOG_Delta'] = np.nan
    
    # Iterate over each unique trip_id
    for trip_id in df['tripID'].unique():
        trip_df = df[df['tripID'] == trip_id]
        # Calculate the deltas for LAT, LON, and SOG
        df.loc[trip_df.index, 'LAT_Delta'] = trip_df['LAT'].diff().fillna(0)
        df.loc[trip_df.index, 'LON_Delta'] = trip_df['LON'].diff().fillna(0)
        df.loc[trip_df.index, 'SOG_Delta'] = trip_df['SOG'].diff().fillna(0)
    
    return df

# Manipulation algorithm ----------
parser = argparse.ArgumentParser(description="Process raw data from CSV files.")
parser.add_argument("directory", type=str, help="Directory where raw data is stored")
args = parser.parse_args()
directory = args.directory

geo_path = os.path.join(directory, "geo_data.csv")

print("---------- Loading Data ----------")
geo = pd.read_csv(geo_path)

print("---------- Correcting Column Names ----------")
geo = regex_fix(geo)

print("---------- Cleaning Format ----------")
#geo["date"] = pd.to_datetime(geo["time"]).dt.date
geo = get_mean_values(geo)
geo = geo_imputation(geo)
geo = add_deltas(geo)
geo = normalize_rpm(geo)

# Ensure the dataframe is sorted by node_name, date, and time
geo = geo.sort_values(by=['node_name', 'date', 'time'])

# Save manipulated data ----------
geo.to_csv(geo_path, index=False)
print("---------- Finished ----------")
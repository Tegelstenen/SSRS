import pandas as pd

import logging
import os
from pathlib import Path

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging

config = ConfigManager()

class DataImputer:
    def __init__(self):
        setup_logging()
        self.directory = config.get("TEMP_DATA_DIR")
        self.run(self.directory)

    def impute_geo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in geo data."""
        logging.pipeline("Imputing geo data")
        df["SOG"] = df["SOG"].fillna(0)
        df = self.identify_trips(df)
        df = self.ffill_all(df)
        return df

    def impute_engine_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in engine data."""
        logging.pipeline("Imputing engine data")
        try:
            df = self.impute_events(df)
            df = self.identify_trips(df)
            df = self.ffill_all(df)
        except Exception as e:
            logging.error(f"An error occurred during engine data imputation: {str(e)}")
        return df

    def impute_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in event data."""
        logging.pipeline("Imputing event data")
        try:
            EVENT_MEASUREMENTS = config.get("EVENT_MEASUREMENTS")
            for event in EVENT_MEASUREMENTS:
                if event in df.columns:
                    df[event] = df.get(event, 0)
        except Exception as e:
            logging.error(f"An error occurred during event data imputation: {str(e)}")
        return df

    def ffill_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill all columns."""
        logging.pipeline("Forward filling all columns")
        for column in df.columns:
            df[column] = df.groupby(["node_name", "TRIP_ID", "signal_instance"])[column].ffill()
        return df

    def impute_eng_temp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in engine temperature data."""
        logging.pipeline("Imputing engine temperature data")
        try:
            df["ENGTEMP"] = (
                df.groupby(["node_name", "TRIP_ID", "signal_instance"])["ENGTEMP"]
                .bfill()
                .ffill()
            )
        except Exception as e:
            logging.error(f"An error occurred during engine temperature data imputation: {str(e)}")
        return df
    
    def impute_rpm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in RPM data."""
        logging.pipeline("Imputing RPM data")
        try:
            df["RPM"] = (
                df.groupby(["node_name", "TRIP_ID", "signal_instance"])["RPM"]
                .ffill()
                .bfill()
            )
        except Exception as e:
            logging.error(f"An error occurred during engine temperature data imputation: {str(e)}")
        return df

    def identify_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify trips in the data for each node."""
        try:
            logging.pipeline("Identifying trips")
            df = df.sort_values(["node_name", "date", "time"])
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

            def process_group(group):
                group["time_diff"] = group["datetime"].diff()
                new_day = group["date"] != group["date"].shift()
                long_gap = group["time_diff"] >= pd.Timedelta(minutes=30)
                new_trip = new_day | long_gap
                group["TRIP_ID"] = new_trip.cumsum()
                return group

            # Use groupby().apply() without reset_index
            df = df.groupby("node_name", group_keys=False).apply(process_group)
            df = df.drop(["datetime", "time_diff"], axis=1)

            return df
        except Exception as e:
            logging.error(f"An error occurred while identifying trips: {e}")
            return df

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from a CSV file."""
        logging.pipeline(f"Loading data from {file_path}")
        return pd.read_csv(file_path)

    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame to a CSV file."""
        df.to_csv(file_path, index=False)
        logging.pipeline(f"Saved imputed data to {file_path}")

    def run(self, directory: Path) -> None:
        """Main function to impute and save geo and engine data."""
        logging.info(f"Imputing data")
        geo_path = os.path.join(directory, "geo_data.csv")
        engine_path = os.path.join(directory, "engine_data.csv")

        geo_df = self.load_data(geo_path)
        engine_df = self.load_data(engine_path)

        imputed_geo_df = self.impute_geo_data(geo_df)
        imputed_engine_df = self.impute_engine_data(engine_df)

        self.save_data(imputed_geo_df, geo_path)
        self.save_data(imputed_engine_df, engine_path)


import pandas as pd
import certifi
from dotenv import load_dotenv
from influxdb_client_3 import InfluxDBClient3, flight_client_options

import concurrent.futures
from datetime import timedelta
import os
from typing import List, Tuple
import logging

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging
from utils.helper import HelperFunctions


class InfluxDBClient:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self) -> InfluxDBClient3:
        load_dotenv()
        api_token = os.getenv("TOKEN")
        if not api_token:
            raise ValueError("InfluxDB API token not found in environment variables")

        with open(certifi.where(), "r") as fh:
            cert = fh.read()

        return InfluxDBClient3(
            host=self.config.get("INFLUXDB_HOST"),
            token=api_token,
            org=self.config.get("INFLUXDB_ORG"),
            database=self.config.get("INFLUXDB_DATABASE"),
            flight_client_options=flight_client_options(tls_root_certs=cert),
        )

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self.client.query(sql).to_pandas()
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise

    def is_connected(self, list_tables: bool = False) -> bool:
        try:
            tables = self.client.query("SHOW TABLES")
            logging.pipeline("Successfully connected to InfluxDB")
            if list_tables:
                db_df = tables.to_pandas()
                logging.info("Available tables:\n%s", db_df.to_string())
            return True
        except Exception as e:
            logging.error("Error connecting to InfluxDB: %s", e, exc_info=True)
            return False


class DataBaseQuery:
    def __init__(self, start: str, stop: str):
        self.start = start
        self.stop = stop
        self.config = ConfigManager()
        setup_logging()
        self.db_client = InfluxDBClient(self.config)
        self.iterator = 0
        self.run()

    def run(self):
        logging.info(f"Starting data query for period: {self.start} to {self.stop}")

        self.db_client.is_connected(list_tables=False)

        train_data_folder, query_start, query_stop, boats = self.get_arguments()

        logging.pipeline(f"Processing {len(boats)} boats")

        # Split the date range into 10-day intervals
        start_date = pd.to_datetime(query_start)
        end_date = pd.to_datetime(query_stop)
        date_ranges = pd.date_range(start=start_date, end=end_date, freq="13D").tolist()
        if date_ranges[-1] != end_date:
            date_ranges.append(end_date)

        start_stop_times_list = []

        for i in range(len(date_ranges) - 1):
            interval_start = date_ranges[i].strftime("%Y-%m-%dT%H:%M:%SZ")
            interval_stop = date_ranges[i + 1].strftime("%Y-%m-%dT%H:%M:%SZ")
            logging.pipeline(
                f"Fetching querying intervals for: {interval_start} to {interval_stop}"
            )
            start_stop_times = self.all_start_stop_times(
                interval_start, interval_stop, boats
            )
            start_stop_times_list.append(start_stop_times)

        for start_stop_times in start_stop_times_list:
            start = start_stop_times["start"].min()
            stop = start_stop_times["stop"].max()
            logging.pipeline(f"Fetching all data for interval: {start} to {stop}")
            self.query_sql(start_stop_times, train_data_folder)
            self.iterator += 1000
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        non_zero_mask = df["value"] > 0
        df["last_non_zero_time"] = df.loc[non_zero_mask, "time"].ffill()
        df["next_non_zero_time"] = df.loc[non_zero_mask, "time"].bfill()
        time_diff_forward = df["time"] - df["last_non_zero_time"]
        time_diff_backward = df["next_non_zero_time"] - df["time"]
        keep_mask = (time_diff_forward <= pd.Timedelta(minutes=10)) | (
            time_diff_backward <= pd.Timedelta(minutes=10)
        )

        return df[keep_mask].drop(["last_non_zero_time", "next_non_zero_time"], axis=1)

    def extract_trip_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["trip_id", "time"])
        first_rows = df.groupby("trip_id").first()
        last_rows = df.groupby("trip_id").last()
        result = pd.concat([first_rows, last_rows]).sort_values(["trip_id", "time"])
        return result.reset_index()

    def generate_trip_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("time")
        df["trip_id"] = 1
        current_trip_id = 1
        previous_time = df["time"].iloc[0]

        for index, row in df.iterrows():
            current_time = row["time"]
            if current_time - previous_time > timedelta(minutes=20):
                current_trip_id += 1
            df.at[index, "trip_id"] = current_trip_id
            previous_time = current_time

        return df

    def pivot_trip_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["trip_id", "time"])
        df["start_stop"] = df.groupby("trip_id").cumcount().map({0: "start", 1: "stop"})
        df = df.pivot(
            index=["trip_id", "node_name"], columns="start_stop", values="time"
        )
        df["start"] = df["start"] - pd.Timedelta(minutes=10)
        df["stop"] = df["stop"] + pd.Timedelta(minutes=10)

        return df.reset_index()

    def get_start_stop_times(
        self, query_start: str, query_stop: str, boat: str
    ) -> pd.DataFrame:
        sql = f"""
            SELECT time, node_id, node_name, signal_name_alias, value, signal_instance
            FROM "RPM"
            WHERE time >= '{query_start}' AND time <= '{query_stop}'
                AND (node_name = '{boat}')
                AND (value > 0)
            ORDER BY time ASC
        """

        try:
            df = self.db_client.query(sql)

            if df.empty:
                logging.pipeline(f"No RPM data for boat: {boat}")
                return None

            df = self.generate_trip_id(df)
            df = self.extract_trip_endpoints(df)

            if df.shape[0] > 1:
                df = self.pivot_trip_data(df)
            else:
                df["start"] = df["time"]
                df["stop"] = df["time"]
                df = df[["trip_id", "start", "stop", "node_name"]]

            return df

        except Exception as e:
            logging.error(f"Error querying RPM for {boat}: {e}")
            logging.error("Exception details:", exc_info=True)
            return None

    def get_arguments(self) -> tuple:
        train_data_folder = HelperFunctions.get_data_folder()
        os.makedirs(train_data_folder, exist_ok=True)

        query_start = pd.to_datetime(self.start).strftime("%Y-%m-%dT%H:%M:%SZ")
        query_stop = pd.to_datetime(self.stop).strftime("%Y-%m-%dT%H:%M:%SZ")
        boats = self.config.get("BOATS")
        return train_data_folder, query_start, query_stop, boats

    def all_start_stop_times(
        self, query_start: str, query_stop: str, boats: List[str]
    ) -> pd.DataFrame:
        start_stop_times = pd.DataFrame()

        def process_boat(boat):
            return self.get_start_stop_times(query_start, query_stop, boat)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_boat = {
                executor.submit(process_boat, boat): boat for boat in boats
            }
            for future in concurrent.futures.as_completed(future_to_boat):
                boat = future_to_boat[future]
                try:
                    boat_data = future.result()
                    if boat_data is not None:
                        start_stop_times = pd.concat(
                            [start_stop_times, boat_data], ignore_index=True
                        )
                except Exception as e:
                    logging.error(f"Error processing boat {boat}: {e}")

        return start_stop_times

    def write_csv(self, output_data: List[Tuple[str, str, pd.DataFrame]]):
        def write_single_csv(node_folder, output_csv_path, group_df):
            try:
                os.makedirs(node_folder, exist_ok=True)
                group_df.to_csv(output_csv_path, index=False)
            except Exception as e:
                logging.error(f"Error writing CSV for {output_csv_path}: {e}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    write_single_csv, node_folder, output_csv_path, group_df
                )
                for node_folder, output_csv_path, group_df in output_data
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # We can handle exceptions here if needed
                except Exception as e:
                    logging.error(f"Error during CSV writing: {e}")

    def query_sql(self, start_stop_times: pd.DataFrame, train_data_folder: str):
        all_measurements = self.config.get("TO_QUERY")

        # Group all trips for all boats together
        all_trips = start_stop_times.groupby("node_name")

        def process_measurement(measurement):
            self.process_all_boats(measurement, all_trips, train_data_folder)

        # Process each measurement in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_measurement = {
                executor.submit(process_measurement, measurement): measurement
                for measurement in all_measurements
            }
            for future in concurrent.futures.as_completed(future_to_measurement):
                measurement = future_to_measurement[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Measurement {measurement} generated an exception: {e}")

    def process_all_boats(
        self,
        measurement: str,
        all_trips: pd.core.groupby.DataFrameGroupBy,
        train_data_folder: str,
    ):
        all_boat_conditions = []
        for boat, boat_trips in all_trips:
            boat_conditions = " OR ".join(
                [
                    f"(time >= '{row['start']}' AND time <= '{row['stop']}' AND node_name = '{boat}')"
                    for _, row in boat_trips.iterrows()
                ]
            )
            all_boat_conditions.append(f"({boat_conditions})")
        combined_conditions = " OR ".join(all_boat_conditions)

        sql = f"""
                SELECT time, node_id, node_name, signal_name_alias, value, signal_instance
                FROM "{measurement}"
                WHERE {combined_conditions}
                ORDER BY time ASC
            """

        try:
            df = self.db_client.query(sql)

            output_data = []
            for boat, boat_trips in all_trips:
                for _, row in boat_trips.iterrows():
                    trip_df = df[
                        (df["time"] >= row["start"])
                        & (df["time"] <= row["stop"])
                        & (df["node_name"] == boat)
                    ]
                    for node_id, group_df in trip_df.groupby("node_id"):
                        node_folder = os.path.join(
                            train_data_folder,
                            node_id,
                            f"trip_{row['trip_id']+self.iterator}",
                        )
                        output_csv_path = os.path.join(
                            node_folder, f"{measurement}.csv"
                        )
                        output_data.append((node_folder, output_csv_path, group_df))

            self.write_csv(output_data)

        except Exception as e:
            logging.error(f"Error querying {measurement}: {e}")
            logging.error("Exception details:", exc_info=True)

        logging.pipeline(f"Completed processing for {measurement}")

from dotenv import load_dotenv
from influxdb_client_3 import InfluxDBClient3, flight_client_options
import pandas as pd

import os
import certifi
import logging

from utils.config_manager import ConfigManager

config = ConfigManager()

class InfluxDBClient:
    def __init__(self):
        self.client = self._initialize_client()

    def _initialize_client(self) -> InfluxDBClient3:
        load_dotenv()
        api_token = os.getenv("TOKEN")
        if not api_token:
            raise ValueError("InfluxDB API token not found in environment variables")

        with open(certifi.where(), "r") as fh:
            cert = fh.read()

        return InfluxDBClient3(
            host=config.get("INFLUXDB_HOST"),
            token=api_token,
            org=config.get("INFLUXDB_ORG"),
            database=config.get("INFLUXDB_DATABASE"),
            flight_client_options=flight_client_options(tls_root_certs=cert),
        )

    def query(self, boat: str, date: str, table: str) -> pd.DataFrame:
        sql = self._create_query(boat, date, table)
        logging.debug(f"Executing SQL query: {sql}")
        try:
            return self.client.query(sql).to_pandas()
        except Exception as e:
            logging.error(f"Query failed: {e}")
            raise ValueError(f"Failed to execute query: {e}")


    def is_connected(self, list_tables: bool = False) -> bool:
        try:
            tables = self.client.query("SHOW TABLES")
            if list_tables:
                db_df = tables.to_pandas()
                return db_df
            return True
        except Exception as e:
            return False

    def _create_query(self, boat: str, date: str, table: str) -> str:
        start_time = date + "T00:00:00Z"
        stop_time = date + "T23:59:59Z"
        query = f'''
                SELECT time, node_name, value, signal_instance
                FROM "{table}"
                WHERE time >= '{start_time}' AND time <= '{stop_time}'
                    AND (node_name = '{boat}')
                    AND (value > 0)
                ORDER BY time ASC
                '''
        return query
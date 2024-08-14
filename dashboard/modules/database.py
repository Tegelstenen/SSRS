import pandas as pd
import numpy as np
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import awswrangler as wr
import boto3

from decimal import Decimal 
import logging
import concurrent.futures
import time
import random

from utils.config_manager import ConfigManager

config = ConfigManager()

# Configure logging
log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) 
root_logger.addHandler(console_handler)
 

class Database:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.resource = boto3.resource('dynamodb', region_name='eu-north-1')
        self.client = boto3.client('dynamodb', region_name='eu-north-1')
        self._create_table()

    def get_data(self, node_name: str, date=None) -> pd.DataFrame:
        """
        Makes a query to the DynamoDB table and returns the data for the given node_name (a.k.a. boat) and date.
        """
        if date:
            result = wr.dynamodb.read_items(table_name=self.table_name, key_condition_expression=(Key("node_name").eq(node_name)), filter_expression=(Attr("date").eq(date)))
        else:
            result = wr.dynamodb.read_items(table_name=self.table_name, key_condition_expression=(Key("node_name").eq(node_name)))
        feature_columns = config.get("MODEL_FEATURES")
        geo_columns = config.get("GEO_FEATURES")
        result = self._object_to_float(result, feature_columns)
        result = self._object_to_float(result, geo_columns)
        result = result.replace(-99999, pd.NA)
        return result

    def _object_to_float(self, df:pd.DataFrame, feature_columns:list) -> pd.DataFrame:
        for col in feature_columns:
            if col != "signal_instance":
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(float)
        return df

        
    def write_to_table(self, df: pd.DataFrame) -> None:
        """
        Writes the dataframe to the DynamoDB table using parallel processing.
        """
        items = self._prepare_data(df)
        table = self.resource.Table(self.table_name)
        root_logger.info(f"Writing {len(items)} items to the table with batch")
        self._parallel_processing(items, table)

    def _parallel_processing(self, items: list, table) -> None:
        """
        Internal method for parallel processing of the items.
        """
        chunks = self._get_chunks(items)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = self._get_futures(table, chunks, executor)
            self._iterate_futures(futures)

    def _iterate_futures(self, futures):
        """
        Iterating over the futures.
        """
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                root_logger.error(f"Error writing chunk: {e}")

    def _get_futures(self, table, chunks, executor):
        """
        Returns the futures used by the executor.
        """
        futures = [executor.submit(self._write_chunk, chunk, table) for chunk in chunks]
        return futures

    def _get_chunks(self, items):
        """
        Returns the chunks used to determine futures.
        """
        chunks = [items[i:i + 25] for i in range(0, len(items), 25)]
        return chunks
    
    def _write_chunk(self, chunk, table):
        """
        Writes the chunk to the table.
        """
        retries = 0
        max_retries = 9
        while retries < max_retries:
            try:
                with table.batch_writer() as batch:
                    for item in chunk:
                        batch.put_item(Item=item)
                return
            except ClientError as e:
                self._retries_error_handeling(retries, e)
            except Exception as e:
                root_logger.error(f"Unexpected error: {e}")
                raise

    def _retries_error_handeling(self, retries, e):
        error_code = e.response['Error']['Code']
        if error_code == 'ProvisionedThroughputExceededException':
            retries += 1
            sleep_time = (2 ** retries) + random.uniform(0, 1)
            root_logger.warning(f"Provisioned throughput exceeded, retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            root_logger.error(f"Failed to write chunk: {e}")
            raise

    def _add_sort_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a sort key to the dataframe.
        """
        df['sort_key'] = df['date'] + '#' + df['time'] + '#' + df['signal_instance']
        return df

    def _float_to_decimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts float columns to decimal.
        """
        float_columns = df.select_dtypes(include=['float64']).columns
        for col in float_columns:
            df[col] = df[col].apply(lambda x: Decimal(str(x)))
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataframe by replacing NaN and infinite values.
        """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(-99999, inplace=True)  # Replace NaN with 0 or another appropriate value
        return df

    def _create_table(self):
        """
        Creates the DynamoDB table if it does not exist.
        """
        client = boto3.client('dynamodb', region_name='eu-north-1')
        
        if self._table_exist(self.table_name):
            root_logger.info(f"Table {self.table_name} already exists")
            pass
        else:
            try:
                root_logger.info(f"Creating table {self.table_name}") 
                table = client.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "node_name", "KeyType": "HASH"},  # Partition key
                    {"AttributeName": "sort_key", "KeyType": "RANGE"}  # Sort key
                ],
                AttributeDefinitions=[
                    {"AttributeName": "node_name", "AttributeType": "S"},
                    {"AttributeName": "sort_key", "AttributeType": "S"}
                ],
                ProvisionedThroughput={
                    "ReadCapacityUnits": 10,  # Increase these values
                    "WriteCapacityUnits": 10,  # Increase these values
                },
                )
                table.wait_until_exists()
            except ClientError as err:
                root_logger.error(
                    "Couldn't create table %s. Here's why: %s: %s",
                self.table_name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
        
    def _table_exist(self, table_name: str) -> bool:
        """
        Checks if the table already exists.
        """
        if table_name in self.client.list_tables()['TableNames']:
            return True
        else:
            return False
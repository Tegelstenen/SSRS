import pymysql
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

class Database:
    def __init__(self):
        timeout = 10
        self.connection = pymysql.connect(
            charset="utf8mb4",
            connect_timeout=timeout,
            cursorclass=pymysql.cursors.DictCursor,
            db="defaultdb",
            host="ssrs-sensacareers.i.aivencloud.com",
            password=os.getenv("AIVEN_PASSWORD"),
            read_timeout=timeout,
            port=28183,
            user="avnadmin",
            write_timeout=timeout,
        )

    def make_table(self, table_name: str):
        with self.connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    node_name VARCHAR(255),
                    date DATETIME,
                    RPM FLOAT,
                    FUELRATE FLOAT,
                    ENGINE_LOAD FLOAT,
                    ENGTEMP FLOAT,
                    ENGOILPRES FLOAT,
                    PRIMARY KEY (node_name, date)
                )
            """)
            self.connection.commit()
            print(f"Table {table_name} created or already exists.")

    def fetch_all(self, table_name: str):
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name}")
            return cursor.fetchall()

    def add_data(self, table_name: str, data: pd.DataFrame):
        with self.connection.cursor() as cursor:
            for _, row in data.iterrows():
                cursor.execute(f"""
                    INSERT INTO {table_name} (node_name, date, RPM, FUELRATE, ENGINE_LOAD, ENGTEMP, ENGOILPRES)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, tuple(row))
            self.connection.commit()
            print(f"Data inserted into {table_name}.")

    def close(self):
        self.connection.close()

import openmeteo_requests
import requests
import pandas as pd
from retry_requests import retry
import numpy as np

from datetime import timedelta 
import os
import logging

from utils.config_manager import ConfigManager
from utils.log_setup import setup_logging

config = ConfigManager()

class WeatherData:    
    def __init__(self):
        pass
    
    def fetch_weather_historic(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        logging.pipeline("Fetching weather data")
        # Setup the Open-Meteo API client with retry on error
        session = requests.Session()
        retry_session = retry(session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Define the API URL and parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["wind_speed_10m", "wind_direction_10m"],
            "wind_speed_unit": "ms",
            "timezone": "Europe/Berlin"
        }

        # Make the API request
        responses = openmeteo.weather_api(url, params=params)

        # Process the response for the first location
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

        return pd.DataFrame(data=hourly_data)

    def assign_nearest_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        logging.pipeline("Assigning nearest weather data")
        # Ensure both 'datetime' and 'date' columns are in datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        # Create a column to store the nearest weather time
        df['nearest_weather_time'] = pd.NaT

        # Define a time window range (e.g., 30 minutes before and after the weather timestamp)
        time_window = timedelta(minutes=30)

        for index, weather_row in weather_df.iterrows():
            weather_time = weather_row['date']
            # Find boat data within the time window
            mask = (df['datetime'] >= weather_time - time_window) & (df['datetime'] <= weather_time + time_window)
            df.loc[mask, 'nearest_weather_time'] = weather_time

        return df

    def calculate_alignment_factor(self, boat_heading: float, wind_direction: float) -> float:
        """
        Calculate the alignment factor between the boat's heading (COG) and the wind direction.
        The alignment factor ranges from -1 (opposite direction) to 1 (same direction).
        """
        # Normalize the angles to be within 0-360 degrees
        boat_heading = boat_heading % 360
        wind_direction = wind_direction % 360

        # Calculate the difference between the wind direction and the boat's heading
        angle_difference = wind_direction - boat_heading

        # Normalize the difference to be within -180 to 180 degrees
        if angle_difference > 180:
            angle_difference -= 360
        elif angle_difference < -180:
            angle_difference += 360

        # Scale the difference to range from -1 to 1
        alignment_factor = angle_difference / 180

        return alignment_factor


class WeatherProcessor:
    def __init__(self):
        setup_logging()
        self.directory = config.get("TEMP_DATA_DIR")
        self.weather_data = WeatherData()
        self.run()

    def process_file(self, csv_file: str) -> pd.DataFrame:
        logging.pipeline("Processing data")
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Combine 'date' and 'time' columns to create a datetime column
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        # Group by 'node_name' and calculate the first and last date for each group
        result_times = df.groupby('node_name')['datetime'].agg(['min', 'max']).reset_index()

        # Rename columns for clarity
        result_times.columns = ['node_name', 'first_timestamp', 'last_timestamp']

        # Split the first and last timestamps into separate date and time columns
        result_times['first_date'] = result_times['first_timestamp'].dt.date
        result_times['last_date'] = result_times['last_timestamp'].dt.date

        # Drop the original timestamp columns if not needed
        result_times = result_times.drop(columns=['first_timestamp', 'last_timestamp'])


        try:
            mean_coords = df.groupby('node_name')[['LAT', 'LON']].mean().reset_index()
        except KeyError:
            logging.error("No LAT or LON columns found in the DataFrame.")
            logging.error("Returning with NaN values")
            df['wind_velocity'] = np.nan
            df['wind_angle'] = np.nan
            df['alignment_factor'] = np.nan
            df['LON'] = np.nan
            df['LAT'] = np.nan
            df.drop(columns=['datetime'], inplace=True)
            return df

        # Merge the mean coordinates with the result DataFrame
        result_times = pd.merge(result_times, mean_coords, on='node_name')

        weather_data_frames = []
        for index, row in result_times.iterrows():
            boat_id = row['node_name']
            first_date = row['first_date']
            last_date = row['last_date']
            lat = row['LAT']
            lon = row['LON']

            weather_df = self.weather_data.fetch_weather_historic(lat, lon, first_date, last_date)
            weather_df['node_name'] = boat_id
            weather_data_frames.append(weather_df)

        full_weather_df = pd.concat(weather_data_frames, ignore_index=True)
        full_weather_df['date'] = pd.to_datetime(full_weather_df['date']).dt.tz_localize(None)

        # Now we map each boat's data to the nearest weather timestamp
        df = self.weather_data.assign_nearest_weather(df, full_weather_df)

        df['nearest_weather_time'] = pd.to_datetime(df['nearest_weather_time'])
        full_weather_df['date'] = pd.to_datetime(full_weather_df['date'])

        # Merge the boat data with the weather data
        merged_df = pd.merge(df, full_weather_df, left_on=['nearest_weather_time', 'node_name'], right_on=['date', 'node_name'], how='left')

        # Drop the nearest_weather_time and date_y columns if not needed
        merged_df = merged_df.drop(columns=['nearest_weather_time', 'date_y'])

        # Directly use the wind-related data from the weather data
        merged_df['wind_velocity'] = merged_df['wind_speed_10m']
        merged_df['wind_angle'] = merged_df['wind_direction_10m']

        logging.pipeline("Calculating alignment factor")
        # Calculate the alignment factor
        merged_df['alignment_factor'] = merged_df.apply(
            lambda row: self.weather_data.calculate_alignment_factor(row['COG'], row['wind_angle']), axis=1
        )

        # Rename date_x to date
        merged_df.rename(columns={'date_x': 'date'}, inplace=True)

        # Drop columns no longer needed
        merged_df = merged_df.drop(columns=['wind_speed_10m', 'wind_direction_10m', 'datetime'])

        return merged_df

    def run(self):
        logging.info("Adding weather data")
        try:
            csv_file = os.path.join(self.directory, 'data.csv')
            merged_df = self.process_file(csv_file)
            merged_df.to_csv(csv_file, index=False)
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")

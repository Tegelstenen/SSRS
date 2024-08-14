#Imports
import pandas as pd
import numpy as np
import folium
import h3
import os
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import keras
from dotenv import load_dotenv
from geopy.distance import geodesic
import subprocess
import glob
import streamlit as st
from influxdb_client_3 import InfluxDBClient3, flight_client_options
import os
import certifi
import pandas as pd
import datetime
from dotenv import load_dotenv
from folium import IFrame
import time
#import yaml
import os
import re
from collections import defaultdict

##--------------------------------------------------------------##
# This file makes a DB call for all boats and puts it into a tmpfolder,
# data from this folder is manipulated and cleaned.
# It compares this data to a pretrained LSTM binary classifier and
# draws a map of any possible GPS interferance.
##--------------------------------------------------------------##

# TODO
# LIGHT/DARKMODE Switch in streamlit add the functionality
#
# Add feature so that the user can save "anomalies" (red/yellow) -->
# --> that were incorrecly labelled and save the data so the model -->
# --> can be retrained on more labelled data at a later stage.

##--------------------------------------------------------------##
# FIX THE DATABASE UTCNOW SHOULD BE CORRECT BUT LAT/LON DOESNT EXIST
##--------------------------------------------------------------##

##--------------------------------------------------------------##
# pre database call check
##--------------------------------------------------------------##

class GPSDataProcessor: 

    def __init__(self, last_hour_csv_path):
        ###### FIX THIS WHEN DB IS LIVE
        self.current_time = datetime.datetime.utcnow() - datetime.timedelta(days=25)
        ###### 
        self.csv_path = last_hour_csv_path
        self.time_comparison_result = self.compare_times(self.csv_path)

        if isinstance(self.time_comparison_result, tuple):
            self.is_time_delta_large, self.current_hour, self.previous_hour = self.time_comparison_result
        else:
            self.is_time_delta_large = self.time_comparison_result
            self.current_hour = None
            self.previous_hour = None

        #------------------------ DB
        # Load dotenv file
        load_dotenv()
        self.API_TOKEN = os.getenv('token')
        # Read certificate
        with open(certifi.where(), "r") as fh:
            self.cert = fh.read()

        # Initialize the InfluxDB client
        self.client = InfluxDBClient3(
            host="eu-central-1-1.aws.cloud2.influxdata.com",
            token=self.API_TOKEN,
            org="maritime",
            database="ship2shore-prod",
            flight_client_options=flight_client_options(
                tls_root_certs=self.cert))

        # (debug too see in terminal) Verify connection and list databases: DOes not work anymore (AUG-13 2024)
        # try:
        #     databases = self.client.query('SHOW DATABASES')
        #     print(databases.to_pandas().to_markdown())
        # except Exception as e:
        #     print(f"Error listing databases: {e}")
        #     exit(1)
        
        # Main folder to store trip data
        #self.main_folder = os.path.join('Filip', 'app', 'data', 'tmpfiles') ### Change to this later: 
        self.current_working_directory = os.getcwd()
        self.main_folder = os.path.join(self.current_working_directory, 'dashboard', 'data', 'tmpfiles')
        os.makedirs(self.main_folder, exist_ok=True)

        # Call database_call if time delta is large
        if self.is_time_delta_large:
           print("database call")
           self.database_call()
           print("database call done")
           #self.remove_geo_data_file()        
        #------------------------------ DB end
      
    def compare_times(self, csv_path):
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        current_time_rounded_down = self.current_time.replace(minute=0, second=0, microsecond=0)
        previous_time_rounded_down = current_time_rounded_down - datetime.timedelta(hours=1)
        
        max_csv_time = df['datetime'].max()
        time_delta = current_time_rounded_down - max_csv_time
        
        ship2shoreTimeFormat_current_hour = current_time_rounded_down.strftime("%Y-%m-%dT%H:%M:%SZ")
        ship2shoreTimeFormat_previous_hour = previous_time_rounded_down.strftime("%Y-%m-%dT%H:%M:%SZ")

        #print(f"Current Hour: {ship2shoreTimeFormat_current_hour}")  # Debug: Print current hour
        #print(f"Previous Hour: {ship2shoreTimeFormat_previous_hour}")  # Debug: Print previous hour

        # Check if the time delta is larger than 1 hour
        if time_delta >= datetime.timedelta(hours=1):
            return True, ship2shoreTimeFormat_current_hour, ship2shoreTimeFormat_previous_hour
        else:
            return False
        
    #just a debugger function can remove later    
    def use_comparison_results(self):
        if self.is_time_delta_large:
            print(f"Time delta is large: {self.is_time_delta_large}")
            print(f"Current Hour: {self.current_hour}")
            print(f"Previous Hour: {self.previous_hour}")
            # st.write(f"Time delta is large: {self.is_time_delta_large}")
            # st.write(f"Current Hour: {self.current_hour}")
            # st.write(f"Previous Hour: {self.previous_hour}")
            return True # added 11.06
        else:
            print(f"Time delta is not large: {self.is_time_delta_large}")
            return False # added 11.06
            #st.write(f"Time delta is not large: {self.is_time_delta_large}")
            #st.write(self.previous_hour)
            #st.write(self.current_hour)
    
    ##--------------------------------------------------------------##
    # remove the old geo_data_file
    ##--------------------------------------------------------------##

    def remove_geo_data_file(self):
        file_name = os.path.join(self.main_folder, "geo_data.csv")
    
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"Removed file: {file_name}")
        else:
            print(f"File not found: {file_name}")

    ##--------------------------------------------------------------##
    # Database call
    ##--------------------------------------------------------------##

    def database_call(self):
        #test only: 
        #GPS_measurements = ["SOG"]
        #real one
        GPS_measurements = ["LAT", "LON", "SOG", "COG", "RPM", "CELL"]
        print({self.previous_hour})
        print({self.current_hour})

        for measurement in GPS_measurements:
            # SQL query
            sql = f"""
                SELECT time, node_id, node_name, signal_name_alias, value
                FROM "{measurement}"
                WHERE time >= '{self.previous_hour}' AND time <= '{self.current_hour}'
                    AND value != 80.00861904
                    AND value != 8191.875
                ORDER BY
                    time ASC
                """

            try:
                # Query the database
                table = self.client.query(sql)
                #print(table) # to debug

                # Convert the result to a DataFrame
                df = table.to_pandas()
                #### CHANGE 
                self.current_working_directory = os.getcwd()
                folder_name = os.path.join(self.current_working_directory, 'dashboard', 'data', 'tmpfiles')
                #folder_name = r"C:\Users\axela\.cursor-tutor\AI\SSRS\Filip\app\data\tmpfiles"
                #### CHANGE NAME ABOVE
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Check if file exists and append timestamp if it does
                file_name = os.path.join(folder_name, f"GPSJAM_data_{measurement}.csv")
                if os.path.exists(file_name):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    file_name = os.path.join(folder_name, f"GPSJAM_data_{measurement}_{timestamp}.csv")

                df.to_csv(file_name, index=False)
                print(df) #debug for now

            except Exception as e:
                print(f"Error querying '{measurement}': {e}")

##--------------------------------------------------------------##
# Clean & Format all the data from DB call for GPSjam
##--------------------------------------------------------------##

class GPSCleanData:
    def __init__(self, train_data_dir, process_raw_path, manipulate_path):
        self.train_data_dir = train_data_dir
        self.process_raw_path = process_raw_path
        self.manipulate_path = manipulate_path
        self.env = os.environ.copy()

    def run_process_raw_script(self):
        process_raw_script = ["python", self.process_raw_path, self.train_data_dir]
        subprocess.run(process_raw_script, check=True, text=True, bufsize=1)

    def run_manipulate_script(self):
        # manipulate_script = ["python", self.manipulate_path, self.train_data_dir]
        # subprocess.run(manipulate_script, check=True, text=True, bufsize=1)

        manipulate_script = ["python", self.manipulate_path, self.train_data_dir]
        try:
            result = subprocess.run(manipulate_script, check=True, text=True, capture_output=True)
            print(result.stdout)  # Print standard output
        except subprocess.CalledProcessError as e:
            print(f"Error running manipulate script: {e}")
            print(f"Standard Output: {e.stdout}")
            print(f"Standard Error: {e.stderr}")

    def process_data(self):
        self.run_process_raw_script()
        self.run_manipulate_script()

    #deletes all other csv files in the directory
    def delete_csv_files_except_geo_data(self, folder_path):
        print("---------- Clearing Files ----------")
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for file in csv_files:
            if os.path.basename(file) != 'geo_data.csv':
                os.remove(file)
        print("---------- Files Cleared ----------")

##--------------------------------------------------------------##
# To skip CSS & JS because it doesnt work (for plot) in streamlit
##--------------------------------------------------------------##

class ScreenDimensions:
    def __init__(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.screen_width = (self.root.winfo_screenwidth()) * 0.9
        self.screen_height = (self.root.winfo_screenheight()) * 0.73

    def get_dimensions(self):
        return self.screen_width, self.screen_height
    
##--------------------------------------------------------------##
# Calculates data and shows the GPSplot
##--------------------------------------------------------------##

class GPSAnalyzer:
    ##----------------------------------------- init
    def __init__(self, gps_file_path, model_path, seq_size=60, resolution=3):
        load_dotenv()
        self.api_key = os.getenv('API_KEY_STADIA')
        self.seq_size = seq_size
        self.resolution = resolution
        self.gps_file_path = gps_file_path
        self.model_path = model_path
        self.engine = pd.read_csv(self.gps_file_path)
        self.model = keras.models.load_model(self.model_path)
        self.prepare_data()
    ##----------------------------------------- Data
    def adjust_rows_per_trip(self, df):
        adjusted_df = pd.DataFrame()
        for trip_id in df['tripID'].unique():
            trip_df = df[df['tripID'] == trip_id]
            excess_rows = len(trip_df) % self.seq_size
            if excess_rows != 0:
                trip_df = trip_df[:-excess_rows]
            adjusted_df = pd.concat([adjusted_df, trip_df], ignore_index=True)
        return adjusted_df
    
    ##----------------------------------------- Mapfix
    def to_sequence_non_overlapping(self, x):
        x_values = []
        for i in range(0, len(x) - self.seq_size + 1, self.seq_size):
            x_values.append(x.iloc[i:(i + self.seq_size)].values)
        return np.array(x_values)
    
    def create_hexagons(self, lat_range, lon_range):
        hex_addresses = set()
        lat_start, lat_end = lat_range
        lon_start, lon_end = lon_range
        lat = lat_start
        while lat < lat_end:
            lon = lon_start
            while lon < lon_end:
                hex_address = h3.geo_to_h3(lat, lon, self.resolution)
                hex_addresses.add(hex_address)
                lon += 0.1
            lat += 0.1
        return list(hex_addresses)
    
    def get_color(self, count):
        if count > 1:
            return '#910606'    #burgundy
        elif count == 1:
            return '#FFFF00'  # Yellow
        else:
            return '#086e1f'   #darkgreen
        
    def get_color_lighmode(self, count):
        if count > 1:
            return '#FF0000'  # Red
        elif count == 1:
            return '#FFFF00'  # Yellow
        else:
            return '#00FF00'  # Green 
    ##----------------------------------------- Data again
    def prepare_data(self):
        self.engine = self.adjust_rows_per_trip(self.engine)
        self.org_data = self.engine.copy()
        
        if 'RPM' not in self.engine.columns:
            self.engine['RPM'] = 0
        self.engine['RPM'] = self.engine['RPM'].apply(lambda x: 1 if x > 0 else 0)
        features = ['LAT_Delta', 'LON_Delta', 'RPM', 'COG', 'SOG', 'CELL']
        scaler = MinMaxScaler()
        cosine_scaler = lambda x: np.cos(np.deg2rad(x))
        self.engine['COG'] = cosine_scaler(self.engine['COG'])
        features_except_cog = ['LAT_Delta', 'LON_Delta', 'RPM', 'SOG', 'CELL']
        self.engine.loc[:, features_except_cog] = scaler.fit_transform(self.engine[features_except_cog])
        self.engine = self.engine[features]
        self.X_train = self.to_sequence_non_overlapping(self.engine)
        self.predictions = self.model.predict(self.X_train)
        ## What we need to chnage for binary
        self.test_anomalies = (self.predictions >= 0.994).astype(int) #0.72 was the old
        #######__________________        
        self.anomaly_indices = np.where(self.test_anomalies)[0] * self.seq_size
        self.anomaly_tripIDs = self.org_data.iloc[self.anomaly_indices]['tripID'].unique()
        self.anomaly_trips = pd.DataFrame()

        #newcode
        if len(self.anomaly_tripIDs) > 0:
            print(self.anomaly_tripIDs) # <-- Problemet är att det är fucked här
            for trip_id in self.anomaly_tripIDs:
                trip_data = self.org_data[self.org_data['tripID'] == trip_id]
                self.anomaly_trips = pd.concat([self.anomaly_trips, trip_data])
            self.normal_trips = self.org_data[~self.org_data['tripID'].isin(self.anomaly_tripIDs)]
        else:
            print("No anomaly tripIDs found.")
            self.normal_trips = self.org_data.copy()

        #old code, works if anomaly tripIDs >0 but not if it is equal to
        # print(self.anomaly_tripIDs) # <-- Problemet är att det är fucked här
        # for trip_id in self.anomaly_tripIDs:
        #     trip_data = self.org_data[self.org_data['tripID'] == trip_id]
        #     self.anomaly_trips = pd.concat([self.anomaly_trips, trip_data])
        # self.normal_trips = self.org_data[~self.org_data['tripID'].isin(self.anomaly_tripIDs)]
        # print(self.anomaly_trips)
    ##----------------------------------------- mapfix again
    def plot_results(self, lat_range, lon_range, map_path):
        center_lat = (lat_range[0] + lat_range[1]) / 2
        center_lon = (lon_range[0] + lon_range[1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4.5) #zoom rounds to whole number
        hex_addresses = self.create_hexagons(lat_range, lon_range)

        if self.api_key:
            folium.TileLayer(
                tiles=f'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{{z}}/{{x}}/{{y}}{{r}}.png?api_key={self.api_key}',
                attr='&copy; <a href="https://stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>',
                name='Stadia.AlidadeSmoothDark'
            ).add_to(m)
        hex_counts = defaultdict(set)
        #added try and except here
        try:
            for _, row in self.anomaly_trips.iterrows():
                hex_addr = h3.geo_to_h3(row['LAT'], row['LON'], self.resolution)
                hex_counts[hex_addr].add(row['tripID'])
        except Exception as e:
            print(f"Error processing anomaly trips: {e}")

        for _, row in self.normal_trips.iterrows():
            hex_addr = h3.geo_to_h3(row['LAT'], row['LON'], self.resolution)
            if hex_addr not in hex_counts:
                hex_counts[hex_addr] = set()

        for hex_addr in hex_counts:
            hexagon = h3.h3_to_geo_boundary(hex_addr)
            hex_points = [(lat, lon) for lat, lon in hexagon]
            count = len(hex_counts[hex_addr])
            color = self.get_color(count)  # This should now work correctly
            folium.Polygon(locations=hex_points, color=color, fill=True, fill_opacity=0.5, fill_color=color, weight=0.35).add_to(m)
        print(self.anomaly_trips) #debug

        #Added if statements here -> to handle if we have no anomalies
        if len(self.anomaly_tripIDs) > 0:   
            median_positions = self.anomaly_trips.groupby('tripID').agg({
                'LAT': 'median',
                'LON': 'median',
                'node_name': lambda x: ', '.join(sorted(x.unique()))
            }).reset_index()

            #Added this into the if statement
            for _, row in median_positions.iterrows():
                trip_id = row['tripID']
                node_name = row['node_name']
                max_distance = round(self.calculate_max_distance_to_median().loc[self.calculate_max_distance_to_median()['tripID'] == trip_id, 'max_distance_to_median (nm)'].values[0], 2)
                max_sog = round(self.calculate_max_sog_per_tripid().loc[self.calculate_max_sog_per_tripid()['tripID'] == trip_id, 'max_SOG (knots)'].values[0], 2)
                total_distance = round(self.calculate_total_distance_per_tripid().loc[self.calculate_total_distance_per_tripid()['tripID'] == trip_id, 'total_distance (nm)'].values[0], 2)

                popup_content = f"""
                <div style="font-family: sans-serif;">
                    <b>Vessel:</b> {node_name}<br>
                    <b>Max Distance:</b> {max_distance} nm<br>
                    <b>Max SOG:</b> {max_sog} knots<br>
                    <b>Total Distance:</b> {total_distance} nm
                </div>
                """
                
                iframe = IFrame(popup_content, width=200, height=110)
                popup = folium.Popup(iframe, max_width=300)
                folium.Marker(
                    location=[row['LAT'], row['LON']],
                    popup=popup,
                    icon=folium.Icon(color='darkblue', icon='info-sign')
                ).add_to(m)

            #Just this works too. 
            # folium.Marker(
            #     location=[row['LAT'], row['LON']],
            #     popup=f"Vessel: {row['node_name']}",
            #     icon=folium.Icon(color='darkblue', icon='info-sign')
            # ).add_to(m)

        ##--------------------------------------------------------------##
        # Changes the directory for the old map before saving the new one
        ##--------------------------------------------------------------## 
        
        current_time = datetime.datetime.now()
        date_folder = current_time.strftime("%Y-%m-%d")
        hour_folder = current_time.strftime("%H")
        #CHHHHANGE DIRRRR
        self.current_working_directory = os.getcwd()
        base_directory = os.path.join(self.current_working_directory, 'dashboard', 'graphics', 'OldMaps')
        #base_directory = r"C:\Users\axela\.cursor-tutor\AI\SSRS\Filip\app\graphics\OldMaps"
        full_folder_path = os.path.join(base_directory, date_folder)
        os.makedirs(full_folder_path, exist_ok=True)

        # Checks if the map already exists and renamess it
        if os.path.exists(map_path):
            old_map_name = f"Map_start_{hour_folder}.html"
            old_map_path = os.path.join(full_folder_path, old_map_name)
            # Check if the old map path already exists
            if not os.path.exists(old_map_path):
                os.rename(map_path, old_map_path)
                print(f"Old map saved as: {old_map_path}")
            else:
                print(f"Map already exists: {old_map_path}. No action taken.")

        # saves map
        m.save(map_path)
        return m
    
    ##--------------------------------------------------------------##
    # All functions below: Calculates values for the anomalies 
    ##--------------------------------------------------------------##

    def calculate_max_distance_to_median(self):
        max_distance_to_median = []
        for trip_id, group in self.anomaly_trips.groupby('tripID'):
            median_lat = group['LAT'].median()
            median_lon = group['LON'].median()
            median_location = (median_lat, median_lon)
            distances = group.apply(
                lambda row: geodesic((row['LAT'], row['LON']), median_location).nm, axis=1
            )
            max_distance = distances.max()
            node_name = group['node_name'].iloc[0]
            max_distance_to_median.append({
                'tripID': trip_id,
                'max_distance_to_median (nm)': max_distance,
                'node_name': node_name
            })
        #self.max_distance_to_median_df = pd.DataFrame(max_distance_to_median)
        
        return pd.DataFrame(max_distance_to_median)

    def calculate_max_sog_per_tripid(self):
        max_sog_per_tripid = []
        for trip_id, group in self.anomaly_trips.groupby('tripID'):
            trip_data_test = self.org_data[self.org_data['tripID'] == trip_id]
            max_sog = trip_data_test['SOG'].max()
            node_name = trip_data_test['node_name'].iloc[0]
            max_sog_per_tripid.append({
                'tripID': trip_id,
                'max_SOG (knots)': max_sog,
                'node_name': node_name
            })
        #self.max_sog_per_tripid_df = pd.DataFrame(max_sog_per_tripid)
        
        return pd.DataFrame(max_sog_per_tripid)

    def calculate_total_distance_per_tripid(self):
        total_distance_per_tripid = []
        for tripid in self.anomaly_tripIDs:
            trip_data = self.org_data[self.org_data['tripID'] == tripid]
            total_distance = 0.0
            for i in range(1, len(trip_data)):
                start_point = (trip_data.iloc[i-1]['LAT'], trip_data.iloc[i-1]['LON'])
                end_point = (trip_data.iloc[i]['LAT'], trip_data.iloc[i]['LON'])
                total_distance += geodesic(start_point, end_point).nm
            node_name = trip_data['node_name'].iloc[0]
            total_distance_per_tripid.append({
                'tripID': tripid,
                'total_distance (nm)': total_distance,
                'node_name': node_name
            })
        #self.total_distance_per_tripid_df = pd.DataFrame(total_distance_per_tripid)
        
        return pd.DataFrame(total_distance_per_tripid)
    


    # BEFORE
    # def run_all_calculations(self):
    #     max_distance_to_median_df = self.calculate_max_distance_to_median()
    #     max_sog_per_tripid_df = self.calculate_max_sog_per_tripid()
    #     total_distance_per_tripid_df = self.calculate_total_distance_per_tripid()
    #     return max_distance_to_median_df, max_sog_per_tripid_df, total_distance_per_tripid_df

##--------------------------------------------------------------##
#  Class to change the button and be able to pick date/time in main
##--------------------------------------------------------------##

##--------------------------------------------------------------##
#  Class to display the older maps
##--------------------------------------------------------------##

class GPSDispMapHistory:
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def get_map_file(self, date, time):
        # Construct the folder and filename based on date and time
        date_folder = date.strftime("%Y-%m-%d")
        hour_folder = f"Map_start_{time:02d}.html"
        file_path = os.path.join(self.base_directory, date_folder, hour_folder)
        return file_path if os.path.exists(file_path) else None

    def display_map(self, date, time, width, height):
        map_file = self.get_map_file(date, time)
        if map_file:
            with open(map_file, 'r') as f:
                map_html = f.read()
            # Using st. components here sice the map is html and not folium.map
            st.components.v1.html(map_html, width=width, height=height)
        else:
            st.write(f"No map file found for {date} at {time:02d}:00")

    ## test
    def display_map_with_sidebar(self, screen_width, screen_height):
        st.sidebar.header("Map Options")

        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = datetime.datetime.now().date()
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = datetime.datetime.now().hour

        selected_date = st.sidebar.date_input("Select Date", value=st.session_state.selected_date, key="date_slidebar") #added key
        selected_time = st.sidebar.selectbox("Select Time", list(range(24)), index=st.session_state.selected_time)

        col1, col2 = st.sidebar.columns(2)
        with col1: 
            Now = st.button("Now", key="now_button")
        with col2:
            Update = st.button("Update date", key="update_button") #added key now

        if Now:
            current_time = datetime.datetime.now()
            #st.empty()
            st.session_state.selected_date = current_time.date()
            st.session_state.selected_time = current_time.hour - 1
            #self.display_map(selected_date, selected_time, screen_width, screen_height)
            return True
        elif Update:
            current_time = datetime.datetime.now()
            #st.empty()
            st.session_state.selected_date = selected_date
            st.session_state.selected_time = selected_time
            self.display_map(selected_date, selected_time, screen_width, screen_height)
            #st.write(selected_date)
            #st.write(selected_time)
            return False
        else:
            print("non selected")
            return False     
        
##--------------------------------------------------------------##
#  (SUPERMUCHTESTING) Autorefresh page, automatically update the map
##--------------------------------------------------------------##  

class GPSAutomaticReload:
 
    
    def __init__(self):
        self.refresh_bool = False
    ########################### varje kvart
    def HelperLoop(self):
        try:
            now = datetime.datetime.utcnow()
            next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            print(next_hour)
            while True:
                time_to_sleep = (next_hour - datetime.datetime.utcnow()).total_seconds() + 30
                if time_to_sleep <= 0:
                    break
                time.sleep(min(900, time_to_sleep))  # Sleep in shorter intervals
                if st.session_state.get("update_button"):
                    self.refresh_bool = True
                    return self.refresh_bool
            self.refresh_bool = True
            return self.refresh_bool
        except KeyboardInterrupt:
            import sys
            sys.exit()
    ###########################


    #______________ DETTA ÄR VARJE MINUT
    # def HelperLoop(self):
    #     try:
    #         now = datetime.datetime.utcnow()
    #         next_minute = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
    #         print(next_minute)
    #         while True:
    #             time_to_sleep = (next_minute - datetime.datetime.utcnow()).total_seconds()
    #             #print("1")
    #             if time_to_sleep <= 0:
    #                 break
    #             time.sleep(min(1, time_to_sleep))  # Sleep in shorter intervals (1 second)
    #             #print("2")
    #             if st.session_state.get("update_button"):
    #                 self.refresh_bool = True
    #                 return self.refresh_bool
    #         self.refresh_bool = True
    #         return self.refresh_bool
    #     except KeyboardInterrupt:
    #         import sys
    #         sys.exit()

##--------------------------------------------------------------##
# get all directories
##--------------------------------------------------------------##  

# class GPSGetAllDirs:

#     def __init__(self):
#         with open('../config.yaml', 'r') as file:
#             config = yaml.safe_load(file)
#         self.base_dir = config['BASE_DIR']
#         self.geo_data_path = os.path.join(self.base_dir, *config['GEO_DATA_PATH'].split('/'))
#         self.train_data_dir = os.path.join(self.base_dir, *config['TRAIN_DATA_DIR'].split('/'))
#         self.process_raw_path = os.path.join(self.base_dir, *config['PROCESS_RAW_PATH'].split('/'))
#         self.manipulate_path = os.path.join(self.base_dir, *config['MANIPULATE_PATH'].split('/'))
#         self.model_path = os.path.join(self.base_dir, *config['MODEL_PATH'].split('/'))
#         self.map_path = os.path.join(self.base_dir, *config['MAP_PATH'].split('/'))
#         self.old_map_path = os.path.join(self.base_dir, *config['OLD_MAPS_PATH'].split('/'))

#     def get_base_dir(self):
#         return self.base_dir

#     def get_geo_data_path(self):
#         return self.geo_data_path

#     def get_train_data_dir(self):
#         return self.train_data_dir

#     def get_process_raw_path(self):
#         return self.process_raw_path

#     def get_manipulate_path(self):
#         return self.manipulate_path

#     def get_model_path(self):
#         return self.model_path

#     def get_map_path(self):
#         return self.map_path
    
#     def get_oldmap_path(self):
#         return self.old_map_path
    
##--------------------------------------------------------------##
# Aggregate all old map files to one for the entire day
##--------------------------------------------------------------##  

class GPSAggregateAllMaps:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('API_KEY_STADIA')

    def getdirfromslide(self):
        selected_date = st.session_state.get("date_slidebar", None)
        if selected_date is None:
            selected_date = st.sidebar.date_input("Select Date", value=st.session_state.selected_date, key="date_slidebar")
        formatted_date = selected_date.strftime("%Y-%m-%d")
        self.current_working_directory = os.getcwd()
        #directory = os.path.join(r"C:\Users\axela\.cursor-tutor\AI\SSRS\Filip\app\graphics\OldMaps", formatted_date)
        directory = os.path.join(self.current_working_directory, 'dashboard', 'graphics', 'OldMaps', formatted_date)
        return directory

    def extract_polygons(self, dir):
        directory = dir

        # Regex to match the polygon data
        polygon_regex = re.compile(r"L\.polygon\(\s*(\[\[.*?\]\])\s*,\s*{.*?\"color\":\s*\"(#[0-9A-Fa-f]{6})\".*?}\s*\)\.addTo\(.*?\);", re.DOTALL)

        # Dictionary to store the polygons and their colors
        polygons = defaultdict(list)
        
        # Read and parse each HTML file
        for filename in os.listdir(directory):
            if filename.endswith(".html"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    matches = polygon_regex.findall(content)
                    for match in matches:
                        coordinates, color = match
                        polygons[coordinates].append(color)

        final_polygons = {}
        #works -----------------------------
        # Determine the final color for each polygon
        # for coordinates, colors in polygons.items():
        #     if all(color == "#086e1f" for color in colors):
        #         final_color = "#086e1f"
        #     else:
        #         final_color = "#FF0000"
        #     final_polygons[coordinates] = final_color

        # return final_polygons
        #-----------------------------------
        # Determine the final color for each polygon
        for coordinates, colors in polygons.items():
            unique_colors = set(colors)
            print(unique_colors)  # Debug print

            if '#FF0000' in unique_colors:
                final_color = '#910606'
            elif '#910606' in unique_colors:
                final_color = '#910606'
            elif '#FFFF00' in unique_colors:
                final_color = '#FFFF00'
            else:
                final_color = '#086e1f'

            final_polygons[coordinates] = final_color
            print(f"Final Color for {coordinates}: {final_color}")
        return final_polygons
    

    def create_map(self, final_polygons, map_path):
        # Create a folium map
        m = folium.Map(location=[62.5, 17.5], zoom_start=4.5)
        if self.api_key:
            try:
                folium.TileLayer(
                    tiles=f'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{{z}}/{{x}}/{{y}}{{r}}.png?api_key={self.api_key}',
                    attr='&copy; <a href="https://stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>',
                    name='Stadia.AlidadeSmoothDark'
                ).add_to(m)
            except Exception as e:
                print(f"An error occurred while adding the tile layer: {e}")

        # Add polygons to the map
        for coordinates, color in final_polygons.items():
            coords = eval(coordinates)
            folium.Polygon(locations=coords, color=color, fill=True, fill_opacity=0.5, fill_color=color, weight=0.35).add_to(m)

        # Save the map
        m.save(map_path)
        print(f"Map saved to {map_path}")
    
    def display_aggregated_map(self, directory, screen_width, screen_height):
        map_path = os.path.join(directory, "aggregated_map.html")
        final_polygons = self.extract_polygons(directory)
        self.create_map(final_polygons, map_path)
        
        # Display the map in Streamlit
        with open(map_path, 'r') as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=screen_height, width=screen_width)

class GPScwd:
    def __init__(self):
        self.current_working_directory = os.getcwd()
        pass
    
    def getdir(self):
        oldmaps = os.path.join(self.current_working_directory, 'dashboard', 'graphics', 'OldMaps')
        geo_data_path = os.path.join(self.current_working_directory, 'dashboard', 'data', 'tmpfiles', 'geo_data.csv')
        train_data_dir = os.path.join(self.current_working_directory, 'dashboard', 'data', 'tmpfiles')
        process_raw_path = os.path.join(self.current_working_directory, 'dashboard', 'modules', 'GpsDataPipeline', 'process_raw.py')
        manipulate_path = os.path.join(self.current_working_directory,'dashboard', 'modules', 'GpsDataPipeline', 'manipulate.py')
        #model_path = os.path.join(self.current_working_directory, '..', 'models', 'model_info', 'LSTM_FINETUNED_v2_model.keras')
        model_path = os.path.join(self.current_working_directory, 'models', 'tunings', 'fine_tune_test.keras')
        current_map_path = os.path.join(self.current_working_directory, 'dashboard', 'graphics', 'results_hexagon_map.html')
        return oldmaps, geo_data_path, train_data_dir, process_raw_path, manipulate_path, model_path, current_map_path
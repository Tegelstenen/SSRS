
# TODO: Write Readme
# TODO: Set up database instead of storing locally csv files
# TODO: Clean up axels shit
# TODO: Think of to deploy the stremlit

import streamlit as st
from streamlit_folium import folium_static

import os

from utils.helper import HelperFunctions
from modules.anomaly import AnomalyPlots, AnomalySelectors
from modules.sidebar import SideBar 
from modules.GPSjam import GPSAnalyzer, ScreenDimensions, GPSCleanData, GPSDataProcessor, GPSDispMapHistory, GPSAutomaticReload, GPSAggregateAllMaps, GPScwd#, GPSGetAllDirs
from modules.sidebar import SideBar
from modules.database import Database

  
##########################  
# Data and Constants 
##########################
DATA, ERRORS, FULL_ERRORS = HelperFunctions.get_all_data()
BOAT_NAMES = HelperFunctions.get_boats()   
FEATURES = HelperFunctions.get_shown_features()
FEATURE_COLORS = HelperFunctions.get_colors() 
full_data_db = Database("full_data")
 
########################## 
# Initial page config  
##########################   
st.set_page_config( 
    page_title="SSRS Anomaly Detection", 
    page_icon="⚓️",
    layout="wide", 
    initial_sidebar_state="expanded", 
) 

##########################
# Body of anomalies 
##########################
def anomly_body():
    tabs = st.tabs(BOAT_NAMES)
    for tab, boat in zip(tabs, BOAT_NAMES): 
        with tab:
 
            HelperFunctions.write_heading("Anomaly Heatmap")
            AnomalyPlots.show_heat_map(boat, ERRORS, FEATURES)
            AnomalyPlots.show_mse_scatter(boat, FULL_ERRORS, FEATURES)
            
            st.divider()

            HelperFunctions.write_heading("Comparisons between Port and Starboard")
            date = AnomalySelectors.show_selections(ERRORS, boat)
            if date: 
                
                data = full_data_db.get_data(boat, date)
                
                # TODO: Add back in and make visually pleasing and infromative. Suggestions to stack both sides on same fig
                # AnomalyPlots.show_daily_data(
                #     date, boat, data, FEATURES, FEATURE_COLORS
                # )
            
                AnomalyPlots.show_differences(
                    data, FEATURES, FEATURE_COLORS, date, boat
                )

##########################
# Main body of GPS disruptions
########################## 

def gps_body(): 
    # Get working direcroty for dynamic paths
    gps_cwd = GPScwd() 
    oldmaps_dir, geo_data_path, train_data_dir, process_raw_path, manipulate_path, model_path, current_map_path = gps_cwd.getdir()
    
    # Get screen dimensions to skip HTML and Streamlit conflicts
    screen = ScreenDimensions()
    screen_width, screen_height = screen.get_dimensions()
    
    # Display older maps
    print(oldmaps_dir)
    map_history = GPSDispMapHistory(base_directory=oldmaps_dir)
    boolean_map_disp = map_history.display_map_with_sidebar(screen_width, screen_height)
    print(f'bool map disp: {boolean_map_disp}')
    
    # Reload page
    map_placeholder = st.empty()
    gps_auto_reload= GPSAutomaticReload() 
     
    # Aggregateing all maps:  
    aggregator = GPSAggregateAllMaps()
    if st.sidebar.button("Display Aggregated Map"):
        try:
            dirren = aggregator.getdirfromslide()
            aggregator.display_aggregated_map(directory=dirren, screen_width=screen_width, screen_height=screen_height)       
        except Exception as e:
            st.write(f"An error occurred: {e}. No map folder found")
    
    while True:
        if boolean_map_disp == True:
            #Check need to update databas -> database call
            gps_processor = GPSDataProcessor(geo_data_path)
            time_delta_large_bool = gps_processor.use_comparison_results() 
            print("hello darkness")
            #time_delta_large_bool = True #debug to check 
            print(f'time delta: {time_delta_large_bool}')
            if time_delta_large_bool:
                gps_cleaner = GPSCleanData(
                train_data_dir=train_data_dir,
                process_raw_path=process_raw_path,
                manipulate_path=manipulate_path
                ) 
                gps_cleaner.process_data() 

                #Cleans up all old CSV-Files  
                gps_cleaner.delete_csv_files_except_geo_data(folder_path=train_data_dir)
                print("Cleaned all tha data for u sire")
            #Plots the map
            gps_plotmat = GPSAnalyzer(
                gps_file_path=geo_data_path,
                model_path=model_path
            )
            map_object = gps_plotmat.plot_results(
            lat_range=(55.0, 70.0),
            lon_range=(10.0, 25.0),
            map_path=current_map_path
            )
    
            if map_object is not None:
                with map_placeholder:
                    screen = ScreenDimensions()
                    screen_width, screen_height = screen.get_dimensions()
                    folium_static(map_object, width=screen_width,height=screen_height)
            else:
                st.write("Failed to create map object")
                if os.path.exists(current_map_path):
                    st.write("Attempting to display saved map file")
                    with open(current_map_path, 'r') as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, width=screen_width, height=screen_height)
                else:
                    st.write(f"No map file found at {current_map_path}")

        refresh_status = gps_auto_reload.HelperLoop()
        print(refresh_status)


##########################
# Sidebar
##########################
def sidebar_body():
    SideBar()
    SideBar.images()
    SideBar.buttons()


##########################
# main
##########################
def main():

    sidebar_body()

    if st.session_state.get("page") == "overview":
        anomly_body()
    elif st.session_state.get("page") == "gps_disruptions":
        gps_body()

    return None


# Run main()
if __name__ == "__main__":
    main()
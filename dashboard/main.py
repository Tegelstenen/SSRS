
# TODO: 
# - create database ?
# - Make heatmap relative to its own value ?
#   - using a baseline value can ensure that
# - if trips overlaps two days make that still visable (I THINK THIS IS TRUE IN How TRIP ID is calculated but not how it is displayed)
# - Fix Engine_load always being anomaly
# - Ensure daily updates
# - 

import streamlit as st
from utils.helper import HelperFunctions
from modules.anomaly import AnomalyPlots, AnomalySelectors
from modules.sidebar import SideBar 

 
##########################
# Data and Constants 
##########################
DATA, ERRORS, FULL_ERRORS = HelperFunctions.get_all_data()
BOAT_NAMES = HelperFunctions.get_boats()   
FEATURES = HelperFunctions.get_shown_features()
FEATURE_COLORS = HelperFunctions.get_colors() 
 
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

            HelperFunctions.write_heading("Data for the day")
            date, signal_instance = AnomalySelectors.show_selections(ERRORS, boat)
            if date:
                AnomalyPlots.show_daily_data(
                    date, boat, DATA, FEATURES, FEATURE_COLORS, signal_instance
                )

            st.divider()

            HelperFunctions.write_heading("Comparisons between Port and Starboard")
            if date:
                AnomalyPlots.show_differences(
                    DATA, FEATURES, FEATURE_COLORS, date, boat
                )

##########################
# Main body of GPS disruptions
########################## 
def gps_body():
    pass


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
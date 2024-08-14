
# TODO: Write Readme
# TODO: Set up database instead of storing locally csv files
# TODO: Clean up axels shit
# TODO: Think of to deploy the stremlit

import streamlit as st
from utils.helper import HelperFunctions
from modules.anomaly import AnomalyPlots, AnomalySelectors
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
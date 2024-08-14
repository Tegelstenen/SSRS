import streamlit as st

import os

from utils.helper import HelperFunctions
from utils.config_manager import ConfigManager

config = ConfigManager()

class SideBar:
    @staticmethod
    def images() -> None:
        grpahics_dir = config.get("GRAPHICS_DIR")
        base_dir = config.get("BASE_DIR")
        st.sidebar.markdown(
                """<p align="center"><img src='data:image/png;base64,{}' class='img-fluid' style='width: 50%;'></p>""".format(
                    HelperFunctions.img_to_bytes(f"{base_dir}/{grpahics_dir}/SSRS.png")
                ),
                unsafe_allow_html=True,
            )
        
        st.sidebar.markdown(
            """<p align="center"><img src='data:image/png;base64,{}' class='img-fluid' style='filter: invert(1); width: 50%;'></p>""".format(
                HelperFunctions.img_to_bytes(f"{base_dir}/{grpahics_dir}/text.png")
            ),
            unsafe_allow_html=True,
        )
    
    @staticmethod
    def buttons() -> None:
        # Custom CSS to make buttons as wide as expanders
        st.sidebar.markdown(
            """
            <style>
            .stButton button {
                width: 100%;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        if st.sidebar.button("__Show Overview__"):
            st.session_state["page"] = "overview"

        st.sidebar.markdown("###")
        if st.sidebar.button("__GPS Disruptions__"):
            st.session_state["page"] = "gps_disruptions"
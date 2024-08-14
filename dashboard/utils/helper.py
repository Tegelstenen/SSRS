import pandas as pd
import plotly.express as px
import streamlit as st

import os
import base64
from pathlib import Path
from typing import Tuple

from utils.config_manager import ConfigManager


class HelperFunctions:
    config = ConfigManager()

    @classmethod
    def img_to_bytes(cls, img_path: str) -> str:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded

    @staticmethod
    def _get_data(dir: str) -> pd.DataFrame:
        return pd.read_csv(dir)

    @staticmethod
    def _get_full_path(cls) -> str:
        #base_dir = cls.config.get("BASE_DIR")
        base_dir = os.getcwd()
        app_data_dir = cls.config.get("APP_DATA_DIR")
        full_path = os.path.join(base_dir, app_data_dir)
        return full_path
    
    @classmethod
    def get_all_data(cls) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_path = cls._get_full_path(cls)
        data = cls._get_data(os.path.join(full_path, "data.csv"))
        errors = cls._get_data(os.path.join(full_path, "errors.csv"))
        full_errors = cls._get_data(os.path.join(full_path, "full_errors.csv"))
        return data, errors, full_errors

    @classmethod
    def get_boats(cls) -> list:
        boats = cls.config.get("BOATS")
        return boats

    @classmethod
    def get_shown_features(cls) -> list:
        features = cls.config.get("ENGINE_FEATURES")
        return features

    @classmethod
    def get_colors(cls) -> dict:
        features = cls.get_shown_features()
        colors = {
            feature: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, feature in enumerate(features)
        }
        return colors
    
    @classmethod
    def write_heading(cls, text: str) -> None:
        st.markdown(
                f"""
            <div style='text-align: center; font-size: 2em;'>
                {text}
            </div> 
            """,
                unsafe_allow_html=True,
            )

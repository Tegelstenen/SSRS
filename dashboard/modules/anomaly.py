import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import plotly.express as px
from modules.get_data import InfluxDBClient
from scipy.stats import zscore

import time
from datetime import timedelta

from utils.config_manager import ConfigManager
from modules.database import Database




config = ConfigManager()

FEATURES = config.get("ENGINE_FEATURES")
FEATURE_COLORS = {
    feature: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    for i, feature in enumerate(FEATURES)
}


class AnomalyPlots:
    @staticmethod
    def plot_engine_features(boat, date, smoothing_window):
        db_client = _get_db_client()
        cols = st.columns(2)
        for i, feature in enumerate(FEATURES):
            col = cols[i % 2]  # Alternate between the two columns
            with col:
                df = _prepare_data(boat, date, db_client, feature)
                traces = _create_traces(df, feature, smoothing_window)
                _plot_figures(feature, traces)
                

    @classmethod
    def show_heat_map(cls, boat):
        data = _get_daily_errors(boat)
        
        # Transpose the data to have features as rows and dates as columns
        heatmap_data = data.set_index("date")[FEATURES].T

        # Define the color scale and range
        color_scale = px.colors.sequential.Reds
        color_range = config.get("HEAT_MAP_SCALE")

        # Create the heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Date", y="Feature", color="Value"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale=color_scale,
            range_color=color_range,
        )

        fig.update_layout(
            dragmode="pan",  # Enable drag mode
            xaxis=dict(fixedrange=True, type="category", showticklabels=False),
            yaxis=dict(fixedrange=True),  # Disable zoom on y-axis
        )

        st.plotly_chart(fig)

    @classmethod
    def show_mse_scatter(cls, boat: str) -> None:
        
        FULL_ERRORS = _get_full_errors()
        
        boat_data = FULL_ERRORS.query(f"node_name == '{boat}'")
    
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        for feature in FEATURES:
            scatter = go.Scatter(
                x=boat_data["date"],
                y=boat_data[feature],
                mode="markers",
                name=feature,
                marker=dict(
                    color=FEATURE_COLORS.get(feature, "black")
                ),  # Use the color mapping
            )
            fig.add_trace(scatter)

            fig.update_layout(
                title=f"Feature Trends for {boat}",
                xaxis_title="Date",
                yaxis_title="Feature Values",
                yaxis=dict(range=[0, 0.35]),  # Set y-axis range to be between 0 and 1
            )

        # Update layout to place legend on top
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.07, xanchor="center", x=0.5
            ),
            hovermode="x",
            xaxis=dict(
                title="Time",
            ),
            yaxis=dict(
                fixedrange=True,  # Prevent zooming on y-axis
            ),
            dragmode="zoom",  # Enable zoom mode
            plot_bgcolor="rgba(225, 228, 233, 0.8)",
        )

        st.plotly_chart(fig)


class AnomalySelectors:
    @classmethod
    def show_selections(cls, boat: str) -> None:
        data = _get_daily_errors(boat)
        dates = data.query(f'node_name == "{boat}"')["date"].unique()
        date = st.selectbox("Select a date", dates, key=f"{boat} at {dates}")
        return date


@st.cache_data(show_spinner="Loading data")
def _prepare_data(boat, date, _db_client, feature):
    df = _get_data(boat, date, _db_client, feature)
    if df.empty:
        return df  # Return the empty DataFrame if no data is found
    df = _generate_trip_id(df)
    df = _set_signal_instance(df)
    df = _remove_outliers(df)
    df.sort_values("time", inplace=True, ascending=True)
    return df


def _set_signal_instance(df: pd.DataFrame) -> pd.DataFrame:
    df["signal_instance"] = df["signal_instance"].astype(str)
    df["signal_instance"] = df["signal_instance"].replace({"0": "P", "1": "SB"})
    return df


def _add_na(data: pd.DataFrame) -> pd.DataFrame:
    # Function to add NA rows
    def add_na_rows(group):
        # Create empty DataFrame with the same structure as the group
        first_row = pd.DataFrame(columns=group.columns).astype(group.dtypes.to_dict())
        last_row = pd.DataFrame(columns=group.columns).astype(group.dtypes.to_dict())

        # Set all columns except groupby columns to NA
        for col in group.columns:
            if col not in ["node_name", "signal_instance", "TRIP_ID"]:
                first_row.at[0, col] = np.nan
                last_row.at[0, col] = np.nan

        # Adjust the time for the new rows
        time_diff = pd.Timedelta(seconds=1)
        first_row.at[0, "time"] = group["time"].iloc[0] - time_diff
        last_row.at[0, "time"] = group["time"].iloc[-1] + time_diff

        # Fill the groupby columns
        first_row.at[0, "node_name"] = group.iloc[0]["node_name"]
        first_row.at[0, "signal_instance"] = group.iloc[0]["signal_instance"]
        first_row.at[0, "TRIP_ID"] = group.iloc[0]["TRIP_ID"]

        last_row.at[0, "node_name"] = group.iloc[-1]["node_name"]
        last_row.at[0, "signal_instance"] = group.iloc[-1]["signal_instance"]
        last_row.at[0, "TRIP_ID"] = group.iloc[-1]["TRIP_ID"]

        # Concatenate the new rows with the original group
        return pd.concat([first_row, group, last_row])

    # Apply the function to each group
    data = data.groupby(["TRIP_ID"], group_keys=False).apply(add_na_rows)

    # Reset the index and sort by time
    data = data.reset_index(drop=True).sort_values(
        by=["node_name", "signal_instance", "TRIP_ID", "time"]
    )

    return data


def _get_data(
    boat: str, date: str, _db_client: InfluxDBClient, feature: str
) -> pd.DataFrame:
    df = _db_client.query(boat, date, feature)
    time.sleep(2)  # to not overload the db
    return df


@st.cache_resource
def _get_db_client():
    db_client = InfluxDBClient()
    return db_client


def _generate_trip_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time")
    df["TRIP_ID"] = 1
    current_trip_id = 1
    previous_time = df["time"].iloc[0]

    for index, row in df.iterrows():
        current_time = row["time"]
        if current_time - previous_time > timedelta(minutes=10):
            current_trip_id += 1
        df.at[index, "TRIP_ID"] = current_trip_id
        previous_time = current_time

    return df


def _smooth_data(df: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    df["value"] = df["value"].rolling(window=smoothing_window).mean()
    return df


def _remove_outliers(df):
    z_scores = zscore(df["value"])
    df = df[z_scores < 3]
    return df


def _plot_figures(feature, traces):
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{feature}",
        yaxis=dict(fixedrange=True),
        dragmode="zoom",
        plot_bgcolor="rgba(225, 228, 233, 0.8)",
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


def _create_traces(df, feature, smoothing_window):
    traces = []
    for instance in df["signal_instance"].unique():
        instance_df = df[df["signal_instance"] == instance]
        instance_df = _add_na(instance_df)
        instance_df = _smooth_data(instance_df, smoothing_window)
        color = FEATURE_COLORS.get(feature, "black")

        # If the signal_instance is 'P', use a lighter color
        if instance == "P":
            # Convert the color to RGB, lighten it, then convert back to hex
            rgb = px.colors.hex_to_rgb(color)
            lightened_rgb = [
                min(255, int(c * 1.5)) for c in rgb
            ]  # Increase brightness by 50%
            color = "#{:02x}{:02x}{:02x}".format(*lightened_rgb)

        traces.append(
            go.Scatter(
                x=instance_df["time"],
                y=instance_df["value"],
                mode="lines",
                name=f"{instance}",
                line=dict(width=2, color=color),
            )
        )

    return traces

@st.cache_data(show_spinner="Loading data")
def _get_full_errors():
    db = Database()
    cursor = db.connection.cursor()
    table_name = "error_data"
    fetched_data = db.fetch_all(table_name)
    full_errors = pd.DataFrame(fetched_data)
    db.close()
    return full_errors

@st.cache_data(show_spinner="Loading data")
def _get_daily_errors(boat):
    FULL_ERRORS = _get_full_errors()
    data = (FULL_ERRORS
            .query(f"node_name == '{boat}'")
            .assign(date=lambda x: pd.to_datetime(x['date']).dt.strftime('%Y-%m-%d'))
            .groupby(["node_name", "date"])
            .mean()
            .reset_index())
            
    return data
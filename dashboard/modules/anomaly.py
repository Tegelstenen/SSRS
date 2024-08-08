# TODO:
# - add somthing that highlight the individual trips
#   - alterntively create a trip separator
# - Att scatter points of extra high anomaly value to the comparison plots
#   - add it under a "show anomalies" ticker
# - maximise usage of config

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

from utils.config_manager import ConfigManager

config = ConfigManager()
AVOID_IN_COMPARISON_PLOTS = config.get("AVOID_IN_COMPARISON_PLOTS")


class AnomalyPlots:
    @staticmethod
    def _add_na(data: pd.DataFrame) -> pd.DataFrame:
        # Function to add NA rows
        def add_na_rows(group):
            first_row = group.iloc[0].copy() 
            last_row = group.iloc[-1].copy()
            
            # Set all columns except groupby columns to NA
            for col in first_row.index:
                if col not in ["node_name", "signal_instance", "TRIP_ID"]:
                    first_row[col] = np.nan
                    last_row[col] = np.nan
            
            # Adjust the time for the new rows
            time_diff = pd.Timedelta(seconds=1)
            first_row['time'] = group['time'].iloc[0] - time_diff
            last_row['time'] = group['time'].iloc[-1] + time_diff
            
            # Concatenate the new rows with the original group
            return pd.concat([first_row.to_frame().T, group, last_row.to_frame().T])

        # Apply the function to each group
        data = data.groupby(["node_name", "signal_instance", "TRIP_ID"], group_keys=False).apply(add_na_rows)
        
        # Reset the index and sort by time
        data = data.reset_index(drop=True).sort_values(by=["node_name", "signal_instance", "TRIP_ID", "time"])
        
        return data
    
    @staticmethod
    def _query_data(
        data: pd.DataFrame, date: str, boat: str, features: list[str]
    ) -> pd.DataFrame:
        data = (
            data.query(f'node_name == "{boat}" & date == "{date}"')
            .filter(items=["node_name", "time", "signal_instance", "TRIP_ID"] + features)
            .rename(columns={"SOG_adapt": "SOG"})
            .sort_values(by="time")
        )
        return data

    @staticmethod
    def _add_date(data: pd.DataFrame, date: str) -> pd.DataFrame:
        data["time"] = pd.to_datetime(
            date + " " + data["time"], format="%Y-%m-%d %H:%M:%S"
        )
        return data

    @staticmethod
    def _smooth_data(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        data[features] = data[features].rolling(window=20).mean()
        return data

    @staticmethod
    def _split_data(data: pd.DataFrame) -> pd.DataFrame:
        sb_data = data[data["signal_instance"] == "SB"]
        p_data = data[data["signal_instance"] == "P"]
        return sb_data, p_data

    @classmethod
    def _preprocess_data(
        cls, data: pd.DataFrame, features: list[str], date: str, boat: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = cls._query_data(data, date, boat, features)
        data = cls._add_date(data, date)
        data = cls._add_na(data)
        data = cls._smooth_data(data, features)
        sb_data, p_data = cls._split_data(data)
        return sb_data, p_data
    
    @staticmethod
    def _create_single_plot(
        data: pd.DataFrame, features: list[str], feature_colors: dict, title: str
    ) -> go.Figure:
        fig = go.Figure()

        for i, feature in enumerate(features):
            fig.add_trace(
                go.Scatter(
                    x=data["time"],
                    y=data[feature],
                    name=feature,
                    line=dict(color=feature_colors.get(feature, "black")),
                    yaxis=f"y{i+1}",
                )
            )

        # Update layout with multiple y-axes
        yaxis_layouts = {
            f"yaxis{i+1}": dict(
                title="",
                titlefont=dict(color=feature_colors.get(feature, "black")),
                tickfont=dict(color=feature_colors.get(feature, "black")),
                overlaying="y" if i > 0 else None,
                side="right" if i % 2 else "left", 
                position=1.0 - (i * 0.04) if i % 2 else 0.0 + (i * 0.04),
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                fixedrange=True,  # Prevent zooming on y-axis
            )
            for i, feature in enumerate(features)
        }

        fig.update_layout(
            height=400,  # Adjust as needed
            title=title,
            legend=dict(groupclick="toggleitem"),
            hovermode="x unified", 
            xaxis=dict(
                title="Time",
            ),
            dragmode="zoom",
            plot_bgcolor='rgba(225, 228, 233, 0.8)',
            paper_bgcolor='rgba(155, 158, 166, 0.8)',
            template="none",
            **yaxis_layouts,
        )

        return fig

    @classmethod
    def _daily_data(
        cls,
        date: str,
        boat: str,
        data: pd.DataFrame,
        features: list[str],
        feature_colors: dict,
    ) -> tuple[go.Figure, go.Figure]:
        sb_data, p_data = cls._preprocess_data(data, features, date, boat)

        sb_fig = cls._create_single_plot(sb_data, features, feature_colors, f"")
        p_fig = cls._create_single_plot(p_data, features, feature_colors, f"")

        return sb_fig, p_fig
    
    @classmethod
    def show_daily_data(cls, date: str, boat: str, data: pd.DataFrame, features: list[str], feature_colors: dict, signal_instance: str) -> None:
        starboard, port = cls._daily_data(date, boat, data, features, feature_colors)
        if signal_instance == "SB":
            st.plotly_chart(starboard, key=f"{boat}_{date}_chart")
        elif signal_instance == "P":
            st.plotly_chart(port, key=f"{boat}_{date}_chart")

    @classmethod 
    def show_heat_map(cls, boat, errors, features):
        data = errors[errors["node_name"] == boat] 
        # Transpose the data to have features as rows and dates as columns
        heatmap_data = data.set_index("date")[features].T

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
            # range_color=color_range,
        )

        #  Update layout to enable drag mode and disable zoom
        fig.update_layout(
            dragmode="pan",  # Enable drag mode
            xaxis=dict(fixedrange=True, type="category", showticklabels=False),
            yaxis=dict(fixedrange=True),  # Disable zoom on y-axis
            title=go.layout.Title(
                text=(
                    "<sup>The plot can be used to see individual days where the error where larger than usual <br>"
                    "Below you can select a date to see the individual values for that day</sup>"
                ),
                xref="paper",
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
        )

        fig.add_annotation(
            text="",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="center",
            x=0.5,
            y=1.3,
            bordercolor="black",
            borderwidth=1,
        )

        st.plotly_chart(fig)

    @classmethod
    def _daily_difference(cls, data: pd.DataFrame, date: str, boat: str, items: str, features: list[str], feature_colors: dict) -> None:
        data = cls._query_data(data, date, boat, features)
        data = cls._add_date(data, date)
        data = cls._add_na(data)
        data = data.sort_values("time")
        base_color = feature_colors.get(
            items[-1], "#000000"
        ) 
        base_color_rgb = tuple(
            int(base_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
        )
        def adjust_color_brightness(color, factor):
            return tuple(min(255, int(c * factor)) for c in color)

        port_color = f"rgb{adjust_color_brightness(base_color_rgb, 0.7)}"  # Darker shade
        starboard_color = (
            f"rgb{adjust_color_brightness(base_color_rgb, 1.3)}"  # Lighter shade
        )

        fig = go.Figure()

        for signal in data["signal_instance"].unique():
            filtered_data = data[data["signal_instance"] == signal]
            color = port_color if signal == "P" else starboard_color
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["time"],
                    y=filtered_data[items[-1]],
                    mode="lines",
                    name=signal,
                    line=dict(color=color),
                    marker=dict(size=5),
                )
            )

        # Update layout to place legend on top
        fig.update_layout(
            title=items[-1],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x",
            xaxis=dict(
                title="Time",
            ),
            yaxis=dict(
                fixedrange=True,  # Prevent zooming on y-axis
            ),
            dragmode="zoom",  # Enable zoom mode
            plot_bgcolor='rgba(225, 228, 233, 0.8)',
            paper_bgcolor='rgba(155, 158, 166, 0.8)',
            template="none",
        )

        st.plotly_chart(fig)

    @classmethod
    def show_differences(cls, data: pd.DataFrame, features: list[str], feature_colors: dict, date: str, boat: str) -> None:
        
        features = [f for f in features if f not in AVOID_IN_COMPARISON_PLOTS]
        num_features = len(features) 

        for i in range(0, num_features, 2):
            # Create a new row with up to two columns
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < num_features:  # Check if there's a feature for this column
                    feature = features[i + j]
                    with col:
                        cls._daily_difference(
                            data,
                            date,
                            boat,
                            ["time", "signal_instance", feature],
                            features,
                            feature_colors,
                        )


    
class AnomalySelectors:
    @classmethod
    def show_selections(cls, data: pd.DataFrame, boat: str) -> None:
        col1, col2 = st.columns([2, 2])    
        with col1:
            dates = data.query(f'node_name == "{boat}"')["date"].unique()
            date = st.selectbox("Select a date", dates, key=f"{boat} at {dates}")
        with col2:
            signal_instance = st.selectbox("Select engine side", ["Starboard", "Port"], key=f"Engine side at {date} for {boat}")
            signal_instance = "SB" if signal_instance == "Starboard" else "P"
        return date, signal_instance
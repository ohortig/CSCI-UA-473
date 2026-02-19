"""NYC map visualizations for Lab 4: Generalization & Data Splits.

Provides choropleth and scatter-map helpers for visualising
data splits and model performance across NYC neighbourhoods.
Uses plotly (no geopandas dependency) with free CARTO tiles.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Inside Airbnb publishes a GeoJSON of neighbourhood boundaries.
GEOJSON_URLS = [
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/visualisations/neighbourhoods.geojson",
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2024-09-04/visualisations/neighbourhoods.geojson",
]
GEOJSON_CACHE = "data/processed/nyc_neighbourhoods.geojson"

NYC_CENTER = {"lat": 40.7128, "lon": -73.95}
NYC_ZOOM = 10
MAP_HEIGHT = 520


# ------------------------------------------------------------------
# GeoJSON loader
# ------------------------------------------------------------------


@st.cache_data(show_spinner="Loading NYC neighbourhood boundaries ...")
def load_neighbourhood_geojson():
    """Download and cache the NYC neighbourhood GeoJSON.

    Returns the parsed GeoJSON dict, or *None* if unavailable.
    """
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE) as f:
            return json.load(f)

    for url in GEOJSON_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            geojson = resp.json()
            os.makedirs(os.path.dirname(GEOJSON_CACHE), exist_ok=True)
            with open(GEOJSON_CACHE, "w") as f:
                json.dump(geojson, f)
            return geojson
        except Exception:
            continue
    return None


# ------------------------------------------------------------------
# Price overview map (Step 1)
# ------------------------------------------------------------------


def display_price_map(df):
    """Scatter map of all listings coloured by nightly price."""
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="price",
        color_continuous_scale="Viridis",
        range_color=[df["price"].quantile(0.05), df["price"].quantile(0.95)],
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=NYC_ZOOM,
        opacity=0.55,
        title="Listing Prices Across NYC",
        hover_data=["name", "neighbourhood_cleansed", "room_type", "price"],
    )
    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Neighbourhood split choropleth (Step 5)
# ------------------------------------------------------------------


def display_split_choropleth(df, train_mask, val_mask, test_mask, geojson):
    """Choropleth showing which neighbourhoods are train / val / test."""
    hoods = df["neighbourhood_cleansed"].values

    rows = []
    for hood in np.unique(hoods):
        if np.any(hoods[train_mask] == hood):
            split = "Train"
        elif np.any(hoods[val_mask] == hood):
            split = "Val"
        elif np.any(hoods[test_mask] == hood):
            split = "Test"
        else:
            continue
        rows.append({"neighbourhood": hood, "Split": split})

    hood_df = pd.DataFrame(rows)

    fig = px.choropleth_mapbox(
        hood_df,
        geojson=geojson,
        locations="neighbourhood",
        featureidkey="properties.neighbourhood",
        color="Split",
        color_discrete_map={"Train": "#636EFA", "Val": "#EF553B", "Test": "#00CC96"},
        category_orders={"Split": ["Train", "Val", "Test"]},
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=NYC_ZOOM,
        opacity=0.65,
        title="Neighbourhood Split Assignment",
    )
    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Error choropleth (neighbourhood-level MAE)
# ------------------------------------------------------------------


def display_error_choropleth(
    df,
    test_mask,
    y_test_actual,
    y_test_pred,
    geojson,
    title="Mean Absolute Error by Neighbourhood (Test Set)",
):
    """Choropleth coloured by neighbourhood-level MAE in dollars."""
    test_df = df.loc[test_mask].copy()
    test_df["abs_error"] = np.abs(np.expm1(y_test_actual) - np.expm1(y_test_pred))

    hood_errors = (
        test_df.groupby("neighbourhood_cleansed")["abs_error"].mean().reset_index()
    )
    hood_errors.columns = ["neighbourhood", "MAE ($)"]

    fig = px.choropleth_mapbox(
        hood_errors,
        geojson=geojson,
        locations="neighbourhood",
        featureidkey="properties.neighbourhood",
        color="MAE ($)",
        color_continuous_scale="RdYlGn_r",
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=NYC_ZOOM,
        opacity=0.7,
        title=title,
    )
    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Error scatter map (listing-level)
# ------------------------------------------------------------------


def display_error_scatter_map(
    df,
    test_mask,
    y_test_actual,
    y_test_pred,
    title="Test Set Prediction Errors",
):
    """Scatter map of test listings coloured by absolute dollar error."""
    test_df = df.loc[test_mask].copy()
    test_df = test_df.reset_index(drop=True)
    test_df["Actual ($)"] = np.expm1(y_test_actual)
    test_df["Predicted ($)"] = np.expm1(y_test_pred)
    test_df["Abs Error ($)"] = np.abs(test_df["Actual ($)"] - test_df["Predicted ($)"])

    fig = px.scatter_mapbox(
        test_df,
        lat="latitude",
        lon="longitude",
        color="Abs Error ($)",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, test_df["Abs Error ($)"].quantile(0.95)],
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=NYC_ZOOM,
        opacity=0.7,
        hover_data=["neighbourhood_cleansed", "Actual ($)", "Predicted ($)"],
        title=title,
    )
    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Comparison: tabbed error maps for all three strategies
# ------------------------------------------------------------------


def display_comparison_maps(df, map_data, geojson):
    """Render error maps for all completed strategies in tabs."""
    labels = [
        ("random", "Random"),
        ("host", "Host-Based"),
        ("neighborhood", "Neighbourhood"),
        ("borough", "Borough"),
    ]
    available = [(k, v) for k, v in labels if k in map_data]
    if not available:
        return

    tabs = st.tabs([label for _, label in available])

    for tab, (key, label) in zip(tabs, available):
        data = map_data[key]
        with tab:
            display_error_scatter_map(
                df,
                data["test_mask"],
                data["test_actuals"],
                data["test_preds"],
                title=f"{label} Split — Test Error Map",
            )
            # Neighbourhood-level choropleth for neighbourhood/borough strategies
            if key in ("neighborhood", "borough") and geojson is not None:
                display_error_choropleth(
                    df,
                    data["test_mask"],
                    data["test_actuals"],
                    data["test_preds"],
                    geojson,
                    title=f"{label} Split — MAE by Neighbourhood",
                )

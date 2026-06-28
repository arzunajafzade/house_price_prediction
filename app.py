from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import MODEL_PATH, train


st.set_page_config(page_title="California House Price Predictor")

FEATURE_DEFAULTS = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}

OCEAN_PROXIMITY_OPTIONS = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]


@st.cache_resource(show_spinner="Training model for the first run...")
def load_model():
    if not Path(MODEL_PATH).exists():
        model, _ = train()
        return model
    return joblib.load(MODEL_PATH)


def build_input() -> pd.DataFrame:
    col_left, col_right = st.columns(2)

    with col_left:
        longitude = st.number_input("Longitude", value=FEATURE_DEFAULTS["longitude"])
        latitude = st.number_input("Latitude", value=FEATURE_DEFAULTS["latitude"])
        housing_median_age = st.number_input(
            "Housing median age",
            min_value=0.0,
            value=FEATURE_DEFAULTS["housing_median_age"],
        )
        total_rooms = st.number_input(
            "Total rooms",
            min_value=0.0,
            value=FEATURE_DEFAULTS["total_rooms"],
        )
        total_bedrooms = st.number_input(
            "Total bedrooms",
            min_value=0.0,
            value=FEATURE_DEFAULTS["total_bedrooms"],
        )

    with col_right:
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=FEATURE_DEFAULTS["population"],
        )
        households = st.number_input(
            "Households",
            min_value=0.0,
            value=FEATURE_DEFAULTS["households"],
        )
        median_income = st.number_input(
            "Median income",
            min_value=0.0,
            value=FEATURE_DEFAULTS["median_income"],
        )
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            OCEAN_PROXIMITY_OPTIONS,
            index=OCEAN_PROXIMITY_OPTIONS.index(FEATURE_DEFAULTS["ocean_proximity"]),
        )

    return pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ]
    )


st.title("California House Price Predictor")
st.caption("Random Forest regression model trained on the California housing dataset.")

model = load_model()
features = build_input()

if st.button("Predict price", type="primary"):
    prediction = model.predict(features)[0]
    st.metric("Estimated median house value", f"${prediction:,.0f}")

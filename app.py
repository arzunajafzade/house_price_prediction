import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
# Note: Ensure 'model.pkl' is in the same directory as this file
try:
    model = joblib.load('model.pkl')
except:
    st.error("Model file 'model.pkl' not found. Please save your model from the notebook first.")

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Prediction")
st.write("Enter the details of the area to estimate the median house value.")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

def get_user_input():
    longitude = st.sidebar.slider("Longitude", -124.35, -114.31, -118.0)
    latitude = st.sidebar.slider("Latitude", 32.54, 41.95, 34.0)
    housing_median_age = st.sidebar.number_input("Housing Median Age", 1, 52, 28)
    total_rooms = st.sidebar.number_input("Total Rooms", 1, 40000, 2000)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 6500, 500)
    population = st.sidebar.number_input("Population", 3, 35000, 1500)
    households = st.sidebar.number_input("Households", 1, 6000, 500)
    median_income = st.sidebar.number_input("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
    
    ocean_proximity = st.sidebar.selectbox("Ocean Proximity", 
                                          ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    # Create a dictionary for the features
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    return pd.DataFrame(data, index=[0])

# Get inputs
input_df = get_user_input()

# Preprocessing (Mirroring your notebook logic)
def preprocess_data(df):
    processed_df = df.copy()
    
    # 1. Log transformations (as seen in your notebook)
    processed_df['total_rooms'] = np.log(processed_df['total_rooms'] + 1)
    processed_df['total_bedrooms'] = np.log(processed_df['total_bedrooms'] + 1)
    processed_df['population'] = np.log(processed_df['population'] + 1)
    processed_df['households'] = np.log(processed_df['households'] + 1)
    
    # 2. One-Hot Encoding for ocean_proximity
    # We must ensure all categories from training are present
    categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for cat in categories:
        processed_df[cat] = 1 if processed_df['ocean_proximity'].iloc[0] == cat else 0
    
    # Drop the original categorical column
    processed_df = processed_df.drop('ocean_proximity', axis=1)
    
    return processed_df

# Display input summary
st.subheader("Selected Area Features")
st.write(input_df)

# Prediction
if st.button("Predict Price"):
    try:
        final_features = preprocess_data(input_df)
        prediction = model.predict(final_features)
        
        st.subheader("Result")
        st.success(f"Estimated Median House Value: **${prediction[0]:,.2f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Tip: Make sure the feature names in app.py match exactly with those used during model.fit()")

st.divider()
st.caption("Based on the California Housing Dataset Analysis.")
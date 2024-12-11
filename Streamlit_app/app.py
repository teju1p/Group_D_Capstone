import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib  # For loading models

# Streamlit app title
st.title("DriveDynamics")

# Load data and models
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

# File paths
base_path = r"C:\Users\kanik\OneDrive\Desktop\capstone\\"
taxi_zones_file = base_path + "taxi_zones_wgs84.csv"
average_speed_h_file = base_path + "df_average_speed_h.csv"
average_speed_m_file = base_path + "df_average_speed_m.csv"
model_peak_file = base_path + "model_peak.joblib"
model_fare_file = base_path + "model_fare_linear.joblib"
model_duration_file = base_path + "model_lgb_duration.joblib"

# Load files
try:
    taxi_zones = load_csv(taxi_zones_file)
    average_speed_h_df = load_csv(average_speed_h_file)
    average_speed_m_df = load_csv(average_speed_m_file)
    model_peak = load_model(model_peak_file)
    model_fare = load_model(model_fare_file)
    model_duration = load_model(model_duration_file)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Utility functions
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlon = lng2 - lng1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def get_coordinates(location_id, zones_df):
    match = zones_df[taxi_zones['LocationID'] == location_id]
    return (match['X'].values[0], match['Y'].values[0]) if not match.empty else (None, None)

def match_average_speed(df, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat, month, day, hour, speed_column):
    matched_row = df[
        (df['pickup_longitude'] == pickup_lon) &
        (df['pickup_latitude'] == pickup_lat) &
        (df['dropoff_longitude'] == dropoff_lon) &
        (df['dropoff_latitude'] == dropoff_lat) &
        (df['Month'] == month) & (df['DayofMonth'] == day) & (df['Hour'] == hour)
    ]
    return matched_row[speed_column].values[0] if not matched_row.empty else 0

# User Inputs
st.header("Enter Trip Details")
pcount = st.number_input("Passenger Count", min_value=1, max_value=10, step=1, value=1)
pid = st.number_input("Pickup Location ID", min_value=1, step=1, value=1)
did = st.number_input("Dropoff Location ID", min_value=1, step=1, value=1)
pickup_hour = st.selectbox("Pickup Hour (0-23)", list(range(24)), index=0)
minutes = st.number_input("Pickup Min", min_value=0, max_value=59, step=1, value=0)
DayofMonth = st.number_input("Day of Month", min_value=1, max_value=31, step=1, value=1)
month = st.selectbox("Month", list(range(1, 13)))

# Compute Features
trip_date = datetime(2024, month, DayofMonth)
day_of_week = trip_date.weekday()

pickup_longitude, pickup_latitude = get_coordinates(pid, taxi_zones)
dropoff_longitude, dropoff_latitude = get_coordinates(did, taxi_zones)

if pickup_longitude and dropoff_longitude:
    distance_haversine = haversine_array(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    distance_dummy_manhattan = dummy_manhattan_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    direction = 0  # Placeholder for direction calculation
    avg_speed_h = match_average_speed(average_speed_h_df, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, month, DayofMonth, pickup_hour, 'avg_speed_h')
    avg_speed_m = match_average_speed(average_speed_m_df, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, month, DayofMonth, pickup_hour, 'avg_speed_m')

    # Peak Hour Prediction
    peak_features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                     'dropoff_latitude', 'distance_haversine', 'distance_dummy_manhattan', 'direction',
                     'Month', 'DayofMonth', 'Hour', 'dayofweek', 'avg_speed_h', 'avg_speed_m']
    
    features_peak = pd.DataFrame({
        'passenger_count': [pcount],
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'distance_haversine': [distance_haversine],
        'distance_dummy_manhattan': [distance_dummy_manhattan],
        'direction': [direction],
        'Month': [month],
        'DayofMonth': [DayofMonth],
        'Hour': [pickup_hour],
        'dayofweek': [day_of_week],
        'avg_speed_h': [avg_speed_h],
        'avg_speed_m': [avg_speed_m]
    })[peak_features]

    peak_hour = model_peak.predict(features_peak)[0]

    # Duration Prediction
    features_duration = features_peak.copy()  # Same features without modification
    duration = model_duration.predict(features_duration)[0]


    # Fare Prediction
    fare_features = ['distance_haversine', 'distance_dummy_manhattan', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude', 'peak_hour', 'Month', 'DayofMonth',
                     'Hour', 'dayofweek', 'average_speed_h', 'average_speed_m']
    
    features_fare = pd.DataFrame({
        'distance_haversine': [distance_haversine],
        'distance_dummy_manhattan': [distance_dummy_manhattan],
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'peak_hour': [peak_hour],
        'Month': [month],
        'DayofMonth': [DayofMonth],
        'Hour': [pickup_hour],
        'dayofweek': [day_of_week],
        'average_speed_h': [avg_speed_h],
        'average_speed_m': [avg_speed_m]
    })[fare_features]

    fare = model_fare.predict(features_fare)[0]

    # Display Results
    st.subheader("Predicted Results")
    st.write(f"**Peak Hour Prediction:** {peak_hour}")
    st.write(f"**Predicted Trip Duration:** {duration:.2f} minutes")
    st.write(f"**Estimated Fare Price:** ${fare:.2f}")
else:
    st.error("Could not find coordinates for the given Pickup or Dropoff Location IDs.")

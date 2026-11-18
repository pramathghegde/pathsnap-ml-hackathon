import folium
import streamlit.components.v1 as components
import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="PathSnap | Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Backend configuration


API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")

# Page heading
st.title("🗺️ PathSnap")
st.subheader("GPS Map Matching & Route Reconstruction")
st.caption("Evaluator Dashboard | Distributed ML System")

st.divider()

# Backend status
st.subheader("Backend Connection")

try:
    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if response.status_code == 200:
        st.success("Backend connected successfully")
    else:
        st.error("Backend returned an unexpected response")

except requests.RequestException:
    st.error("Backend is unavailable. Please start the FastAPI server.")

st.divider()

# Trip selection
st.subheader("Select Evaluation Trip")

trip_ids = [f"{i:08d}" for i in range(10)]

selected_trip = st.selectbox(
    "Choose a trip",
    trip_ids
)

# Prediction button
if st.button("Run Route Prediction", type="primary"):

    with st.spinner("Sending prediction request..."):

        try:
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                json={"trip_id": selected_trip},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("status") == "not_ready":
                    st.warning(
                        "Prediction engine is awaiting "
                        "ML model and HMM integration."
                    )
                else:
                    st.success("Prediction request completed")
                    st.json(result)

            else:
                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.RequestException as error:
            st.error(f"Connection failed: {error}")

st.divider()

# Results section
st.subheader("Route Reconstruction Results")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Trip", selected_trip)
col2.metric("Top-1 Accuracy", "Pending")
col3.metric("Route Continuity", "Pending")

st.info(
    "Interactive Folium route visualisation will be "
    "integrated after the prediction pipeline is ready."
)

st.caption("PathSnap ML Hackathon | Evaluator Interface")
st.divider()

st.subheader("Geographical Route Visualisation")

st.info(
    "The interactive map will display raw GPS observations, "
    "road network and reconstructed route after model integration."
)

# Map preview
map_center = [12.9716, 77.5946]

route_map = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="OpenStreetMap"
)

components.html(
    route_map._repr_html_(),
    height=500
)
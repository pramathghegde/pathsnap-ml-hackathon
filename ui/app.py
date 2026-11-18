
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PathSnap | Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide"
)

API_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# --------------------------------------------------
# Page heading
# --------------------------------------------------

st.title("🗺️ PathSnap")
st.subheader("GPS Map Matching & Route Reconstruction")
st.caption("Evaluator Dashboard | Distributed ML System")

st.divider()

# --------------------------------------------------
# Backend status
# --------------------------------------------------

st.subheader("Backend Connection")

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:
        st.success("Backend connected successfully")
    else:
        st.error("Backend returned an unexpected response")

except requests.RequestException:
    st.error("Backend unavailable. Start the FastAPI server.")

st.caption(f"API endpoint: {API_URL}")

st.divider()

# --------------------------------------------------
# Trip selection
# --------------------------------------------------

st.subheader("Select Evaluation Trip")

selected_trip = st.text_input(
    "Enter an available student-test Trip ID",
    value="00000000",
    max_chars=8,
    help="Enter an eight-digit trip ID from the student_test dataset."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Run Route Prediction", type="primary"):

    if not selected_trip.isdigit() or len(selected_trip) != 8:
        st.error("Please enter a valid eight-digit Trip ID.")

    else:
        with st.spinner("Running ML prediction and HMM reconstruction..."):

            try:
                response = requests.post(
                    f"{API_URL}/api/v1/predict",
                    json={"trip_id": selected_trip},
                    timeout=60
                )

                if response.status_code == 200:
                    st.session_state.prediction_result = response.json()
                    st.success("Route prediction completed successfully!")

                else:
                    st.session_state.prediction_result = None

                    try:
                        error_detail = response.json().get(
                            "detail",
                            response.text
                        )
                    except ValueError:
                        error_detail = response.text

                    st.error(
                        f"API Error ({response.status_code}): {error_detail}"
                    )

            except requests.RequestException as error:
                st.error(f"Connection failed: {error}")

# --------------------------------------------------
# Results
# --------------------------------------------------

result = st.session_state.prediction_result

if result:

    st.divider()
    st.subheader("Route Reconstruction Results")

    metrics = result.get("metrics", {})

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Trip ID",
        result.get("trip_id", "N/A")
    )

    col2.metric(
        "Predicted Route Arcs",
        metrics.get("num_route_arcs", "N/A")
    )

    col3.metric(
        "Route Continuity",
        f"{metrics.get('continuity_score', 0):.1f}%"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "GPS Observations",
        metrics.get("num_observations", "N/A")
    )

    col5.metric(
        "Total Runtime",
        f"{metrics.get('runtime_ms', 0):.2f} ms"
    )

    col6.metric(
        "Viterbi Runtime",
        f"{metrics.get('viterbi_runtime_ms', 0):.2f} ms"
    )

    st.caption(
        f"Gaps filled: {metrics.get('gap_count', 0)} | "
        f"Disconnected transitions: "
        f"{metrics.get('disconnect_count', 0)}"
    )

    st.info(
        "Route continuity measures connected road segments. "
        "It is not the same as ground-truth prediction accuracy."
    )

    # --------------------------------------------------
    # Predicted route map
    # --------------------------------------------------

    st.divider()
    st.subheader("Geographical Route Visualisation")

    geometry = result.get("route_geometry", [])

    if len(geometry) >= 2:

        # API geometry is [longitude, latitude].
        # Folium expects [latitude, longitude].
        route_points = [
            [point[1], point[0]]
            for point in geometry
        ]

        center_lat = sum(p[0] for p in route_points) / len(route_points)
        center_lon = sum(p[1] for p in route_points) / len(route_points)

        route_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        folium.PolyLine(
            route_points,
            color="blue",
            weight=5,
            opacity=0.85,
            tooltip="Predicted PathSnap Route"
        ).add_to(route_map)

        folium.Marker(
            route_points[0],
            tooltip="Route Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(route_map)

        folium.Marker(
            route_points[-1],
            tooltip="Route End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(route_map)

        route_map.fit_bounds(route_points)

        components.html(
            route_map._repr_html_(),
            height=550,
            scrolling=False
        )

        st.success(
            f"Displayed {len(route_points)} route coordinate points."
        )

    else:
        st.warning("The API did not return enough coordinates to draw a route.")

    # --------------------------------------------------
    # Prediction details
    # --------------------------------------------------

    with st.expander("View prediction response"):
        st.json(result)

st.divider()

st.caption(
    "PathSnap ML Hackathon | FastAPI + Streamlit + "
    "HistGradientBoosting + HMM/Viterbi"
)

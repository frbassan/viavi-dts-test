import os
import glob
import json
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="DTS 3D/2D Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Data Parsing & Caching
# ---------------------------------------------------------
@st.cache_data
def parse_single_json(file_content):
    """Parses single DTS JSON string/bytes into arrays."""
    data = json.loads(file_content)
    test_entry = data["tests"][0]
    meas = test_entry["results"]["data"]["measuredResults"][0]
    dt_str = meas.get("acqDateTime", test_entry["results"].get("testTime"))
    dt = datetime.fromisoformat(dt_str)
    pts = meas["dataPoints"]
    return {
        "datetime": dt,
        "positions": np.array(pts["positionM"], dtype=float),
        "temperatures": np.array(pts["temperatureC"], dtype=float),
    }

@st.cache_data
def load_and_grid_data(uploaded_files=None):
    records = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                records.append(parse_single_json(content))
            except Exception as e:
                st.warning(f"Error reading {uploaded_file.name}: {e}")
    else:
        local_files = glob.glob(os.path.join(os.path.dirname(__file__), "*dts.json"))
        if not local_files:
            local_files = glob.glob("*dts.json")
        for fpath in local_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    records.append(parse_single_json(f.read()))
            except Exception as e:
                st.warning(f"Error reading {os.path.basename(fpath)}: {e}")

    if not records:
        return None, None, None

    records.sort(key=lambda r: r["datetime"])
    times = [r["datetime"] for r in records]
    ref_positions = records[0]["positions"]
    
    temp_matrix = []
    for r in records:
        if len(r["positions"]) != len(ref_positions) or not np.allclose(r["positions"], ref_positions):
            t_interp = np.interp(ref_positions, r["positions"], r["temperatures"])
            temp_matrix.append(t_interp)
        else:
            temp_matrix.append(r["temperatures"])

    return times, ref_positions, np.array(temp_matrix)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("Data Controls")
uploaded = st.sidebar.file_uploader(
    "Upload .dts.json files (optional)",
    type=["json"],
    accept_multiple_files=True
)

times, positions, temp_matrix = load_and_grid_data(uploaded)

if times is None or len(times) == 0:
    st.error("No DTS data found. Place `.dts.json` files in the app folder or upload them via the sidebar.")
    st.stop()

time_strs = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
z_min = float(np.min(temp_matrix))
z_max = float(np.max(temp_matrix))

st.sidebar.markdown(f"**Loaded Slices:** {len(times)}")
st.sidebar.markdown(f"**Spatial Points:** {len(positions)}")
st.sidebar.markdown(f"**Distance Range:** {positions[0]:.2f}m – {positions[-1]:.2f}m")

# Time slice selector (Slider)
selected_idx = st.sidebar.slider(
    "Select Time Index",
    min_value=0,
    max_value=len(times) - 1,
    value=0,
    step=1,
    format="%d"
)
selected_time_str = time_strs[selected_idx]
st.sidebar.info(f"Current Slice:\n`{selected_time_str}`")

# ---------------------------------------------------------
# Main Visualizations (1.5x font sizes)
# ---------------------------------------------------------
st.title("Distributed Temperature Sensing (DTS) Explorer")

col1, col2 = st.columns([1.2, 1], gap="medium")

# --- 3D Plot ---
with col1:
    fig_3d = go.Figure(
        data=[
            go.Surface(
                x=positions,
                y=time_strs,
                z=temp_matrix,
                colorscale="Jet",
                cmin=z_min,
                cmax=z_max,
                colorbar=dict(
                    title=dict(text="Temp (°C)", font=dict(size=24, family="Arial")),
                    tickfont=dict(size=20, family="Arial"),
                    len=0.8,
                ),
                hoverinfo="x+y+z",
            )
        ]
    )

    fig_3d.update_layout(
        title=dict(
            text="3D Temperature Evolution",
            font=dict(size=30, family="Arial"),
            x=0.5,
        ),
        font=dict(size=21, family="Arial"),
        scene=dict(
            xaxis=dict(
                title=dict(text="Distance (m)", font=dict(size=21, family="Arial")),
                tickfont=dict(size=18, family="Arial"),
            ),
            yaxis=dict(
                title=dict(text="Timestamp", font=dict(size=21, family="Arial")),
                tickfont=dict(size=17, family="Arial"),
            ),
            zaxis=dict(
                title=dict(text="Temperature (°C)", font=dict(size=21, family="Arial")),
                tickfont=dict(size=18, family="Arial"),
            ),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.3)),
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        height=620,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- 2D Profile Plot ---
with col2:
    fig_2d = go.Figure(
        data=[
            go.Scatter(
                x=positions,
                y=temp_matrix[selected_idx],
                mode="lines",
                line=dict(color="#d62728", width=3.0),
                name="Temperature",
            )
        ]
    )

    fig_2d.update_layout(
        title=dict(
            text=f"Trace Profile @ {selected_time_str}",
            font=dict(size=27, family="Arial"),
            x=0.5,
        ),
        font=dict(size=21, family="Arial"),
        xaxis=dict(
            title=dict(text="Distance (m)", font=dict(size=24, family="Arial")),
            tickfont=dict(size=20, family="Arial"),
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title=dict(text="Temperature (°C)", font=dict(size=24, family="Arial")),
            tickfont=dict(size=20, family="Arial"),
            range=[z_min - 2, z_max + 2],
            gridcolor="#e0e0e0",
        ),
        plot_bgcolor="white",
        margin=dict(l=40, r=20, b=30, t=60),
        height=620,
    )
    st.plotly_chart(fig_2d, use_container_width=True)
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk  
import numpy as np
import os

os.environ["MAPBOX_API_KEY"] = "YOUR MAPBOX API TOKEN"

st.set_page_config(page_title="SuDS Water-Level Digital Twin (PoC)", layout="wide")


def load_cfg(path="twin_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


cfg = load_cfg()
csv_path = cfg.get("csv_path", "data/map16_sensor.csv")
out_dir = cfg.get("outputs_dir", "outputs")
assets_path = cfg.get("assets_path", "data/assets.csv")

LEVEL_UNIT = cfg.get("units", {}).get("water_level", "units")
ROR_UNIT = cfg.get("units", {}).get("rate_of_rise", "units/hr")

st.title("SuDS Water-Level Digital Twin (PoC)")
st.caption("Twin State • Events • Data Health • Map • Event Zoom")


@st.cache_data
def load_raw():
    return pd.read_csv(csv_path)


@st.cache_data
def load_outputs():
    twin_state = pd.read_csv(f"{out_dir}/twin_state.csv")
    health = pd.read_csv(f"{out_dir}/health.csv")
    events = pd.read_csv(f"{out_dir}/events.csv")

    if "latest_time" in twin_state.columns:
        twin_state["latest_time"] = pd.to_datetime(twin_state["latest_time"], errors="coerce")

    for c in ["event_start", "event_end", "peak_time"]:
        if c in events.columns:
            events[c] = pd.to_datetime(events[c], errors="coerce")

    return twin_state, health, events


@st.cache_data
def load_assets():
    try:
        a = pd.read_csv(assets_path)
        if "device_id" in a.columns:
            a["device_id"] = a["device_id"].astype(str)
        for c in ["lat", "lon"]:
            if c in a.columns:
                a[c] = pd.to_numeric(a[c], errors="coerce")
        return a
    except Exception:
        return pd.DataFrame()


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


raw = load_raw()
twin_state, health, events = load_outputs()
assets = load_assets()

# Detect raw timestamp column using config
ts_candidates = cfg.get("timestamp_candidates", ["date_id", "timestamp", "datetime", "time"])
raw_cols_lower = {c.lower(): c for c in raw.columns}


def pick_raw_col(cands):
    for c in cands:
        if c.lower() in raw_cols_lower:
            return raw_cols_lower[c.lower()]
    return None


RAW_TS_COL = pick_raw_col(ts_candidates)
RAW_LEVEL_COL = "water_level" if "water_level" in raw.columns else None

# Detect twin_state columns (support new + old)
STATE_LEVEL_COL = pick_col(twin_state, ["latest_level_mm", "latest_level_m", "latest_level"])
STATE_ROR_COL = pick_col(twin_state, ["rate_of_rise_mm/hr", "rate_of_rise_m/hr", "rate_of_rise_m_per_hr", "rate_of_rise"])
STATE_VOLT_COL = pick_col(twin_state, ["latest_voltage"])
STATE_RSSI_COL = pick_col(twin_state, ["latest_rssi"])
STATE_SIG_COL = pick_col(twin_state, ["signal_status_mode"])

# ----------------------------
# Sidebar selection
# ----------------------------
device_ids = []
if "device_id" in twin_state.columns and not twin_state.empty:
    device_ids = sorted(twin_state["device_id"].astype(str).unique().tolist())
elif "device_id" in raw.columns:
    device_ids = sorted(raw["device_id"].dropna().astype(str).unique().tolist())

st.sidebar.header("Selection")
selected = st.sidebar.selectbox("Sensor / device_id", device_ids if device_ids else ["(none)"])

# ============================
# OVERVIEW ROW: Map + Top Events
# ============================
st.subheader("Overview")
col_map, col_top = st.columns([1, 1])

with col_map:
    st.markdown("### Map View (Assets)")

    if assets.empty or not all(c in assets.columns for c in ["device_id", "lat", "lon"]):
        st.info("Add data/assets.csv with columns: device_id, lat, lon, asset_name(optional), asset_type(optional).")
    else:
        a = assets.dropna(subset=["lat", "lon"]).copy()
        a["device_id"] = a["device_id"].astype(str)

        # Join in latest state + health for tooltips
        ts = twin_state.copy()
        if "device_id" in ts.columns:
            ts["device_id"] = ts["device_id"].astype(str)
        h = health.copy()
        if "device_id" in h.columns:
            h["device_id"] = h["device_id"].astype(str)

        if "device_id" in ts.columns and not ts.empty:
            keep = ["device_id"]
            if STATE_LEVEL_COL: keep.append(STATE_LEVEL_COL)
            if STATE_VOLT_COL: keep.append(STATE_VOLT_COL)
            if STATE_RSSI_COL: keep.append(STATE_RSSI_COL)
            a = a.merge(ts[keep], on="device_id", how="left")

        if "device_id" in h.columns and not h.empty:
            keep_h = ["device_id"]
            for c in ["health_score", "completeness", "flatline", "spikes", "low_voltage"]:
                if c in h.columns:
                    keep_h.append(c)
            a = a.merge(h[keep_h], on="device_id", how="left")

        # Optional: show only sensors that exist in your data
        available = set(device_ids)
        a = a[a["device_id"].isin(available)]

        # Center map around your assets
        mid_lat = float(a["lat"].mean())
        mid_lon = float(a["lon"].mean())

        tooltip = {
            "html": """
            <b>{asset_name}</b><br/>
            device_id: {device_id}<br/>
            type: {asset_type}<br/>
            level: {level}<br/>
            voltage: {voltage}<br/>
            rssi: {rssi}<br/>
            health: {health_score}<br/>
            completeness: {completeness}<br/>
            flatline: {flatline}<br/>
            spikes: {spikes}<br/>
            """,
            "style": {"backgroundColor": "white", "color": "black"},
        }

        # Build a few safe tooltip fields
        a["asset_name"] = a["asset_name"] if "asset_name" in a.columns else ""
        a["asset_type"] = a["asset_type"] if "asset_type" in a.columns else ""

        a["level"] = a[STATE_LEVEL_COL] if (STATE_LEVEL_COL and STATE_LEVEL_COL in a.columns) else np.nan
        a["voltage"] = a[STATE_VOLT_COL] if (STATE_VOLT_COL and STATE_VOLT_COL in a.columns) else np.nan
        a["rssi"] = a[STATE_RSSI_COL] if (STATE_RSSI_COL and STATE_RSSI_COL in a.columns) else np.nan
        a["health_score"] = a["health_score"] if "health_score" in a.columns else np.nan
        a["completeness"] = a["completeness"] if "completeness" in a.columns else np.nan
        a["flatline"] = a["flatline"] if "flatline" in a.columns else np.nan
        a["spikes"] = a["spikes"] if "spikes" in a.columns else np.nan

        # Point size by severity proxy: worse health => larger point (optional)
        a["point_size"] = 60
        if "health_score" in a.columns:
            a["point_size"] = (120 - a["health_score"].fillna(60)).clip(20, 120)

        # Small pins (pixels). You can also scale by health if you want.
        a["point_radius_px"] = 8

        # Optional: scale a little by health (smaller is better)
        if "health_score" in a.columns:
            a["point_radius_px"] = (4 + (100 - a["health_score"].fillna(60)) / 20).clip(4, 14)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=a,
            get_position="[lon, lat]",
            get_radius="point_radius_px",
            radius_units="pixels",
            pickable=True,
            auto_highlight=True,
            opacity=0.8,
        )

        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=14, pitch=0)
        style_name = st.selectbox(
            "Map style",
            ["Light", "Streets", "Outdoors", "Satellite"],
            index=0
        )

        style_map = {
            "Light": "mapbox://styles/mapbox/light-v11",
            "Streets": "mapbox://styles/mapbox/streets-v12",
            "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
            "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
        }
    
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=style_map[style_name])

        st.pydeck_chart(deck, height=520)

        # Show table below map
        show_cols = [c for c in ["device_id", "asset_name", "asset_type", "lat", "lon"] if c in a.columns]
        if STATE_LEVEL_COL and STATE_LEVEL_COL in a.columns:
            show_cols.append(STATE_LEVEL_COL)
        if "health_score" in a.columns:
            show_cols.append("health_score")
        st.dataframe(a[show_cols], width="stretch", height=220)

with col_top:
    st.markdown("### Top Events (All Sensors)")
    if events.empty or "severity" not in events.columns:
        st.info("events.csv is empty or missing severity.")
    else:
        evf = events.copy()

        min_sev = float(evf["severity"].min())
        max_sev = float(evf["severity"].max())
        sev_range = st.slider("Severity range", min_value=min_sev, max_value=max_sev, value=(min_sev, max_sev))
        evf = evf[(evf["severity"] >= sev_range[0]) & (evf["severity"] <= sev_range[1])]

        if "event_start" in evf.columns and evf["event_start"].notna().any():
            ev_min = evf["event_start"].min()
            ev_max = evf["event_start"].max()
            d1, d2 = st.slider(
                "Event start time window",
                min_value=ev_min.to_pydatetime(),
                max_value=ev_max.to_pydatetime(),
                value=(ev_min.to_pydatetime(), ev_max.to_pydatetime()),
            )
            evf = evf[(evf["event_start"] >= pd.Timestamp(d1)) & (evf["event_start"] <= pd.Timestamp(d2))]

        topn = st.number_input("Show Top N events", min_value=5, max_value=200, value=20, step=5)
        ev_top = evf.sort_values("severity", ascending=False).head(int(topn))

        cols = [c for c in ["device_id", "event_start", "event_end", "duration_min", "peak_level", "peak_excess", "max_rise_rate_per_hr", "severity"] if c in ev_top.columns]
        st.dataframe(ev_top[cols], use_container_width=True, height=520)

st.divider()

# ============================
# SELECTED SENSOR PANEL
# ============================
st.subheader(f"Selected Sensor: {selected}")

row_state = twin_state[twin_state["device_id"].astype(str) == str(selected)] if ("device_id" in twin_state.columns and not twin_state.empty) else pd.DataFrame()
row_health = health[health["device_id"].astype(str) == str(selected)] if ("device_id" in health.columns and not health.empty) else pd.DataFrame()

c1, c2, c3, c4, c5, c6 = st.columns(6)

latest_level = float(row_state[STATE_LEVEL_COL].iloc[0]) if (not row_state.empty and STATE_LEVEL_COL) else float("nan")
ror = float(row_state[STATE_ROR_COL].iloc[0]) if (not row_state.empty and STATE_ROR_COL) else float("nan")
score = float(row_health["health_score"].iloc[0]) if (not row_health.empty and "health_score" in row_health.columns) else float("nan")
comp = float(row_health["completeness"].iloc[0]) if (not row_health.empty and "completeness" in row_health.columns) else float("nan")

latest_v = float(row_state[STATE_VOLT_COL].iloc[0]) if (not row_state.empty and STATE_VOLT_COL and STATE_VOLT_COL in row_state.columns) else float("nan")
latest_rssi = float(row_state[STATE_RSSI_COL].iloc[0]) if (not row_state.empty and STATE_RSSI_COL and STATE_RSSI_COL in row_state.columns) else float("nan")

c1.metric("Latest Level", f"{latest_level:.1f} {LEVEL_UNIT}" if latest_level == latest_level else "n/a")
c2.metric("Rate of Rise", f"{ror:.2f} {ROR_UNIT}" if ror == ror else "n/a")
c3.metric("Health", f"{score:.0f}/100" if score == score else "n/a")
c4.metric("Completeness", f"{comp*100:.1f}%" if comp == comp else "n/a")
c5.metric("Voltage", f"{latest_v:.2f} V" if latest_v == latest_v else "n/a")
c6.metric("RSSI", f"{latest_rssi:.0f}" if latest_rssi == latest_rssi else "n/a")

# --- Load + prep raw for plotting
if RAW_TS_COL is None or RAW_LEVEL_COL is None or "device_id" not in raw.columns:
    st.error(f"Raw CSV must include device_id, water_level, and a timestamp column (one of {ts_candidates}).")
    st.stop()

df = raw.copy()
df[RAW_TS_COL] = pd.to_datetime(df[RAW_TS_COL], errors="coerce")
df[RAW_LEVEL_COL] = pd.to_numeric(df[RAW_LEVEL_COL], errors="coerce")
df = df.dropna(subset=[RAW_TS_COL]).sort_values(RAW_TS_COL)

df_sel = df[df["device_id"].astype(str) == str(selected)].dropna(subset=[RAW_LEVEL_COL]).copy()

# --- Events for this device
ev_sel = pd.DataFrame()
if not events.empty and "device_id" in events.columns:
    ev_sel = events[events["device_id"].astype(str) == str(selected)].sort_values("event_start", ascending=False).copy()

# ============================
# Event zoom selection
# ============================
st.markdown("### Event Zoom")
zoom_mode = st.radio("Plot mode", ["Manual time window", "Zoom to selected event"], horizontal=True)

if df_sel.empty:
    st.warning("No data for this device_id in the selected CSV.")
    st.stop()

min_t, max_t = df_sel[RAW_TS_COL].min(), df_sel[RAW_TS_COL].max()

if zoom_mode == "Zoom to selected event" and not ev_sel.empty and "event_start" in ev_sel.columns and "event_end" in ev_sel.columns:
    ev_sel = ev_sel.dropna(subset=["event_start", "event_end"]).copy()
    ev_sel["label"] = ev_sel.apply(
        lambda r: f"{r['event_start']} → {r['event_end']} | sev={float(r.get('severity', float('nan'))):.2f}",
        axis=1,
    )

    chosen_label = st.selectbox("Select an event (acts like click-to-zoom)", ev_sel["label"].tolist())
    chosen = ev_sel[ev_sel["label"] == chosen_label].iloc[0]
    t1 = chosen["event_start"]
    t2 = chosen["event_end"]

    pad_minutes = st.slider("Padding around event (minutes)", 0, 720, 120, step=30)
    t1 = t1 - pd.Timedelta(minutes=pad_minutes)
    t2 = t2 + pd.Timedelta(minutes=pad_minutes)

    if t1 < min_t:
        t1 = min_t
    if t2 > max_t:
        t2 = max_t
else:
    t1, t2 = st.slider(
        "Time window (history)",
        min_value=min_t.to_pydatetime(),
        max_value=max_t.to_pydatetime(),
        value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
    )
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)

# ============================
# Plot
# ============================
st.markdown("### Water Level History")
plot_df = df_sel[(df_sel[RAW_TS_COL] >= t1) & (df_sel[RAW_TS_COL] <= t2)].copy()

fig = plt.figure()
plt.plot(plot_df[RAW_TS_COL], plot_df[RAW_LEVEL_COL])
plt.xlabel("Time")
plt.ylabel(f"Water level ({LEVEL_UNIT})")
st.pyplot(fig, clear_figure=True)

# ============================
# Tables
# ============================
st.markdown("### Detected Events (Selected Sensor)")
if not ev_sel.empty:
    show_cols = [c for c in ["event_start", "event_end", "duration_min", "peak_level", "peak_excess", "max_rise_rate_per_hr", "severity"] if c in ev_sel.columns]
    st.dataframe(ev_sel[show_cols], use_container_width=True, height=260)
else:
    st.info("No events for this sensor.")

st.markdown("### Data Health Details (Selected Sensor)")
if not row_health.empty:
    st.dataframe(row_health, use_container_width=True)
else:
    st.info("No health record for this sensor.")

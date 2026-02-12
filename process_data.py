import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# REQUIRED
# =========================
DEVICE_COL = "device_id"
LEVEL_COL = "water_level"

# Accept both timestamp column names:
TS_CANDIDATES = ["date_id", "timestamp", "datetime", "time"]

# Optional columns
RSSI_COL = "rssi"
VOLT_COL = "voltage"
SIG_COL = "signal_status"
MEASURE_COL = "measure"

LEVEL_UNIT = "mm"     # keep native units
RATE_UNIT = "mm/hr"   # rate-of-rise

def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan

def compute_health(
    g: pd.DataFrame,
    ts_col: str,
    stale_minutes: int,
    flatline_minutes: int,
    max_jump_units: float,
    low_voltage_threshold: float,
) -> dict:
    g = g.sort_values(ts_col)
    n = len(g)
    missing = g[LEVEL_COL].isna().sum()
    completeness = 0.0 if n == 0 else 1.0 - (missing / n)

    # Flatline: no change in last flatline window
    flat = False
    tail = g[[ts_col, LEVEL_COL]].dropna().set_index(ts_col)[LEVEL_COL].sort_index()
    if len(tail) >= 3:
        t_end = tail.index.max()
        t_start = t_end - pd.Timedelta(minutes=flatline_minutes)
        window = tail.loc[tail.index >= t_start]
        if len(window) >= 3:
            flat = float(window.max() - window.min()) < 1e-6

    # Spike count
    spikes = 0
    if len(tail) >= 2:
        diffs = tail.diff().abs()
        spikes = int((diffs > max_jump_units).sum())

    # Stale (PoC compares to dataset end => effectively False)
    stale = False

    # Voltage
    latest_voltage = np.nan
    low_voltage = False
    if VOLT_COL in g.columns:
        vv = g[[ts_col, VOLT_COL]].dropna().sort_values(ts_col)[VOLT_COL]
        if not vv.empty:
            latest_voltage = float(vv.iloc[-1])
            low_voltage = latest_voltage < low_voltage_threshold

    score = 100.0
    score -= (1.0 - completeness) * 50.0
    if flat:
        score -= 25.0
    if spikes > 0:
        score -= min(25.0, spikes * 5.0)
    if low_voltage:
        score -= 15.0
    if stale:
        score -= 25.0
    score = float(max(0.0, min(100.0, score)))

    return {
        "records": int(n),
        "completeness": float(completeness),
        "flatline": bool(flat),
        "spikes": int(spikes),
        "stale": bool(stale),
        "latest_voltage": latest_voltage,
        "low_voltage": bool(low_voltage),
        "health_score": score,
    }

def detect_events(
    ts: pd.Series,
    level: pd.Series,
    baseline_hours: int,
    start_threshold_units: float,
    end_threshold_units: float,
    min_event_minutes: int,
    min_gap_minutes: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"ts": ts, "level": level}).dropna().sort_values("ts").set_index("ts")

    cols = [
        "event_start", "event_end", "duration_min",
        "peak_level", "peak_time", "peak_excess",
        "max_rise_rate_per_hr", "severity"
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Use lowercase 'h' to avoid pandas FutureWarning
    window = f"{baseline_hours}h"
    baseline = df["level"].rolling(window=window, min_periods=max(3, baseline_hours)).median()
    excess = df["level"] - baseline

    in_event = False
    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start_t: Optional[pd.Timestamp] = None

    for t, ex in excess.items():
        if (not in_event) and pd.notna(ex) and ex >= start_threshold_units:
            in_event = True
            start_t = t
        elif in_event and pd.notna(ex) and ex <= end_threshold_units:
            in_event = False
            if start_t is not None:
                events.append((start_t, t))
            start_t = None

    if in_event and start_t is not None:
        events.append((start_t, df.index.max()))

    if not events:
        return pd.DataFrame(columns=cols)

    # merge gaps
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    gap = pd.Timedelta(minutes=min_gap_minutes)
    cur_s, cur_e = events[0]
    for s, e in events[1:]:
        if s - cur_e <= gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    rows = []
    for s, e in merged:
        dur_min = (e - s).total_seconds() / 60.0
        if dur_min < min_event_minutes:
            continue

        seg = df.loc[(df.index >= s) & (df.index <= e)]
        seg_base = baseline.loc[seg.index]
        seg_ex = (seg["level"] - seg_base).dropna()
        if seg.empty or seg_ex.empty:
            continue

        peak_level = float(seg["level"].max())
        peak_time = seg["level"].idxmax()
        peak_excess = float(seg_ex.max())

        dt_hr = seg.index.to_series().diff().dt.total_seconds() / 3600.0
        dl = seg["level"].diff()
        rise_rate = (dl / dt_hr).replace([np.inf, -np.inf], np.nan)
        max_rise_rate = float(rise_rate.max(skipna=True)) if rise_rate.notna().any() else 0.0

        severity = (peak_excess * 0.7) + (max(0.0, max_rise_rate) * 0.3)

        rows.append({
            "event_start": s,
            "event_end": e,
            "duration_min": float(dur_min),
            "peak_level": peak_level,
            "peak_time": peak_time,
            "peak_excess": peak_excess,
            "max_rise_rate_per_hr": max_rise_rate,
            "severity": float(severity),
        })

    return pd.DataFrame(rows, columns=cols)

def build_twin_state(df: pd.DataFrame, ts_col: str, resample_minutes: int) -> pd.DataFrame:
    use_cols = [ts_col, DEVICE_COL, LEVEL_COL]
    for c in [RSSI_COL, VOLT_COL, SIG_COL, MEASURE_COL]:
        if c in df.columns:
            use_cols.append(c)

    df = df[use_cols].copy()
    df[ts_col] = _safe_to_datetime(df[ts_col])
    df[LEVEL_COL] = _ensure_numeric(df[LEVEL_COL])
    if RSSI_COL in df.columns:
        df[RSSI_COL] = _ensure_numeric(df[RSSI_COL])
    if VOLT_COL in df.columns:
        df[VOLT_COL] = _ensure_numeric(df[VOLT_COL])

    df = df.dropna(subset=[ts_col, DEVICE_COL]).sort_values(ts_col)

    states = []
    for dev, g in df.groupby(DEVICE_COL):
        g = g.sort_values(ts_col).set_index(ts_col)

        rs = g.resample(f"{resample_minutes}min").median(numeric_only=True)
        if rs.empty or rs[LEVEL_COL].dropna().empty:
            continue

        latest_ts = rs.index.max()
        latest_level = float(rs[LEVEL_COL].iloc[-1])

        ror = 0.0
        if len(rs) >= 2 and pd.notna(rs[LEVEL_COL].iloc[-2]):
            dt_hr = (rs.index[-1] - rs.index[-2]).total_seconds() / 3600.0
            dl = rs[LEVEL_COL].iloc[-1] - rs[LEVEL_COL].iloc[-2]
            ror = float(dl / dt_hr) if dt_hr > 0 else 0.0

        # latest signal metrics from raw
        tail = g.reset_index().sort_values(ts_col)

        latest_voltage = float(tail[VOLT_COL].dropna().iloc[-1]) if VOLT_COL in tail.columns and tail[VOLT_COL].dropna().any() else np.nan
        latest_rssi = float(tail[RSSI_COL].dropna().iloc[-1]) if RSSI_COL in tail.columns and tail[RSSI_COL].dropna().any() else np.nan
        sig_mode = _mode_or_nan(tail[SIG_COL]) if SIG_COL in tail.columns else np.nan

        states.append({
            "device_id": dev,
            "latest_time": latest_ts,
            f"latest_level_{LEVEL_UNIT}": latest_level,
            f"rate_of_rise_{RATE_UNIT}": ror,
            "latest_voltage": latest_voltage,
            "latest_rssi": latest_rssi,
            "signal_status_mode": sig_mode,
        })

    return pd.DataFrame(states).sort_values("device_id")

def main():
    cfg = _load_config("twin_config.json")
    csv_path = cfg.get("csv_path", "data/map16_sensor.csv")
    out_dir = cfg.get("outputs_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    resample_minutes = int(cfg.get("resample_minutes", 5))

    ev = cfg.get("event_detection", {})
    baseline_hours = int(ev.get("baseline_hours", 48))
    start_threshold = float(ev.get("start_threshold_units", 30))  # mm
    end_threshold = float(ev.get("end_threshold_units", 10))      # mm
    min_event_minutes = int(ev.get("min_event_minutes", 30))
    min_gap_minutes = int(ev.get("min_gap_minutes", 20))

    hcfg = cfg.get("health", {})
    stale_minutes = int(hcfg.get("stale_minutes", 60))
    flatline_minutes = int(hcfg.get("flatline_minutes", 60))
    max_jump = float(hcfg.get("max_jump_units", 200))             # mm
    low_voltage_threshold = float(hcfg.get("low_voltage_threshold", 3.3))

    df = pd.read_csv(csv_path)

    # choose timestamp column
    ts_col = _pick_first_existing(df, TS_CANDIDATES)
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Tried {TS_CANDIDATES}. Found: {list(df.columns)}")

    # validate required cols
    for c in [DEVICE_COL, LEVEL_COL]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column '{c}'. Found: {list(df.columns)}")

    # parse types
    df[ts_col] = _safe_to_datetime(df[ts_col])
    df[LEVEL_COL] = _ensure_numeric(df[LEVEL_COL])
    if VOLT_COL in df.columns:
        df[VOLT_COL] = _ensure_numeric(df[VOLT_COL])
    if RSSI_COL in df.columns:
        df[RSSI_COL] = _ensure_numeric(df[RSSI_COL])

    df = df.dropna(subset=[ts_col, DEVICE_COL]).sort_values(ts_col)

    # Twin state
    twin_state = build_twin_state(df, ts_col=ts_col, resample_minutes=resample_minutes)
    twin_state.to_csv(os.path.join(out_dir, "twin_state.csv"), index=False)

    # Health
    health_rows = []
    base_cols = [ts_col, LEVEL_COL]
    if VOLT_COL in df.columns:
        base_cols.append(VOLT_COL)

    for dev, g in df.groupby(DEVICE_COL):
        g2 = g[base_cols].copy()
        h = compute_health(
            g2,
            ts_col=ts_col,
            stale_minutes=stale_minutes,
            flatline_minutes=flatline_minutes,
            max_jump_units=max_jump,
            low_voltage_threshold=low_voltage_threshold,
        )
        h["device_id"] = dev
        health_rows.append(h)
    health = pd.DataFrame(health_rows).sort_values("device_id")
    health.to_csv(os.path.join(out_dir, "health.csv"), index=False)

    # Events
    event_frames = []
    for dev, g in df.groupby(DEVICE_COL):
        g = g[[ts_col, LEVEL_COL]].dropna().sort_values(ts_col).set_index(ts_col)
        series = g[[LEVEL_COL]].resample(f"{resample_minutes}min").median()
        events = detect_events(
            ts=series.index.to_series(),
            level=series[LEVEL_COL],
            baseline_hours=baseline_hours,
            start_threshold_units=start_threshold,
            end_threshold_units=end_threshold,
            min_event_minutes=min_event_minutes,
            min_gap_minutes=min_gap_minutes,
        )
        if not events.empty:
            events.insert(0, "device_id", dev)
            event_frames.append(events)

    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(
        columns=["device_id","event_start","event_end","duration_min","peak_level","peak_time","peak_excess","max_rise_rate_per_hr","severity"]
    )
    events_all.to_csv(os.path.join(out_dir, "events.csv"), index=False)

    print("✅ Outputs generated:")
    print(f"- {os.path.join(out_dir, 'twin_state.csv')}")
    print(f"- {os.path.join(out_dir, 'health.csv')}")
    print(f"- {os.path.join(out_dir, 'events.csv')}")
    print(f"\nUsing timestamp column: {ts_col}")
    print(f"Water level treated as native units ({LEVEL_UNIT}); rate-of-rise is {RATE_UNIT}.")
    print("Tune thresholds in twin_config.json if needed.")

if __name__ == "__main__":
    main()
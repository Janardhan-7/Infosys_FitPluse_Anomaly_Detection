# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import io
import os

st.set_page_config(page_title="Fitness Data Preprocessor", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ Fitness Tracker")
target_freq = st.sidebar.selectbox("C: Target Frequency:", ["1min", "5min", "15min", "30min", "1H"])
fill_method = st.sidebar.selectbox("C: Missing Value Fill Method:", ["interpolate", "ffill", "bfill", "mean", "drop"])
use_sample = st.sidebar.checkbox("Use Sample Data\n(demonstrates A+B+C)", value=True)

# Screenshot files from session (developer-provided)
screenshot_1 = "/mnt/data/Screenshot 2025-11-22 204902.png"
screenshot_2 = "/mnt/data/Screenshot 2025-11-22 204917.png"

# ---------- Utility helpers ----------
def detect_timestamp_col(df):
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    if ts_candidates:
        return ts_candidates[0]
    # fallback: any datetime-like column
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c].dropna().iloc[:5])
            return c
        except Exception:
            continue
    return None

def generate_sample_data():
    # produce ~8 hours of per-second HR data and per-minute others
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=4)
    # heart_rate sub-minute (every 30 seconds)
    hr_times = pd.date_range(start, periods=481, freq="30S")
    heart_rate = np.clip(60 + 10*np.sin(np.linspace(0, 10, len(hr_times))) + np.random.normal(0, 3, len(hr_times)), 45, 160)
    hr_df = pd.DataFrame({"timestamp": hr_times, "heart_rate": heart_rate})

    # steps - sparse events across minutes
    steps_times = pd.date_range(start, periods=6, freq="15min")
    steps = np.random.randint(0, 200, len(steps_times))
    steps_df = pd.DataFrame({"timestamp": steps_times, "steps": steps})

    # stress (per-minute)
    stress_times = pd.date_range(start, periods=241, freq="1min")
    stress = np.clip(10 + 5*np.sin(np.linspace(0, 6, len(stress_times))) + np.random.normal(0, 1.2, len(stress_times)), 0, 40)
    stress_df = pd.DataFrame({"timestamp": stress_times, "stress": stress})

    # spo2 (per-minute)
    spo2 = np.clip(97 + np.random.normal(0, 0.8, len(stress_times)), 90, 100)
    spo2_df = pd.DataFrame({"timestamp": stress_times, "spo2": spo2})

    # sleep (label values 0/1 sparse)
    sleep = np.zeros(len(stress_times))
    sleep[len(sleep)//3:len(sleep)//3 + 40] = 1
    sleep_df = pd.DataFrame({"timestamp": stress_times, "sleep": sleep})

    # combine by concatenation (different frequencies)
    df = pd.concat([hr_df, steps_df, stress_df, spo2_df, sleep_df], ignore_index=True, sort=False)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows to mimic raw upload
    return df

def load_user_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    else:
        # try to read as csv
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_json(uploaded_file)

def validation_report(df_raw, ts_col):
    report = {}
    report["original_rows"] = len(df_raw)
    # drop rows with invalid timestamps
    df = df_raw.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    invalid_ts = df[ts_col].isna().sum()
    df = df.dropna(subset=[ts_col]).copy()
    report["rows_removed_invalid_ts"] = int(invalid_ts)
    # missing value handling count (before cleaning)
    report["missing_values_before"] = int(df.isna().sum().sum())

    # Outlier detection: IQR on numeric columns
    numeric = df.select_dtypes(include=[np.number]).copy()
    outlier_counts = {}
    total_outliers = 0
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or numeric[col].dropna().empty:
            outlier_counts[col] = 0
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (numeric[col] < low) | (numeric[col] > high)
        count = int(mask.sum())
        outlier_counts[col] = count
        total_outliers += count

    report["outliers_flagged_total"] = int(total_outliers)
    report["outliers_per_column"] = outlier_counts
    report["missing_values_after"] = int(df.isna().sum().sum())  # unchanged yet
    return report

def apply_time_alignment(df, ts_col, freq_str="1min", fill_method="interpolate"):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.set_index(ts_col)

    # original frequency estimate
    if len(df.index) < 3:
        orig_freq = "unknown"
    else:
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
        median = diffs.median()
        if median < 60:
            orig_freq = "sub_minute"
        elif median == 60:
            orig_freq = "1min"
        else:
            orig_freq = f"{int(median)}s"

    original_rows = len(df)
    # resample
    rule = freq_str
    resampled = df.resample(rule).mean()

    gaps_before = int(resampled.isna().any(axis=1).sum())

    # Fill method
    if fill_method == "interpolate":
        resampled = resampled.interpolate(limit_direction="both")
    elif fill_method == "ffill":
        resampled = resampled.ffill()
    elif fill_method == "bfill":
        resampled = resampled.bfill()
    elif fill_method == "mean":
        resampled = resampled.fillna(resampled.mean())
    elif fill_method == "drop":
        resampled = resampled.dropna(how="any")
    else:
        resampled = resampled.ffill()

    gaps_filled = max(0, gaps_before - int(resampled.isna().any(axis=1).sum()))
    resampled_rows = len(resampled)

    report = {
        "original_frequency": orig_freq,
        "target_frequency": freq_str,
        "original_rows": int(original_rows),
        "resampled_rows": int(resampled_rows),
        "gaps_before": int(gaps_before),
        "gaps_filled": int(gaps_filled),
        "fill_method": fill_method,
        "status": "Success" if resampled_rows > 0 else "Failed",
    }
    return resampled, report

# ---------- Component A: Import ----------
st.markdown("## 🅐 COMPONENT A: Import")
colA1, colA2 = st.columns([3,1])
with colA1:
    uploaded_file = st.file_uploader("Upload CSV or JSON (contains timestamp, heart_rate, steps, sleep, stress, spo2)", type=["csv","json"])
    st.caption("If you don't have a file, check 'Use Sample Data' in the sidebar.")
with colA2:
    st.write("")  # spacing

if use_sample:
    df_raw = generate_sample_data()
    st.success("Sample data loaded (demonstration).")
else:
    if uploaded_file:
        df_raw = load_user_file(uploaded_file)
    else:
        df_raw = pd.DataFrame()  # empty

st.write("### Raw Data (first 10 rows)")
if not df_raw.empty:
    st.dataframe(df_raw.head(10))
else:
    st.info("No data loaded yet. Upload a file or enable sample data.")
# ================================================================
# COMPONENT B : VALIDATION & CLEANING (DYNAMIC DATASET SELECTION)
# ================================================================

st.markdown("---")
st.header("🅱 COMPONENT B: Validating and cleaning data...")

# Ensure raw data exists
if df_raw.empty:
    st.warning("No data available for validation. Upload a file or enable sample data.")
else:

    # --- Split raw data into individual signals ---
    # Create dataset-specific dataframes (filtered from df_raw)
    hr_df    = df_raw[['timestamp', 'heart_rate']].dropna(subset=['heart_rate'])     if 'heart_rate' in df_raw else pd.DataFrame()
    steps_df = df_raw[['timestamp', 'steps']].dropna(subset=['steps'])               if 'steps' in df_raw else pd.DataFrame()
    stress_df= df_raw[['timestamp', 'stress']].dropna(subset=['stress'])             if 'stress' in df_raw else pd.DataFrame()
    spo2_df  = df_raw[['timestamp', 'spo2']].dropna(subset=['spo2'])                 if 'spo2' in df_raw else pd.DataFrame()
    sleep_df = df_raw[['timestamp', 'sleep']].dropna(subset=['sleep'])               if 'sleep' in df_raw else pd.DataFrame()

    # --- Dataset selection dropdown ---
    dataset_choice = st.selectbox(
        "Select dataset for validation report:",
        ["Heart Rate", "Steps", "Stress", "SpO2", "Sleep"]
    )

    # Map dataset names to dataframes
    dataset_map = {
        "Heart Rate": hr_df,
        "Steps": steps_df,
        "Stress": stress_df,
        "SpO2": spo2_df,
        "Sleep": sleep_df
    }

    selected_df = dataset_map[dataset_choice].copy()

    if selected_df.empty:
        st.error(f"No data found for {dataset_choice}.")
    else:
        st.subheader(f"📄 {dataset_choice} Validation Results")
        st.write("📘 **DATA VALIDATION REPORT**")
        st.write("===========================================")

        # Count original rows
        original_rows = len(selected_df)

        # 1️⃣ FIX TIMESTAMP
        selected_df['timestamp'] = pd.to_datetime(selected_df['timestamp'], errors='coerce')
        selected_df = selected_df.dropna(subset=['timestamp'])

        # 2️⃣ MISSING VALUES
        missing_vals = selected_df.isna().sum().sum()

        # 3️⃣ INTERPOLATE missing values
        selected_df = selected_df.interpolate(method='linear')

        # 4️⃣ OUTLIER COUNTING (simple rule-based)
        col = selected_df.columns[1]   # second column = measurement

        if dataset_choice == "Heart Rate":
            outliers = (selected_df[col] > 220).sum()
        elif dataset_choice == "Steps":
            outliers = (selected_df[col] > 20000).sum()
        elif dataset_choice == "Stress":
            outliers = (selected_df[col] > 100).sum()
        elif dataset_choice == "SpO2":
            outliers = (selected_df[col] > 100).sum()
        elif dataset_choice == "Sleep":
            outliers = (selected_df[col] > 1).sum()
        else:
            outliers = 0

        # 5️⃣ Final rows
        final_rows = len(selected_df)

        # Output report
        st.write(f"Original rows: {original_rows}")
        st.write(f"Final rows: {final_rows}")
        st.write(f"Rows removed (bad timestamps): {original_rows - final_rows}")
        st.write(f"Missing values handled: {missing_vals}")
        st.write(f"Outliers flagged: {outliers}")
        st.write("Issues Found:")
        st.write(f"• Total missing values: {missing_vals}")
        st.write("===========================================")

        # Preview cleaned data
        with st.expander("Preview cleaned data"):
            st.dataframe(selected_df.head())

        # Update the corresponding dataset so Component-C uses cleaned data
        if dataset_choice == "Heart Rate":
            hr_df = selected_df
        elif dataset_choice == "Steps":
            steps_df = selected_df
        elif dataset_choice == "Stress":
            stress_df = selected_df
        elif dataset_choice == "SpO2":
            spo2_df = selected_df
        elif dataset_choice == "Sleep":
            sleep_df = selected_df

# ---------- Component C: Time Alignment ----------
st.markdown("---")
st.markdown("## 🅒 COMPONENT C: Time Alignment & Resampling")

if df_raw.empty:
    st.warning("No data to align.")
else:
    ts_col = detect_timestamp_col(df_raw)
    if ts_col is None:
        st.error("Timestamp column not found, cannot align.")
    else:
        resampled_df, ta_report = apply_time_alignment(df_raw.copy(), ts_col, freq_str=target_freq, fill_method=fill_method)
        st.subheader("⏱️ Time Alignment Results")
        st.markdown("#### 🕒 TIME ALIGNMENT REPORT")
        st.text("=================================")
        st.write(f"Original frequency: {ta_report['original_frequency']}")
        st.write(f"Target frequency: {ta_report['target_frequency']}")
        st.write(f"Original rows: {ta_report['original_rows']}")
        st.write(f"Resampled rows: {ta_report['resampled_rows']}")
        st.write(f"Gaps filled: {ta_report['gaps_filled']}")
        st.write(f"Fill method: {ta_report['fill_method']}")
        st.write("")
        st.write("Status: ", "✅ " + ta_report["status"] if ta_report["status"] == "Success" else "❌ " + ta_report["status"])

        st.write("### Cleaned & Time-Aligned Data (first 10 rows)")
        st.dataframe(resampled_df.reset_index().head(10))

# ---------- Dashboard Visualizations ----------
st.markdown("---")
st.markdown("## 📈 Dashboard")

if df_raw.empty:
    st.info("No data to visualize.")
else:
    # Use the resampled_df if it contains relevant columns, else try to build a merged dataframe by resampling each signal separately
    df_vis = resampled_df.copy() if 'resampled_df' in locals() and not resampled_df.empty else None

    # If resampled df missing certain columns, try to create a per-minute combined view
    if df_vis is None or df_vis.empty:
        # attempt to build a minute-based index from earliest to latest timestamp
        try:
            ts_col = detect_timestamp_col(df_raw)
            df_raw[ts_col] = pd.to_datetime(df_raw[ts_col], errors="coerce")
            idx = pd.date_range(df_raw[ts_col].min().floor('T'), df_raw[ts_col].max().ceil('T'), freq=target_freq)
            df_vis = pd.DataFrame(index=idx)
            # merge numeric signals by resampling original
            for col in ['heart_rate','steps','stress','spo2','sleep']:
                if col in df_raw.columns:
                    tmp = df_raw.set_index(ts_col)[col].resample(target_freq).mean()
                    df_vis[col] = tmp
            df_vis = df_vis.interpolate(limit_direction='both')
        except Exception:
            df_vis = df_raw.set_index(ts_col).resample(target_freq).mean().interpolate(limit_direction='both')

    # Heart Rate
    if 'heart_rate' in df_vis.columns:
        fig_hr = px.line(df_vis, y='heart_rate', title="Heart Rate", labels={"index":"time","heart_rate":"BPM"})
        st.plotly_chart(fig_hr, use_container_width=True)

    # Steps (bar)
    if 'steps' in df_vis.columns:
        fig_steps = px.bar(df_vis, y='steps', title="Steps (per interval)", labels={"index":"time","steps":"steps"})
        st.plotly_chart(fig_steps, use_container_width=True)

    # Stress (area)
    if 'stress' in df_vis.columns:
        fig_stress = px.area(df_vis, y='stress', title="Stress Level", labels={"index":"time","stress":"stress"})
        st.plotly_chart(fig_stress, use_container_width=True)

    # Spo2
    if 'spo2' in df_vis.columns:
        fig_spo2 = px.line(df_vis, y='spo2', title="Blood Oxygen (SpO₂)", labels={"index":"time","spo2":"SpO₂ (%)"})
        st.plotly_chart(fig_spo2, use_container_width=True)

    # Sleep scatter / heat
    if 'sleep' in df_vis.columns:
        fig_sleep = px.scatter(df_vis.reset_index(), x=df_vis.reset_index().columns[0], y='sleep', title="Sleep Pattern (0=awake,1=asleep)")
        st.plotly_chart(fig_sleep, use_container_width=True)

st.markdown("---")
st.caption("Pipeline: Component A (Import) → Component B (Validation/Cleaning) → Component C (Time Alignment/Resample).")


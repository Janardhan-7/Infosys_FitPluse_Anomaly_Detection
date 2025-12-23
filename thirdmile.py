# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict
import warnings
import io
import json

warnings.filterwarnings("ignore")

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="FitPulse Milestone 3 - Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Custom CSS (keeps your style)
# -------------------------
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(to bottom right, #FEF3C7, #FDE68A, #FCA5A5);
    }
    h1 {
        color: #dc2626;
        font-weight: 700;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    .anomaly-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #ef4444;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Timestamp helper (fix tz-aware vs tz-naive)
# -------------------------
def fix_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize timestamp column:
    - parse to datetime (utc=True),
    - convert tz-aware -> tz-naive by removing tz info,
    - drop rows with NaT timestamps.
    """
    if df is None or df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Convert tz-aware -> tz-naive
        try:
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        except Exception:
            # If dt.tz_convert fails (already naive), use tz_localize(None) on tz-aware
            try:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            except Exception:
                # Last resort: remove tz info by astype if possible
                pass
        # Drop rows with invalid timestamps
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df

# -------------------------
# Detector Classes (your pattern)
# -------------------------
class RuleBasedAnomalyDetector:
    """Detect anomalies using rule-based thresholds + statistics"""

    def __init__(self):
        self.thresholds = {
            "heart_rate": {"min": 40, "max": 180, "std_multiplier": 3},
            "stress": {"min": 0, "max": 100, "std_multiplier": 2.5},
            "spo2": {"min": 90, "max": 100, "std_multiplier": 2},
            "steps": {"min": 0, "max": 500, "std_multiplier": 3},
            "sleep": {"min": 0, "max": 60, "std_multiplier": 3},
        }
        self.anomalies = {}

    def detect_threshold_anomalies(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        metric_columns = {
            "heart_rate": "heart_rate",
            "stress": "stress_level",
            "spo2": "spo2_level",
            "steps": "step_count",
            "sleep": "duration_minutes",
        }
        if data_type not in metric_columns:
            return pd.DataFrame()
        metric_col = metric_columns[data_type]
        if metric_col not in df.columns:
            return pd.DataFrame()
        thresholds = self.thresholds.get(data_type, {})
        min_val = thresholds.get("min", -np.inf)
        max_val = thresholds.get("max", np.inf)
        anomalies_df = df[(df[metric_col] < min_val) | (df[metric_col] > max_val)].copy()
        if not anomalies_df.empty:
            anomalies_df["anomaly_type"] = "threshold"
            anomalies_df["severity"] = anomalies_df[metric_col].apply(
                lambda x: "high"
                if abs(x - (min_val + max_val) / 2) > (max_val - min_val) / 2
                else "medium"
            )
            anomalies_df["reason"] = anomalies_df[metric_col].apply(lambda x: f"Value {x:.2f} outside range [{min_val}, {max_val}]")
        return anomalies_df

    def detect_statistical_anomalies(self, df: pd.DataFrame, data_type: str, z_threshold: float = 3.0) -> pd.DataFrame:
        metric_columns = {
            "heart_rate": "heart_rate",
            "stress": "stress_level",
            "spo2": "spo2_level",
            "steps": "step_count",
            "sleep": "duration_minutes",
        }
        if data_type not in metric_columns:
            return pd.DataFrame()
        metric_col = metric_columns[data_type]
        if metric_col not in df.columns:
            return pd.DataFrame()
        if df[metric_col].std() == 0 or len(df[metric_col]) < 3:
            return pd.DataFrame()
        z_scores = np.abs(stats.zscore(df[metric_col].fillna(df[metric_col].mean())))
        Q1 = df[metric_col].quantile(0.25)
        Q3 = df[metric_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies_df = df[(z_scores > z_threshold) | (df[metric_col] < lower_bound) | (df[metric_col] > upper_bound)].copy()
        if not anomalies_df.empty:
            anomalies_df["anomaly_type"] = "statistical"
            anomalies_df["z_score"] = z_scores[anomalies_df.index]
            anomalies_df["severity"] = anomalies_df["z_score"].apply(lambda x: "high" if x > 4 else "medium" if x > 3 else "low")
            anomalies_df["reason"] = "Statistical outlier (Z-score or IQR)"
        return anomalies_df

    def detect_all_rule_based(self, df: pd.DataFrame, data_type: str, z_threshold: float = 3.0) -> pd.DataFrame:
        threshold_anomalies = self.detect_threshold_anomalies(df, data_type)
        statistical_anomalies = self.detect_statistical_anomalies(df, data_type, z_threshold)
        if not threshold_anomalies.empty and not statistical_anomalies.empty:
            all_anomalies = pd.concat([threshold_anomalies, statistical_anomalies], ignore_index=True)
        elif not threshold_anomalies.empty:
            all_anomalies = threshold_anomalies.copy()
        elif not statistical_anomalies.empty:
            all_anomalies = statistical_anomalies.copy()
        else:
            all_anomalies = pd.DataFrame()
        if not all_anomalies.empty and "timestamp" in all_anomalies.columns:
            all_anomalies = all_anomalies.drop_duplicates(subset=["timestamp"])
        self.anomalies[data_type] = all_anomalies
        return all_anomalies


class ModelBasedAnomalyDetector:
    """Detect anomalies using machine learning models"""

    def __init__(self):
        self.models = {}
        self.anomalies = {}

    def detect_isolation_forest(self, feature_matrix: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        if feature_matrix.empty:
            return pd.DataFrame()
        try:
            X = feature_matrix.select_dtypes(include=[np.number]).fillna(0)
            if X.empty:
                return pd.DataFrame()
            model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            predictions = model.fit_predict(X)
            scores = model.score_samples(X)
            anomaly_mask = predictions == -1
            anomalies_df = feature_matrix[anomaly_mask].copy()
            if not anomalies_df.empty:
                anomalies_df["anomaly_type"] = "isolation_forest"
                anomalies_df["anomaly_score"] = -scores[anomaly_mask]
                anomalies_df["severity"] = "medium"
                anomalies_df["reason"] = "Isolated pattern detected by ML model"
            self.models["isolation_forest"] = model
            return anomalies_df
        except Exception as e:
            st.warning(f"Isolation Forest failed: {str(e)}")
            return pd.DataFrame()


class AnomalyVisualizer:
    """Visualize anomalies using Plotly"""

    @staticmethod
    def plot_time_series_with_anomalies(df: pd.DataFrame, anomalies_df: pd.DataFrame, data_type: str, metric_col: str):
        fig = go.Figure()
        # Ensure timestamp is datetime and tz-naive
        df_loc = df.copy()
        df_loc = fix_timestamp_column(df_loc)
        if metric_col not in df_loc.columns:
            # nothing to plot
            return fig
        fig.add_trace(go.Scatter(x=df_loc["timestamp"], y=df_loc[metric_col], mode="lines+markers", name="Normal", line=dict(width=2), marker=dict(size=4)))
        if anomalies_df is not None and not anomalies_df.empty and "timestamp" in anomalies_df.columns:
            anomalies_local = anomalies_df.copy()
            anomalies_local = fix_timestamp_column(anomalies_local)
            severity_colors = {"low": "#fbbf24", "medium": "#f97316", "high": "#ef4444"}
            for severity, color in severity_colors.items():
                severity_data = anomalies_local[anomalies_local.get("severity", "") == severity]
                if not severity_data.empty and metric_col in severity_data.columns:
                    fig.add_trace(go.Scatter(
                        x=severity_data["timestamp"],
                        y=severity_data[metric_col],
                        mode="markers",
                        name=f"{severity.capitalize()} Anomaly",
                        marker=dict(size=12, color=color, symbol="x", line=dict(width=2, color="white")),
                        text=severity_data.get("reason", ""),
                        hovertemplate="<b>Anomaly</b><br>Value: %{y:.2f}<br>%{text}<extra></extra>"
                    ))
        fig.update_layout(title=f"Anomaly Detection: {data_type.replace('_', ' ').title()}", xaxis_title="Time", yaxis_title=metric_col.replace("_", " ").title(), hovermode="x unified", height=500, template="plotly_white")
        return fig

    @staticmethod
    def plot_anomaly_distribution(anomalies_df: pd.DataFrame, data_type: str):
        if anomalies_df is None or anomalies_df.empty:
            return None
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Anomalies by Type", "Anomalies by Severity"), specs=[[{"type": "bar"}, {"type": "pie"}]])
        if "anomaly_type" in anomalies_df.columns:
            type_counts = anomalies_df["anomaly_type"].value_counts()
            fig.add_trace(go.Bar(x=type_counts.index, y=type_counts.values, text=type_counts.values, textposition="auto"), row=1, col=1)
        if "severity" in anomalies_df.columns:
            severity_counts = anomalies_df["severity"].value_counts()
            colors = ["#ef4444", "#f97316", "#fbbf24"]
            fig.add_trace(go.Pie(labels=severity_counts.index, values=severity_counts.values, marker=dict(colors=colors)), row=1, col=2)
        fig.update_layout(title_text=f"Anomaly Analysis: {data_type.replace('_', ' ').title()}", height=400, showlegend=True)
        return fig

    @staticmethod
    def plot_anomaly_heatmap(all_anomalies: Dict[str, pd.DataFrame]):
        if not all_anomalies:
            return None
        heatmap_data = []
        for data_type, anomalies_df in all_anomalies.items():
            if anomalies_df is None or anomalies_df.empty:
                continue
            tmp = anomalies_df.copy()
            tmp = fix_timestamp_column(tmp)
            if "timestamp" in tmp.columns:
                tmp["hour"] = tmp["timestamp"].dt.hour
                hourly_counts = tmp.groupby("hour").size()
                for hour in range(24):
                    heatmap_data.append({"Data Type": data_type.replace("_", " ").title(), "Hour": hour, "Anomalies": int(hourly_counts.get(hour, 0))})
        if not heatmap_data:
            return None
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(index="Data Type", columns="Hour", values="Anomalies").fillna(0)
        fig = go.Figure(data=go.Heatmap(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index, colorscale="Reds", text=pivot_df.values, texttemplate="%{text}", textfont={"size": 10}))
        fig.update_layout(title="Anomaly Frequency Heatmap (by Hour)", xaxis_title="Hour of Day", yaxis_title="Data Type", height=400)
        return fig

    @staticmethod
    def plot_severity_timeline(anomalies_df: pd.DataFrame, data_type: str):
        if anomalies_df is None or anomalies_df.empty or "timestamp" not in anomalies_df.columns:
            return None
        tmp = anomalies_df.copy()
        tmp = fix_timestamp_column(tmp)
        severity_colors = {"low": "#fbbf24", "medium": "#f97316", "high": "#ef4444"}
        fig = go.Figure()
        for severity, color in severity_colors.items():
            severity_data = tmp[tmp.get("severity", "") == severity]
            if not severity_data.empty:
                fig.add_trace(go.Scatter(x=severity_data["timestamp"], y=[severity] * len(severity_data), mode="markers", name=severity.capitalize(), marker=dict(size=12, color=color, line=dict(width=2, color="white")), text=severity_data.get("reason", ""), hovertemplate="<b>%{text}</b><br>Time: %{x}<extra></extra>"))
        fig.update_layout(title=f"Anomaly Timeline: {data_type.replace('_', ' ').title()}", xaxis_title="Time", yaxis_title="Severity", height=300, showlegend=True, template="plotly_white")
        return fig

# -------------------------
# Sample data creator (keeps sleep and spo2)
# -------------------------
def create_sample_data_with_anomalies() -> Dict[str, pd.DataFrame]:
    timestamps = pd.date_range(start="2024-01-15 08:00:00", end="2024-01-15 16:00:00", freq="1min")
    # heart_rate
    hr_data = []
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        base_hr = 70
        if i in [100, 250, 380]:
            hr = np.random.choice([45, 190])
        elif 9 <= time_of_day < 10:
            hr = base_hr * 1.5 + np.random.normal(0, 3)
        else:
            hr = base_hr + np.random.normal(0, 3)
        hr_data.append(max(40, min(200, hr)))
    # stress
    stress_data = []
    for i, ts in enumerate(timestamps):
        base_stress = 40
        if i in [150, 300, 420]:
            stress = 95
        else:
            stress = base_stress + np.random.normal(0, 5)
        stress_data.append(max(0, min(100, stress)))
    # spo2
    spo2_data = []
    for i, ts in enumerate(timestamps):
        base_spo2 = 98
        if i in [200, 350]:
            spo2 = 88
        else:
            spo2 = base_spo2 + np.random.normal(0, 0.5)
        spo2_data.append(max(85, min(100, spo2)))
    # steps
    steps_data = []
    for i, ts in enumerate(timestamps):
        base_steps = 50
        if i in [180, 320, 450]:
            steps = np.random.choice([0, 400])
        elif 12 <= ts.hour < 13:
            steps = base_steps * 1.8 + np.random.normal(0, 10)
        else:
            steps = base_steps + np.random.normal(0, 8)
        steps_data.append(max(0, min(500, steps)))
    # sleep (hourly blocks)
    sleep_timestamps = pd.date_range(start="2024-01-14 22:00:00", end="2024-01-15 08:00:00", freq="1H")
    sleep_duration_data = []
    for i, ts in enumerate(sleep_timestamps):
        base_duration = 55
        if i in [2, 5]:
            duration = 10
        elif i in [7, 9]:
            duration = 60
        else:
            duration = base_duration + np.random.normal(0, 3)
        sleep_duration_data.append(max(0, min(60, duration)))
    hr_df = pd.DataFrame({"timestamp": timestamps, "heart_rate": hr_data})
    stress_df = pd.DataFrame({"timestamp": timestamps, "stress_level": stress_data})
    spo2_df = pd.DataFrame({"timestamp": timestamps, "spo2_level": spo2_data})
    steps_df = pd.DataFrame({"timestamp": timestamps, "step_count": steps_data})
    sleep_df = pd.DataFrame({"timestamp": sleep_timestamps, "duration_minutes": sleep_duration_data})
    # ensure datetime and tz-normalization
    data_dict = {
        "heart_rate": fix_timestamp_column(hr_df),
        "stress": fix_timestamp_column(stress_df),
        "spo2": fix_timestamp_column(spo2_df),
        "steps": fix_timestamp_column(steps_df),
        "sleep": fix_timestamp_column(sleep_df),
    }
    return data_dict

# -------------------------
# Helpers: parse uploaded files (single combined or multiple)
# -------------------------
def try_read_file(file) -> pd.DataFrame:
    """Read CSV or JSON file-like into DataFrame; lower-case columns and fix timestamps"""
    try:
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        # try json first
        df = pd.DataFrame()
        try:
            parsed = json.loads(raw)
            df = pd.json_normalize(parsed)
        except Exception:
            # fallback to csv
            df = pd.read_csv(io.StringIO(raw))
        df.columns = df.columns.str.lower().str.strip()
        df = fix_timestamp_column(df)
        return df
    except Exception as e:
        st.warning(f"Failed to read uploaded file: {e}")
        return pd.DataFrame()

def build_data_dict_from_uploads(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Accepts uploaded file objects. Supports:
    - single combined CSV/JSON with multiple columns
    - or multiple files each containing timestamp + one metric
    Returns a data_dict with keys: heart_rate, stress, spo2, steps, sleep
    """
    data_frames = {}
    metric_map = {
        "heart_rate": ["heart_rate", "hr"],
        "stress": ["stress", "stress_level"],
        "spo2": ["spo2", "spo2_level"],
        "steps": ["steps", "step_count", "stepcount"],
        "sleep": ["sleep", "duration_minutes", "sleep_minutes", "sleep_duration"],
    }
    # Try reading each uploaded file
    for file in uploaded_files:
        df = try_read_file(file)
        if df is None or df.empty:
            continue
        # If the file looks like combined (has many known columns), extract each metric
        for metric, candidates in metric_map.items():
            found = None
            for cand in candidates:
                if cand in df.columns:
                    found = cand
                    break
            if found is not None and "timestamp" in df.columns:
                colname = {
                    "heart_rate": "heart_rate",
                    "stress": "stress_level",
                    "spo2": "spo2_level",
                    "steps": "step_count",
                    "sleep": "duration_minutes",
                }[metric]
                temp = df[["timestamp", found]].copy()
                temp.columns = ["timestamp", colname]
                temp = fix_timestamp_column(temp)
                if metric in data_frames:
                    # merge by timestamp (nearest within 1 minute)
                    try:
                        merged = pd.merge_asof(data_frames[metric].sort_values("timestamp"), temp.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta("1min"))
                        data_frames[metric] = merged
                    except Exception:
                        # fallback: append rows and drop duplicates
                        append_df = pd.concat([data_frames[metric], temp], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                        data_frames[metric] = append_df
                else:
                    data_frames[metric] = temp.sort_values("timestamp").reset_index(drop=True)
    # If the user uploaded exactly one combined file but build_data_dict_from_uploads returned empty (edge case),
    # try to parse the first file as a combined dataset (handled above but keep safetynet)
    if not data_frames and uploaded_files:
        df_combined = try_read_file(uploaded_files[0])
        if not df_combined.empty and "timestamp" in df_combined.columns:
            for metric, candidates in metric_map.items():
                for cand in candidates:
                    if cand in df_combined.columns:
                        colname = {
                            "heart_rate": "heart_rate",
                            "stress": "stress_level",
                            "spo2": "spo2_level",
                            "steps": "step_count",
                            "sleep": "duration_minutes",
                        }[metric]
                        temp = df_combined[["timestamp", cand]].copy()
                        temp.columns = ["timestamp", colname]
                        temp = fix_timestamp_column(temp)
                        data_frames[metric] = temp.sort_values("timestamp").reset_index(drop=True)
    return data_frames

# -------------------------
# Main UI
# -------------------------
def main():
    # Header
    st.markdown(
        """
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>🚨 FitPulse Milestone 3</h1>
        <p style='color: white; font-size: 20px; margin: 10px 0 0 0;'>
            Anomaly Detection & Visualization
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>📏 Rule-Based</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Threshold Detection</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>🤖 Model-Based</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>ML Detection</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>📊 Visualization</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Interactive Charts</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        st.subheader("📁 Data Source")
        use_sample = st.checkbox("Use Sample Data (with anomalies)", value=True)
        uploaded_files = None
        if not use_sample:
            uploaded_files = st.file_uploader(
                "Upload Data from Milestone 2 (CSV/JSON). Accept single combined CSV or multiple files.",
                type=["csv", "json"],
                accept_multiple_files=True,
            )
        st.divider()
        st.subheader("🎯 Detection Methods")
        use_rule_based = st.checkbox("Rule-Based Detection", value=True)
        use_model_based = st.checkbox("Model-Based Detection", value=True)
        st.divider()
        st.subheader("📊 Rule-Based Settings")
        z_score_threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)
        st.subheader("🤖 Model-Based Settings")
        contamination = st.slider("Isolation Forest Contamination", 0.01, 0.3, 0.1, 0.01)
        residual_threshold = st.slider("Residual Threshold (std)", 2.0, 5.0, 3.0, 0.5)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detect Anomalies", "📊 Visualizations", "📈 Summary Report"])

    if "all_anomalies" not in st.session_state:
        st.session_state.all_anomalies = {}
    if "detection_summary" not in st.session_state:
        st.session_state.detection_summary = {}
    if "data_dict" not in st.session_state:
        st.session_state.data_dict = {}

    # -------------------------
    # Tab 1: Run Detection
    # -------------------------
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1_c, col2_c, col3_c = st.columns([1, 2, 1])
        with col2_c:
            if st.button("🚨 Run Anomaly Detection"):
                # Get data
                if use_sample:
                    data_dict = create_sample_data_with_anomalies()
                else:
                    if not uploaded_files:
                        st.error("❌ Please upload data files")
                        st.stop()
                    data_dict = build_data_dict_from_uploads(uploaded_files)
                    # If build_data_dict_from_uploads yielded nothing (edge case), try parse first file as combined
                    if not data_dict:
                        df_combined = try_read_file(uploaded_files[0])
                        if df_combined is None or df_combined.empty:
                            st.error("Could not parse uploaded file(s). Make sure they contain 'timestamp' and metric columns.")
                            st.stop()
                        # map known metric column names
                        mapping_candidates = {
                            "heart_rate": ["heart_rate", "hr"],
                            "stress": ["stress_level", "stress"],
                            "spo2": ["spo2_level", "spo2"],
                            "steps": ["step_count", "steps"],
                            "sleep": ["duration_minutes", "sleep_minutes", "sleep_duration"],
                        }
                        for metric, candidates in mapping_candidates.items():
                            for cand in candidates:
                                if cand in df_combined.columns:
                                    colname = {
                                        "heart_rate": "heart_rate",
                                        "stress": "stress_level",
                                        "spo2": "spo2_level",
                                        "steps": "step_count",
                                        "sleep": "duration_minutes",
                                    }[metric]
                                    temp = df_combined[["timestamp", cand]].copy()
                                    temp.columns = ["timestamp", colname]
                                    temp = fix_timestamp_column(temp)
                                    data_dict[metric] = temp.sort_values("timestamp").reset_index(drop=True)

                # Initialize detectors
                rule_detector = RuleBasedAnomalyDetector()
                model_detector = ModelBasedAnomalyDetector()

                # Reset session state
                st.session_state.all_anomalies = {}
                st.session_state.detection_summary = {}
                st.session_state.data_dict = data_dict

                # For each metric run detectors
                for data_type, df in data_dict.items():
                    st.markdown(f"### 🔍 Analyzing {data_type.replace('_', ' ').title()}")
                    all_anomalies_list = []

                    if df is None or df.empty:
                        st.info("No data for this metric.")
                        st.session_state.all_anomalies[data_type] = pd.DataFrame()
                        st.session_state.detection_summary[data_type] = {"total_anomalies": 0, "high_severity": 0, "medium_severity": 0, "low_severity": 0}
                        continue

                    # Rule-based detection (threshold + statistical)
                    if use_rule_based:
                        with st.expander("📏 Rule-Based Detection", expanded=True):
                            rule_anomalies = rule_detector.detect_all_rule_based(df, data_type, z_score_threshold)
                            if not rule_anomalies.empty:
                                st.success(f"✅ Found {len(rule_anomalies)} rule-based anomalies")
                                display_cols = [c for c in ["timestamp", "severity", "reason", "anomaly_type"] if c in rule_anomalies.columns]
                                st.dataframe(rule_anomalies[display_cols].head(10))
                                all_anomalies_list.append(rule_anomalies)
                            else:
                                st.info("No rule-based anomalies detected")

                    # Model-based detection
                    if use_model_based:
                        with st.expander("🤖 Model-Based Detection", expanded=True):
                            metric_col = {
                                "heart_rate": "heart_rate",
                                "stress": "stress_level",
                                "spo2": "spo2_level",
                                "steps": "step_count",
                                "sleep": "duration_minutes"
                            }.get(data_type)

                            if metric_col and metric_col in df.columns:
                                feature_matrix = df[[metric_col]].copy()
                                feature_matrix.index = df.index
                                model_anomalies = model_detector.detect_isolation_forest(feature_matrix, contamination)
                                if not model_anomalies.empty:
                                    model_anomalies = model_anomalies.copy()
                                    model_anomalies["timestamp"] = df.loc[model_anomalies.index, "timestamp"].values
                                    model_anomalies[metric_col] = df.loc[model_anomalies.index, metric_col].values
                                    st.success(f"✅ Found {len(model_anomalies)} model-based anomalies")
                                    display_cols = [c for c in ["timestamp", "anomaly_score", "reason", metric_col] if c in model_anomalies.columns]
                                    st.dataframe(model_anomalies[display_cols].head(10))
                                    all_anomalies_list.append(model_anomalies)
                                else:
                                    st.info("No model-based anomalies detected")
                            else:
                                st.info("No suitable metric column for model-based detection")

                    # Combine and dedupe
                    if all_anomalies_list:
                        combined_anomalies = pd.concat(all_anomalies_list, ignore_index=True)
                        if "timestamp" in combined_anomalies.columns:
                            combined_anomalies = fix_timestamp_column(combined_anomalies)
                            combined_anomalies = combined_anomalies.drop_duplicates(subset=["timestamp"])
                        st.session_state.all_anomalies[data_type] = combined_anomalies
                        st.session_state.detection_summary[data_type] = {
                            "total_anomalies": len(combined_anomalies),
                            "high_severity": len(combined_anomalies[combined_anomalies.get("severity", "") == "high"]),
                            "medium_severity": len(combined_anomalies[combined_anomalies.get("severity", "") == "medium"]),
                            "low_severity": len(combined_anomalies[combined_anomalies.get("severity", "") == "low"]),
                        }
                    else:
                        st.session_state.all_anomalies[data_type] = pd.DataFrame()
                        st.session_state.detection_summary[data_type] = {"total_anomalies": 0, "high_severity": 0, "medium_severity": 0, "low_severity": 0}

                    st.markdown("---")

                st.balloons()
                st.success("🎉 Anomaly Detection Complete!")

    # -------------------------
    # Tab 2: Visualizations
    # -------------------------
    with tab2:
        visualizer = AnomalyVisualizer()
        if "all_anomalies" in st.session_state and st.session_state.all_anomalies and "data_dict" in st.session_state:
            data_dict = st.session_state.data_dict
            types = list(data_dict.keys())
            selected_type = st.selectbox("Select Data Type", options=types, index=0)
            df = data_dict.get(selected_type)
            anomalies_df = st.session_state.all_anomalies.get(selected_type, pd.DataFrame())
            metric_col = {
                "heart_rate": "heart_rate",
                "stress": "stress_level",
                "spo2": "spo2_level",
                "steps": "step_count",
                "sleep": "duration_minutes"
            }.get(selected_type)
            if df is not None and metric_col:
                # Time series with anomalies
                st.subheader("📈 Time Series with Detected Anomalies")
                fig1 = visualizer.plot_time_series_with_anomalies(df, anomalies_df, selected_type, metric_col)
                st.plotly_chart(fig1, use_container_width=True)
                # Anomaly distribution
                st.subheader("📊 Anomaly Distribution Analysis")
                fig2 = visualizer.plot_anomaly_distribution(anomalies_df, selected_type)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No anomalies to show distribution")
                # Severity timeline
                st.subheader("⏱️ Anomaly Severity Timeline")
                fig3 = visualizer.plot_severity_timeline(anomalies_df, selected_type)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No anomalies to show timeline")
            # Overall heatmap
            st.subheader("🔥 Cross-Data Anomaly Heatmap")
            fig_heatmap = visualizer.plot_anomaly_heatmap(st.session_state.all_anomalies)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No cross-data anomalies to display (run detection first)")
        else:
            st.info("👈 Please run anomaly detection first in the 'Detect Anomalies' tab")

    # -------------------------
    # Tab 3: Summary & Export
    # -------------------------
    with tab3:
        if "detection_summary" in st.session_state and st.session_state.detection_summary:
            st.header("📈 Anomaly Detection Summary Report")
            total_anomalies = sum(s["total_anomalies"] for s in st.session_state.detection_summary.values())
            total_high = sum(s["high_severity"] for s in st.session_state.detection_summary.values())
            total_medium = sum(s["medium_severity"] for s in st.session_state.detection_summary.values())
            total_low = sum(s["low_severity"] for s in st.session_state.detection_summary.values())
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2 style='margin: 0; font-size: 36px;'>{total_anomalies}</h2>
                    <p style='margin: 5px 0 0 0;'>Total Anomalies</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2 style='margin: 0; font-size: 36px;'>{total_high}</h2>
                    <p style='margin: 5px 0 0 0;'>High Severity</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2 style='margin: 0; font-size: 36px;'>{total_medium}</h2>
                    <p style='margin: 5px 0 0 0;'>Medium Severity</p>
                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2 style='margin: 0; font-size: 36px;'>{total_low}</h2>
                    <p style='margin: 5px 0 0 0;'>Low Severity</p>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            # Detailed breakdown
            st.subheader("📊 Breakdown by Data Type")
            for data_type, summary in st.session_state.detection_summary.items():
                with st.expander(f"🔍 {data_type.replace('_', ' ').title()}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", summary["total_anomalies"])
                    c2.metric("High", summary["high_severity"], delta=None if summary["high_severity"] == 0 else "⚠️")
                    c3.metric("Medium", summary["medium_severity"])
                    c4.metric("Low", summary["low_severity"])
                    if data_type in st.session_state.all_anomalies:
                        anomalies_df = st.session_state.all_anomalies[data_type]
                        if anomalies_df is not None and not anomalies_df.empty:
                            st.markdown("**Recent Anomalies:**")
                            display_cols = ["timestamp", "severity", "anomaly_type", "reason"]
                            available_cols = [col for col in display_cols if col in anomalies_df.columns]
                            st.dataframe(anomalies_df[available_cols].head(10), use_container_width=True)
            # Severity distribution chart
            st.subheader("📈 Overall Severity Distribution")
            severity_data = pd.DataFrame({"Severity": ["High", "Medium", "Low"], "Count": [total_high, total_medium, total_low]})
            fig = px.bar(severity_data, x="Severity", y="Count", color="Severity", color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#fbbf24"}, text="Count")
            fig.update_layout(showlegend=False, height=400, template="plotly_white")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            # Recommendations and Export
            st.subheader("💡 Health Recommendations")
            if total_high > 0:
                st.markdown("""<div class='anomaly-card'><h3 style='color: #ef4444; margin-top: 0;'>⚠️ Critical Alerts</h3><p>High severity anomalies detected! Consider these actions:</p><ul><li>Review activity patterns during anomaly periods</li><li>Consult with healthcare provider if patterns persist</li><li>Check device calibration and fit</li><li>Monitor stress levels and sleep quality</li></ul></div>""", unsafe_allow_html=True)
            if total_medium > 5:
                st.markdown("""<div class='anomaly-card'><h3 style='color: #f97316; margin-top: 0;'>⚡ Attention Needed</h3><p>Multiple moderate anomalies detected:</p><ul><li>Review daily routines and lifestyle factors</li><li>Ensure adequate hydration and nutrition</li><li>Consider stress management techniques</li><li>Maintain consistent sleep schedule</li></ul></div>""", unsafe_allow_html=True)
            if total_anomalies == 0:
                st.success("✅ No significant anomalies detected! Your health metrics are within normal ranges.")
            st.subheader("💾 Export Results")
            colA, colB = st.columns(2)
            with colA:
                report_data = {"timestamp": datetime.now().isoformat(), "summary": st.session_state.detection_summary, "total_statistics": {"total_anomalies": total_anomalies, "high_severity": total_high, "medium_severity": total_medium, "low_severity": total_low}}
                json_str = json.dumps(report_data, indent=2, default=str)
                st.download_button(label="📥 Download Summary Report (JSON)", data=json_str, file_name=f"fitpulse_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            with colB:
                all_anomalies_combined = []
                for data_type, anomalies_df in st.session_state.all_anomalies.items():
                    if anomalies_df is None or anomalies_df.empty:
                        continue
                    temp_df = anomalies_df.copy()
                    temp_df["data_type"] = data_type
                    all_anomalies_combined.append(temp_df)
                if all_anomalies_combined:
                    combined_df = pd.concat(all_anomalies_combined, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    st.download_button(label="📊 Download Detailed Anomalies (CSV)", data=csv, file_name=f"fitpulse_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                else:
                    st.info("No anomalies to export")
        else:
            st.info("👈 Please run anomaly detection first in the 'Detect Anomalies' tab")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px;'>
        <p style='color: #6b7280; margin: 0;'>🏥 FitPulse Health Analytics | Milestone 3: Anomaly Detection & Visualization</p>
        <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>Powered by Streamlit, Plotly, Scikit-learn | © 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

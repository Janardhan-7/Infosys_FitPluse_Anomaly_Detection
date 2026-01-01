import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import io
import json
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from scipy import stats

# Feature Extraction (simplified - remove tsfresh/prophet dependencies)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FitPulse - Complete Health Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 48px;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: white;
        font-size: 20px;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .feature-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    
    .anomaly-alert {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fix_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp column to tz-naive datetime"""
    if df is None or df.empty:
        return df
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        try:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        except:
            try:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            except:
                pass
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    return df

def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """Generate comprehensive sample data for all metrics"""
    timestamps = pd.date_range(start='2024-01-15 08:00:00', 
                               end='2024-01-15 16:00:00', freq='1min')
    
    # Heart Rate with anomalies
    hr_data = []
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        base_hr = 70
        if i in [100, 250, 380]:  # Inject anomalies
            hr = np.random.choice([45, 190])
        elif 9 <= time_of_day < 10 or 14 <= time_of_day < 15:
            hr = base_hr * 1.5 + np.random.normal(0, 3)
        else:
            hr = base_hr + np.random.normal(0, 3)
        hr_data.append(max(40, min(200, hr)))
    
    # Stress levels with anomalies
    stress_data = []
    for i, ts in enumerate(timestamps):
        base_stress = 40
        if i in [150, 300, 420]:  # Inject anomalies
            stress = 95
        else:
            stress = base_stress + np.random.normal(0, 5)
        stress_data.append(max(0, min(100, stress)))
    
    # SpO2 levels with anomalies
    spo2_data = []
    for i, ts in enumerate(timestamps):
        base_spo2 = 98
        if i in [200, 350]:  # Inject anomalies
            spo2 = 88
        else:
            spo2 = base_spo2 + np.random.normal(0, 0.5)
        spo2_data.append(max(85, min(100, spo2)))
    
    # Steps with anomalies
    steps_data = []
    for i, ts in enumerate(timestamps):
        base_steps = 50
        if i in [180, 320, 450]:  # Inject anomalies
            steps = np.random.choice([0, 400])
        elif 12 <= ts.hour < 13:
            steps = base_steps * 1.8 + np.random.normal(0, 10)
        else:
            steps = base_steps + np.random.normal(0, 8)
        steps_data.append(max(0, min(500, steps)))
    
    # Sleep duration
    sleep_timestamps = pd.date_range(start='2024-01-14 22:00:00', 
                                    end='2024-01-15 08:00:00', freq='1H')
    sleep_data = []
    for i, ts in enumerate(sleep_timestamps):
        base_duration = 55
        if i in [2, 5]:  # Inject anomalies
            duration = 10
        else:
            duration = base_duration + np.random.normal(0, 3)
        sleep_data.append(max(0, min(60, duration)))
    
    return {
        'heart_rate': fix_timestamp_column(pd.DataFrame({
            'timestamp': timestamps, 'heart_rate': hr_data
        })),
        'stress': fix_timestamp_column(pd.DataFrame({
            'timestamp': timestamps, 'stress_level': stress_data
        })),
        'spo2': fix_timestamp_column(pd.DataFrame({
            'timestamp': timestamps, 'spo2_level': spo2_data
        })),
        'steps': fix_timestamp_column(pd.DataFrame({
            'timestamp': timestamps, 'step_count': steps_data
        })),
        'sleep': fix_timestamp_column(pd.DataFrame({
            'timestamp': sleep_timestamps, 'duration_minutes': sleep_data
        }))
    }

# ============================================================================
# DATA PREPROCESSING FUNCTIONS (Milestone 1)
# ============================================================================

def apply_time_alignment(df, ts_col, freq_str='1min', fill_method='interpolate'):
    """Resample data to target frequency"""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.set_index(ts_col)
    
    original_rows = len(df)
    resampled = df.resample(freq_str).mean()
    
    if fill_method == 'interpolate':
        resampled = resampled.interpolate(limit_direction='both')
    elif fill_method == 'ffill':
        resampled = resampled.ffill()
    elif fill_method == 'bfill':
        resampled = resampled.bfill()
    elif fill_method == 'mean':
        resampled = resampled.fillna(resampled.mean())
    elif fill_method == 'drop':
        resampled = resampled.dropna(how='any')
    
    return resampled, {
        'original_rows': original_rows,
        'resampled_rows': len(resampled),
        'frequency': freq_str,
        'fill_method': fill_method
    }

# ============================================================================
# FEATURE EXTRACTION (Milestone 2 - Simplified)
# ============================================================================

def extract_statistical_features(df: pd.DataFrame, data_type: str, 
                                window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
    """Extract basic statistical features from time-series"""
    metric_columns = {
        'heart_rate': 'heart_rate',
        'stress': 'stress_level',
        'spo2': 'spo2_level',
        'steps': 'step_count',
        'sleep': 'duration_minutes'
    }
    
    if data_type not in metric_columns:
        return pd.DataFrame(), {}
    
    metric_col = metric_columns[data_type]
    if metric_col not in df.columns:
        return pd.DataFrame(), {}
    
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    features_list = []
    step_size = window_size // 2
    
    for i in range(0, len(df_sorted) - window_size + 1, step_size):
        window = df_sorted.iloc[i:i+window_size][metric_col]
        
        features = {
            'mean': window.mean(),
            'std': window.std(),
            'min': window.min(),
            'max': window.max(),
            'median': window.median(),
            'q25': window.quantile(0.25),
            'q75': window.quantile(0.75),
            'range': window.max() - window.min(),
            'skewness': window.skew(),
            'kurtosis': window.kurtosis()
        }
        features_list.append(features)
    
    feature_df = pd.DataFrame(features_list)
    
    return feature_df, {
        'data_type': data_type,
        'features_extracted': len(feature_df.columns),
        'windows': len(feature_df),
        'window_size': window_size
    }

def perform_clustering(feature_matrix: pd.DataFrame, n_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
    """Perform K-Means clustering on features"""
    if feature_matrix.empty:
        return np.array([]), {}
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix.fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    report = {
        'n_clusters': n_clusters,
        'cluster_sizes': {int(i): int(c) for i, c in 
                         zip(*np.unique(labels, return_counts=True))}
    }
    
    if len(np.unique(labels)) > 1:
        report['silhouette_score'] = silhouette_score(features_scaled, labels)
    
    return labels, report

# ============================================================================
# ANOMALY DETECTION (Milestone 3)
# ============================================================================

class AnomalyDetector:
    """Unified anomaly detection class"""
    
    def __init__(self):
        self.thresholds = {
            'heart_rate': {'min': 40, 'max': 180},
            'stress': {'min': 0, 'max': 100},
            'spo2': {'min': 90, 'max': 100},
            'steps': {'min': 0, 'max': 500},
            'sleep': {'min': 0, 'max': 60}
        }
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str, 
                        z_threshold: float = 3.0,
                        contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies using multiple methods"""
        metric_columns = {
            'heart_rate': 'heart_rate',
            'stress': 'stress_level',
            'spo2': 'spo2_level',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        if data_type not in metric_columns:
            return pd.DataFrame()
        
        metric_col = metric_columns[data_type]
        if metric_col not in df.columns:
            return pd.DataFrame()
        
        all_anomalies = []
        
        # 1. Threshold-based detection
        thresholds = self.thresholds.get(data_type, {})
        min_val = thresholds.get('min', -np.inf)
        max_val = thresholds.get('max', np.inf)
        
        threshold_anomalies = df[
            (df[metric_col] < min_val) | (df[metric_col] > max_val)
        ].copy()
        
        if not threshold_anomalies.empty:
            threshold_anomalies['anomaly_type'] = 'threshold'
            threshold_anomalies['severity'] = 'high'
            threshold_anomalies['reason'] = 'Value outside normal range'
            all_anomalies.append(threshold_anomalies)
        
        # 2. Statistical detection (Z-score)
        if len(df) > 3 and df[metric_col].std() > 0:
            z_scores = np.abs(stats.zscore(df[metric_col].fillna(df[metric_col].mean())))
            statistical_anomalies = df[z_scores > z_threshold].copy()
            
            if not statistical_anomalies.empty:
                statistical_anomalies['anomaly_type'] = 'statistical'
                statistical_anomalies['severity'] = 'medium'
                statistical_anomalies['reason'] = 'Statistical outlier'
                all_anomalies.append(statistical_anomalies)
        
        # 3. Isolation Forest
        try:
            X = df[[metric_col]].fillna(0)
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)
            
            model_anomalies = df[predictions == -1].copy()
            if not model_anomalies.empty:
                model_anomalies['anomaly_type'] = 'ml_based'
                model_anomalies['severity'] = 'medium'
                model_anomalies['reason'] = 'ML model detected anomaly'
                all_anomalies.append(model_anomalies)
        except:
            pass
        
        # Combine all anomalies
        if all_anomalies:
            combined = pd.concat(all_anomalies, ignore_index=True)
            if 'timestamp' in combined.columns:
                combined = combined.drop_duplicates(subset=['timestamp'])
            return combined
        
        return pd.DataFrame()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_time_series_plot(df: pd.DataFrame, metric_col: str, 
                           title: str, anomalies: pd.DataFrame = None):
    """Create interactive time series plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[metric_col],
        mode='lines+markers',
        name='Normal',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4)
    ))
    
    if anomalies is not None and not anomalies.empty:
        severity_colors = {
            'low': '#fbbf24',
            'medium': '#f97316',
            'high': '#ef4444'
        }
        
        for severity, color in severity_colors.items():
            severity_data = anomalies[anomalies.get('severity', '') == severity]
            if not severity_data.empty and metric_col in severity_data.columns:
                fig.add_trace(go.Scatter(
                    x=severity_data['timestamp'],
                    y=severity_data[metric_col],
                    mode='markers',
                    name=f'{severity.capitalize()} Anomaly',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='x',
                        line=dict(width=2, color='white')
                    )
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=metric_col.replace('_', ' ').title(),
        hovermode='x unified',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_summary_metrics(data_dict: Dict[str, pd.DataFrame], 
                          all_anomalies: Dict[str, pd.DataFrame]):
    """Create summary statistics"""
    summary = {
        'total_records': sum(len(df) for df in data_dict.values()),
        'total_anomalies': sum(len(anom) if anom is not None and not anom.empty else 0 
                              for anom in all_anomalies.values()),
        'data_types': len(data_dict),
        'time_range': None
    }
    
    # Calculate time range
    all_timestamps = []
    for df in data_dict.values():
        if 'timestamp' in df.columns:
            all_timestamps.extend(df['timestamp'].tolist())
    
    if all_timestamps:
        summary['time_range'] = (min(all_timestamps), max(all_timestamps))
    
    return summary

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 FitPulse Health Analytics</h1>
        <p>Complete Dashboard for Real-Time Health Insights & Anomaly Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_dict' not in st.session_state:
        st.session_state.data_dict = {}
    if 'all_anomalies' not in st.session_state:
        st.session_state.all_anomalies = {}
    if 'features' not in st.session_state:
        st.session_state.features = {}
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### 📁 Data Source")
        use_sample = st.checkbox("Use Sample Data", value=True, 
                                help="Use pre-generated sample data with anomalies")
        
        uploaded_files = None
        if not use_sample:
            uploaded_files = st.file_uploader(
                "Upload Health Data Files",
                type=['csv', 'json'],
                accept_multiple_files=True,
                help="Upload CSV or JSON files with timestamp and health metrics"
            )
        
        st.markdown("---")
        
        st.markdown("### 🔧 Processing Settings")
        target_freq = st.selectbox(
            "Resampling Frequency",
            ['1min', '5min', '15min', '30min', '1H'],
            help="Target frequency for time alignment"
        )
        
        fill_method = st.selectbox(
            "Missing Value Method",
            ['interpolate', 'ffill', 'bfill', 'mean', 'drop'],
            help="Method to handle missing values"
        )
        
        st.markdown("---")
        
        st.markdown("### 🎯 Anomaly Detection")
        z_threshold = st.slider(
            "Z-Score Threshold",
            2.0, 5.0, 3.0, 0.5,
            help="Higher values = less sensitive"
        )
        
        contamination = st.slider(
            "ML Contamination Rate",
            0.01, 0.3, 0.1, 0.01,
            help="Expected proportion of anomalies"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Feature Extraction")
        window_size = st.slider(
            "Window Size (minutes)",
            10, 120, 60, 10,
            help="Window size for feature extraction"
        )
        
        n_clusters = st.slider(
            "Number of Clusters",
            2, 10, 3,
            help="Number of behavior clusters"
        )
        
        st.markdown("---")
        
        if st.button("🚀 Load & Process Data", type="primary"):
            with st.spinner("Loading and processing data..."):
                # Load data
                if use_sample:
                    st.session_state.data_dict = generate_sample_data()
                else:
                    if uploaded_files:
                        # Simple file loading (implement full parsing if needed)
                        st.session_state.data_dict = generate_sample_data()
                    else:
                        st.error("Please upload files or use sample data")
                        return
                
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
                st.rerun()
    
    # Main Content Area
    if not st.session_state.data_loaded:
        st.info("👈 Please configure settings and load data from the sidebar")
        
        # Show feature overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='feature-badge'>
                <h3 style='margin: 0;'>📊 Milestone 1</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px;'>Data Preprocessing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-badge'>
                <h3 style='margin: 0;'>🔬 Milestone 2</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px;'>Feature Extraction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='feature-badge'>
                <h3 style='margin: 0;'>🚨 Milestone 3</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px;'>Anomaly Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='feature-badge'>
                <h3 style='margin: 0;'>📈 Milestone 4</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px;'>Insights Dashboard</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Process data if loaded
    data_dict = st.session_state.data_dict
    
    # Run anomaly detection if not done
    if not st.session_state.all_anomalies:
        detector = AnomalyDetector()
        for data_type, df in data_dict.items():
            st.session_state.all_anomalies[data_type] = detector.detect_anomalies(
                df, data_type, z_threshold, contamination
            )
    
    # Run feature extraction if not done
    if not st.session_state.features:
        for data_type, df in data_dict.items():
            features, report = extract_statistical_features(df, data_type, window_size)
            st.session_state.features[data_type] = {
                'features': features,
                'report': report
            }
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Data Analysis", 
        "🚨 Anomaly Detection",
        "🔬 Feature Insights",
        "📄 Reports"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.markdown("## 📊 Health Metrics Overview")
        
        # Summary metrics
        summary = create_summary_metrics(data_dict, st.session_state.all_anomalies)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #3b82f6; margin: 0; font-size: 36px;'>{summary['data_types']}</h2>
                <p style='margin: 5px 0 0 0; color: #6b7280;'>Data Types</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #10b981; margin: 0; font-size: 36px;'>{summary['total_records']:,}</h2>
                <p style='margin: 5px 0 0 0; color: #6b7280;'>Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #ef4444; margin: 0; font-size: 36px;'>{summary['total_anomalies']}</h2>
                <p style='margin: 5px 0 0 0; color: #6b7280;'>Anomalies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            anomaly_rate = (summary['total_anomalies'] / summary['total_records'] * 100) if summary['total_records'] > 0 else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #f97316; margin: 0; font-size: 36px;'>{anomaly_rate:.1f}%</h2>
                <p style='margin: 5px 0 0 0; color: #6b7280;'>Anomaly Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick visualization for all metrics
        st.markdown("### 📈 Real-Time Metrics")
        
        for data_type, df in data_dict.items():
            if df.empty:
                continue
            
            metric_col = {
                'heart_rate': 'heart_rate',
                'stress': 'stress_level',
                'spo2': 'spo2_level',
                'steps': 'step_count',
                'sleep': 'duration_minutes'
            }.get(data_type)
            
            if metric_col and metric_col in df.columns:
                anomalies = st.session_state.all_anomalies.get(data_type, pd.DataFrame())
                
                with st.expander(f"📊 {data_type.replace('_', ' ').title()}", expanded=False):
                    fig = create_time_series_plot(
                        df, metric_col,
                        f"{data_type.replace('_', ' ').title()} Over Time",
                        anomalies
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{df[metric_col].mean():.2f}")
                    col2.metric("Std Dev", f"{df[metric_col].std():.2f}")
                    col3.metric("Min", f"{df[metric_col].min():.2f}")
                    col4.metric("Max", f"{df[metric_col].max():.2f}")
    
    # ========================================================================
    # TAB 2: DATA ANALYSIS
    # ========================================================================
    with tab2:
        st.markdown("## 📈 Detailed Data Analysis")
        
        # Data type selector
        data_type_names = [dt.replace('_', ' ').title() for dt in data_dict.keys()]
        selected_name = st.selectbox("Select Health Metric", data_type_names)
        selected_type = list(data_dict.keys())[data_type_names.index(selected_name)]
        
        df = data_dict[selected_type]
        metric_col = {
            'heart_rate': 'heart_rate',
            'stress': 'stress_level',
            'spo2': 'spo2_level',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }.get(selected_type)
        
        if df.empty or not metric_col or metric_col not in df.columns:
            st.warning("No data available for this metric")
        else:
            # Main time series
            anomalies = st.session_state.all_anomalies.get(selected_type, pd.DataFrame())
            fig = create_time_series_plot(
                df, metric_col,
                f"{selected_name} Detailed Analysis",
                anomalies
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Statistical Summary")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                    'Value': [
                        f"{df[metric_col].mean():.2f}",
                        f"{df[metric_col].median():.2f}",
                        f"{df[metric_col].std():.2f}",
                        f"{df[metric_col].min():.2f}",
                        f"{df[metric_col].max():.2f}",
                        f"{df[metric_col].max() - df[metric_col].min():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 📈 Distribution")
                fig_hist = px.histogram(
                    df, x=metric_col,
                    nbins=30,
                    title=f"{selected_name} Distribution"
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Hourly patterns
            st.markdown("### ⏰ Hourly Patterns")
            df_hourly = df.copy()
            df_hourly['hour'] = df_hourly['timestamp'].dt.hour
            hourly_avg = df_hourly.groupby('hour')[metric_col].mean().reset_index()
            
            fig_hourly = px.bar(
                hourly_avg, x='hour', y=metric_col,
                title=f"Average {selected_name} by Hour"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Raw data table
            st.markdown("### 📋 Raw Data")
            st.dataframe(df.head(100), use_container_width=True)
    
    # ========================================================================
    # TAB 3: ANOMALY DETECTION
    # ========================================================================
    with tab3:
        st.markdown("## 🚨 Anomaly Detection Results")
        
        all_anomalies = st.session_state.all_anomalies
        total_anomalies = sum(len(a) if a is not None and not a.empty else 0 
                             for a in all_anomalies.values())
        
        if total_anomalies == 0:
            st.markdown("""
            <div class='success-box'>
                <h3 style='margin: 0;'>✅ No Anomalies Detected</h3>
                <p style='margin: 10px 0 0 0;'>All health metrics are within normal ranges!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Severity breakdown
            total_high = sum(len(a[a.get('severity', '') == 'high']) 
                           if a is not None and not a.empty else 0 
                           for a in all_anomalies.values())
            total_medium = sum(len(a[a.get('severity', '') == 'medium']) 
                             if a is not None and not a.empty else 0 
                             for a in all_anomalies.values())
            total_low = sum(len(a[a.get('severity', '') == 'low']) 
                          if a is not None and not a.empty else 0 
                          for a in all_anomalies.values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: #3b82f6; margin: 0;'>{total_anomalies}</h2>
                    <p style='margin: 5px 0 0 0;'>Total</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: #ef4444; margin: 0;'>{total_high}</h2>
                    <p style='margin: 5px 0 0 0;'>High</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: #f97316; margin: 0;'>{total_medium}</h2>
                    <p style='margin: 5px 0 0 0;'>Medium</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: #fbbf24; margin: 0;'>{total_low}</h2>
                    <p style='margin: 5px 0 0 0;'>Low</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Anomaly timeline
            st.markdown("### 📅 Anomaly Timeline")
            
            timeline_data = []
            for data_type, anomalies in all_anomalies.items():
                if anomalies is not None and not anomalies.empty:
                    for _, row in anomalies.iterrows():
                        timeline_data.append({
                            'timestamp': row['timestamp'],
                            'type': data_type.replace('_', ' ').title(),
                            'severity': row.get('severity', 'medium'),
                            'reason': row.get('reason', 'Unknown')
                        })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                fig_timeline = px.scatter(
                    timeline_df,
                    x='timestamp',
                    y='type',
                    color='severity',
                    color_discrete_map={
                        'high': '#ef4444',
                        'medium': '#f97316',
                        'low': '#fbbf24'
                    },
                    hover_data=['reason'],
                    title="Anomaly Timeline Across All Metrics"
                )
                fig_timeline.update_traces(marker=dict(size=12, symbol='diamond'))
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Detailed anomaly list
            st.markdown("### 📋 Detailed Anomaly Report")
            
            for data_type, anomalies in all_anomalies.items():
                if anomalies is not None and not anomalies.empty:
                    with st.expander(
                        f"🔍 {data_type.replace('_', ' ').title()} - {len(anomalies)} anomalies",
                        expanded=False
                    ):
                        display_cols = [c for c in ['timestamp', 'severity', 'anomaly_type', 'reason'] 
                                      if c in anomalies.columns]
                        st.dataframe(anomalies[display_cols], use_container_width=True)
            
            # Health recommendations
            st.markdown("### 💡 Health Recommendations")
            
            if total_high > 0:
                st.markdown("""
                <div class='anomaly-alert'>
                    <h4 style='margin: 0 0 10px 0;'>⚠️ Critical Alerts Detected</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Consult with healthcare provider immediately</li>
                        <li>Review activity patterns during anomaly periods</li>
                        <li>Check device calibration and proper fit</li>
                        <li>Consider stress management and rest</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if total_medium > 5:
                st.markdown("""
                <div class='info-box'>
                    <h4 style='margin: 0 0 10px 0;'>ℹ️ Moderate Anomalies Detected</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>Monitor patterns over the next few days</li>
                        <li>Maintain consistent sleep schedule</li>
                        <li>Stay hydrated and eat balanced meals</li>
                        <li>Practice stress reduction techniques</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: FEATURE INSIGHTS
    # ========================================================================
    with tab4:
        st.markdown("## 🔬 Feature Extraction & Clustering")
        
        # Select data type
        data_type_names = [dt.replace('_', ' ').title() for dt in data_dict.keys()]
        selected_name = st.selectbox(
            "Select Metric for Analysis",
            data_type_names,
            key='feature_select'
        )
        selected_type = list(data_dict.keys())[data_type_names.index(selected_name)]
        
        feature_data = st.session_state.features.get(selected_type, {})
        
        if not feature_data or feature_data['features'].empty:
            st.info("No features extracted for this metric")
        else:
            features = feature_data['features']
            report = feature_data['report']
            
            # Feature summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Features Extracted", report['features_extracted'])
            with col2:
                st.metric("Windows Analyzed", report['windows'])
            with col3:
                st.metric("Window Size", f"{report['window_size']} min")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top features by variance
            st.markdown("### 📊 Top Features by Variance")
            
            feature_variance = features.var().sort_values(ascending=False).head(10)
            
            fig_features = px.bar(
                x=feature_variance.index,
                y=feature_variance.values,
                labels={'x': 'Feature', 'y': 'Variance'},
                title="Most Variable Features"
            )
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Clustering analysis
            st.markdown("### 🎯 Behavior Clustering")
            
            if st.button("Run Clustering Analysis", key='cluster_btn'):
                with st.spinner("Performing clustering..."):
                    labels, cluster_report = perform_clustering(
                        features, n_clusters
                    )
                    
                    if len(labels) > 0:
                        st.success(f"✅ Identified {cluster_report['n_clusters']} clusters")
                        
                        # Show cluster sizes
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            cluster_sizes = pd.DataFrame({
                                'Cluster': list(cluster_report['cluster_sizes'].keys()),
                                'Size': list(cluster_report['cluster_sizes'].values())
                            })
                            
                            fig_clusters = px.pie(
                                cluster_sizes,
                                values='Size',
                                names='Cluster',
                                title="Cluster Distribution"
                            )
                            st.plotly_chart(fig_clusters, use_container_width=True)
                        
                        with col2:
                            if 'silhouette_score' in cluster_report:
                                st.metric(
                                    "Silhouette Score",
                                    f"{cluster_report['silhouette_score']:.3f}",
                                    help="Higher is better (range: -1 to 1)"
                                )
                            
                            st.markdown("""
                            <div class='info-box'>
                                <h4 style='margin: 0;'>Cluster Interpretation</h4>
                                <p style='margin: 10px 0 0 0;'>
                                    Different clusters represent distinct behavioral patterns
                                    in your health metrics over time.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # PCA visualization
                        if features.shape[1] >= 2:
                            st.markdown("### 📈 PCA Visualization")
                            
                            scaler = StandardScaler()
                            features_scaled = scaler.fit_transform(features.fillna(0))
                            
                            pca = PCA(n_components=2)
                            pca_features = pca.fit_transform(features_scaled)
                            
                            pca_df = pd.DataFrame({
                                'PC1': pca_features[:, 0],
                                'PC2': pca_features[:, 1],
                                'Cluster': labels
                            })
                            
                            fig_pca = px.scatter(
                                pca_df,
                                x='PC1',
                                y='PC2',
                                color='Cluster',
                                title="PCA Projection of Behavioral Clusters"
                            )
                            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Feature correlation
            st.markdown("### 🔗 Feature Correlations")
            
            corr_matrix = features.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # ========================================================================
    # TAB 5: REPORTS
    # ========================================================================
    with tab5:
        st.markdown("## 📄 Export Reports")
        
        # Summary report
        st.markdown("### 📊 Summary Report")
        
        summary = create_summary_metrics(data_dict, st.session_state.all_anomalies)
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_records': summary['total_records'],
                'total_anomalies': summary['total_anomalies'],
                'data_types': summary['data_types'],
                'anomaly_rate': f"{(summary['total_anomalies'] / summary['total_records'] * 100):.2f}%"
                               if summary['total_records'] > 0 else "0%"
            },
            'metrics': {}
        }
        
        # Add per-metric data
        for data_type, df in data_dict.items():
            metric_col = {
                'heart_rate': 'heart_rate',
                'stress': 'stress_level',
                'spo2': 'spo2_level',
                'steps': 'step_count',
                'sleep': 'duration_minutes'
            }.get(data_type)
            
            if metric_col and metric_col in df.columns:
                anomalies = st.session_state.all_anomalies.get(data_type, pd.DataFrame())
                
                report_data['metrics'][data_type] = {
                    'total_records': len(df),
                    'anomalies': len(anomalies) if anomalies is not None and not anomalies.empty else 0,
                    'mean': float(df[metric_col].mean()),
                    'std': float(df[metric_col].std()),
                    'min': float(df[metric_col].min()),
                    'max': float(df[metric_col].max())
                }
        
        # Display report
        st.json(report_data)
        
        # Export buttons
        st.markdown("### 💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON export
            json_str = json.dumps(report_data, indent=2)
            st.download_button(
                "📥 Download JSON Report",
                json_str,
                f"fitpulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV export for all anomalies
            all_anomalies_list = []
            for data_type, anomalies in st.session_state.all_anomalies.items():
                if anomalies is not None and not anomalies.empty:
                    temp = anomalies.copy()
                    temp['data_type'] = data_type
                    all_anomalies_list.append(temp)
            
            if all_anomalies_list:
                combined_anomalies = pd.concat(all_anomalies_list, ignore_index=True)
                csv_data = combined_anomalies.to_csv(index=False)
                
                st.download_button(
                    "📊 Download Anomalies CSV",
                    csv_data,
                    f"fitpulse_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.button(
                    "📊 No Anomalies to Export",
                    disabled=True,
                    use_container_width=True
                )
        
        with col3:
            # Combined data export
            if data_dict:
                # Merge all dataframes
                combined_data = pd.DataFrame()
                
                for data_type, df in data_dict.items():
                    if not df.empty:
                        temp = df.copy()
                        temp['metric_type'] = data_type
                        combined_data = pd.concat([combined_data, temp], ignore_index=True)
                
                if not combined_data.empty:
                    csv_all = combined_data.to_csv(index=False)
                    
                    st.download_button(
                        "📦 Download All Data CSV",
                        csv_all,
                        f"fitpulse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
        # Report summary
        st.markdown("### 📈 Report Summary")
        
        st.markdown(f"""
        <div class='info-box'>
            <h4 style='margin: 0 0 15px 0;'>Health Analytics Summary</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li><strong>Data Types Analyzed:</strong> {summary['data_types']}</li>
                <li><strong>Total Records Processed:</strong> {summary['total_records']:,}</li>
                <li><strong>Anomalies Detected:</strong> {summary['total_anomalies']}</li>
                <li><strong>Processing Frequency:</strong> {target_freq}</li>
                <li><strong>Missing Value Method:</strong> {fill_method}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <p style='color: #6b7280; margin: 0; font-size: 14px;'>
            🏥 <strong>FitPulse Health Analytics Platform</strong> | 
            Complete Dashboard for Real-Time Health Monitoring
        </p>
        <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>
            Powered by Streamlit • Plotly • Scikit-learn | © 2024
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import io
import json

warnings.filterwarnings('ignore')

# Feature Extraction
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

# Time Series Modeling
from prophet import Prophet

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Page Configuration
st.set_page_config(
    page_title="FitPulse Milestone 2 - Feature Extraction & Modeling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(to bottom right, #EEF2FF, #E0E7FF, #DDD6FE);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #6366f1;
    }
    .upload-box {
        border: 2px dashed #6366f1;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background: #f8f9ff;
        transition: all 0.3s;
    }
    .upload-box:hover {
        border-color: #4f46e5;
        background: #eef2ff;
    }
    .step-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #4f46e5;
        font-weight: 700;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Feature Extraction Class
class TSFreshFeatureExtractor:
    """Extract time-series features using TSFresh library"""
    
    def __init__(self, feature_complexity: str = 'minimal'):
        self.feature_complexity = feature_complexity
        self.feature_matrix = None
        self.feature_names = []
        self.extraction_report = {}
        
    def extract_features(self, df: pd.DataFrame, data_type: str, 
                        window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
        """Extract statistical features from time-series data"""
        
        report = {
            'data_type': data_type,
            'original_rows': len(df),
            'window_size': window_size,
            'features_extracted': 0,
            'extraction_time': None,
            'success': False
        }
        
        try:
            # Prepare data for TSFresh
            df_prepared = self._prepare_data_for_tsfresh(df, data_type, window_size)
            
            if df_prepared is None or len(df_prepared) == 0:
                report['error'] = "No data available for feature extraction"
                return pd.DataFrame(), report
            
            # Select feature extraction parameters
            if self.feature_complexity == 'minimal':
                fc_parameters = MinimalFCParameters()
            else:
                fc_parameters = self._get_efficient_parameters()
            
            start_time = datetime.now()
            
            # Extract features
            with st.spinner('🔄 Extracting time-series features...'):
                feature_matrix = extract_features(
                    df_prepared,
                    column_id='window_id',
                    column_sort='timestamp',
                    default_fc_parameters=fc_parameters,
                    disable_progressbar=False,
                    n_jobs=1
                )
            
            # Handle missing values
            feature_matrix = impute(feature_matrix)
            
            # Remove constant features
            feature_matrix = self._remove_constant_features(feature_matrix)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            self.feature_matrix = feature_matrix
            self.feature_names = list(feature_matrix.columns)
            
            report['features_extracted'] = len(self.feature_names)
            report['extraction_time'] = extraction_time
            report['feature_windows'] = len(feature_matrix)
            report['success'] = True
            
            self.extraction_report = report
            
            return feature_matrix, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Feature extraction failed: {str(e)}")
            return pd.DataFrame(), report
    
    def _prepare_data_for_tsfresh(self, df: pd.DataFrame, data_type: str, 
                                  window_size: int) -> pd.DataFrame:
        """Prepare data in TSFresh format with rolling windows"""
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes',
            'stress': 'stress_level',
            'spo2': 'spo2_level'
        }
        
        if data_type not in metric_columns:
            return None
        
        metric_col = metric_columns[data_type]
        
        if metric_col not in df.columns:
            st.warning(f"Metric column '{metric_col}' not found in dataframe")
            return None
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        prepared_data = []
        window_id = 0
        step_size = window_size // 2
        
        for i in range(0, len(df_sorted) - window_size + 1, step_size):
            window_data = df_sorted.iloc[i:i+window_size].copy()
            window_data['window_id'] = window_id
            prepared_data.append(window_data[['window_id', 'timestamp', metric_col]])
            window_id += 1
        
        if not prepared_data:
            return None
        
        df_prepared = pd.concat(prepared_data, ignore_index=True)
        df_prepared = df_prepared.rename(columns={metric_col: 'value'})
        
        return df_prepared
    
    def _get_efficient_parameters(self) -> Dict:
        """Get efficient feature set"""
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "minimum": None,
            "maximum": None,
            "quantile": [{"q": 0.25}, {"q": 0.75}],
            "skewness": None,
            "kurtosis": None,
            "abs_energy": None,
            "absolute_sum_of_changes": None,
        }
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance"""
        constant_features = [col for col in df.columns if df[col].std() == 0]
        if constant_features:
            df = df.drop(columns=constant_features)
        return df
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most variable features"""
        if self.feature_matrix is None or self.feature_matrix.empty:
            return pd.DataFrame()
        
        feature_variance = self.feature_matrix.var().sort_values(ascending=False)
        top_features = feature_variance.head(n)
        
        return pd.DataFrame({
            'Feature': top_features.index,
            'Variance': top_features.values,
            'Mean': [self.feature_matrix[feat].mean() for feat in top_features.index],
            'Std': [self.feature_matrix[feat].std() for feat in top_features.index]
        })


# Prophet Trend Modeler Class
class ProphetTrendModeler:
    """Model time-series trends using Facebook Prophet"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.residuals = {}
        
    def fit_and_predict(self, df: pd.DataFrame, data_type: str,
                       forecast_periods: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """Fit Prophet model and generate predictions"""
        
        report = {
            'data_type': data_type,
            'training_rows': len(df),
            'forecast_periods': forecast_periods,
            'success': False
        }
        
        try:
            metric_columns = {
                'heart_rate': 'heart_rate',
                'steps': 'step_count',
                'sleep': 'duration_minutes',
                'stress': 'stress_level',
                'spo2': 'spo2_level'
            }
            
            if data_type not in metric_columns:
                report['error'] = f"Unknown data type: {data_type}"
                return pd.DataFrame(), report
            
            metric_col = metric_columns[data_type]
            
            if metric_col not in df.columns:
                report['error'] = f"Metric column '{metric_col}' not found"
                return pd.DataFrame(), report
            
            prophet_df = pd.DataFrame({
                'ds': df['timestamp'],
                'y': df[metric_col]
            })
            
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 2:
                report['error'] = "Insufficient data for modeling"
                return pd.DataFrame(), report
            
            with st.spinner('🔄 Training Prophet model...'):
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05,
                    interval_width=0.95
                )
                
                model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=forecast_periods, freq='min')
            forecast = model.predict(future)
            
            merged = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                     on='ds', how='left')
            merged['residual'] = merged['y'] - merged['yhat']
            merged['residual_abs'] = np.abs(merged['residual'])
            
            self.models[data_type] = model
            self.forecasts[data_type] = forecast
            self.residuals[data_type] = merged
            
            report['mae'] = merged['residual_abs'].mean()
            report['rmse'] = np.sqrt((merged['residual'] ** 2).mean())
            report['mape'] = (merged['residual_abs'] / merged['y'].abs()).mean() * 100
            report['success'] = True
            
            return forecast, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Prophet modeling failed: {str(e)}")
            return pd.DataFrame(), report


# Behavior Clusterer Class
class BehaviorClusterer:
    """Cluster behavioral patterns using KMeans and DBSCAN"""
    
    def __init__(self):
        self.scalers = {}
        self.kmeans_models = {}
        self.dbscan_models = {}
        self.cluster_labels = {}
        
    def cluster_features(self, feature_matrix: pd.DataFrame, data_type: str,
                        method: str = 'kmeans', n_clusters: int = 3,
                        eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """Cluster feature vectors to identify behavioral patterns"""
        
        report = {
            'data_type': data_type,
            'method': method,
            'n_samples': len(feature_matrix),
            'n_features': len(feature_matrix.columns),
            'success': False
        }
        
        try:
            if feature_matrix.empty:
                report['error'] = "Empty feature matrix"
                return np.array([]), report
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            self.scalers[data_type] = scaler
            
            with st.spinner(f'🔄 Clustering with {method.upper()}...'):
                if method == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(features_scaled)
                    self.kmeans_models[data_type] = model
                    report['inertia'] = model.inertia_
                    
                elif method == 'dbscan':
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(features_scaled)
                    self.dbscan_models[data_type] = model
                    n_noise = np.sum(labels == -1)
                    report['n_noise_points'] = int(n_noise)
                    report['noise_percentage'] = (n_noise / len(labels)) * 100
            
            self.cluster_labels[data_type] = labels
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                davies_bouldin = davies_bouldin_score(features_scaled, labels)
                
                report['silhouette_score'] = silhouette
                report['davies_bouldin_score'] = davies_bouldin
            
            report['n_clusters'] = len(np.unique(labels))
            report['cluster_sizes'] = {
                int(label): int(count) 
                for label, count in zip(*np.unique(labels, return_counts=True))
            }
            report['success'] = True
            
            return labels, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Clustering failed: {str(e)}")
            return np.array([]), report


def create_sample_data() -> Dict[str, pd.DataFrame]:
    """Create sample data for demonstration"""
    timestamps = pd.date_range(start='2024-01-15 08:00:00', 
                               end='2024-01-15 16:00:00', freq='1min')
    
    # Heart Rate Data
    base_hr = 70
    hr_data = []
    
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        activity_factor = 1.0
        
        if 9 <= time_of_day < 10:
            activity_factor = 1.5
        elif 14 <= time_of_day < 15:
            activity_factor = 1.3
        
        noise = np.random.normal(0, 3)
        hr = base_hr * activity_factor + noise
        hr_data.append(max(50, min(150, hr)))
    
    # Stress Level Data (0-100 scale)
    stress_data = []
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        base_stress = 30
        
        # Higher stress during work hours
        if 10 <= time_of_day < 12:
            base_stress = 60
        elif 13 <= time_of_day < 15:
            base_stress = 55
        
        noise = np.random.normal(0, 5)
        stress = base_stress + noise
        stress_data.append(max(0, min(100, stress)))
    
    # SpO2 Data (95-100% range)
    spo2_data = []
    for i, ts in enumerate(timestamps):
        base_spo2 = 98
        noise = np.random.normal(0, 0.5)
        spo2 = base_spo2 + noise
        spo2_data.append(max(95, min(100, spo2)))
    
    # Steps Data
    step_data = []
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        base_steps = 5
        
        # More steps during active periods
        if 9 <= time_of_day < 10 or 14 <= time_of_day < 15:
            base_steps = 50
        
        noise = np.random.randint(-2, 5)
        steps = base_steps + noise
        step_data.append(max(0, steps))
    
    # Sleep Duration Data (simulated for different time windows)
    sleep_data = []
    for i, ts in enumerate(timestamps):
        # Simulate varying sleep quality
        duration = np.random.randint(60, 120)  # minutes per window
        sleep_data.append(duration)
    
    return {
        'heart_rate': pd.DataFrame({'timestamp': timestamps, 'heart_rate': hr_data}),
        'stress': pd.DataFrame({'timestamp': timestamps, 'stress_level': stress_data}),
        'spo2': pd.DataFrame({'timestamp': timestamps, 'spo2_level': spo2_data}),
        'steps': pd.DataFrame({'timestamp': timestamps, 'step_count': step_data}),
        'sleep': pd.DataFrame({'timestamp': timestamps, 'duration_minutes': sleep_data})
    }


def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>🔬 FitPulse Milestone 2</h1>
        <p style='color: white; font-size: 20px; margin: 10px 0 0 0;'>
            Feature Extraction & Behavioral Modeling
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature badges
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>⚡ TSFresh</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Feature Extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>📈 Prophet</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Trend Modeling</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>🎯 Clustering</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Pattern Discovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>😰 Stress</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Stress Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 18px;'>🫁 SpO2</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Oxygen Levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📁 Data Source")
        use_sample = st.checkbox("Use Sample Data", value=True)
        
        uploaded_files = None
        if not use_sample:
            uploaded_files = st.file_uploader(
                "Upload Data Files (CSV/JSON)",
                type=['csv', 'json'],
                accept_multiple_files=True,
                help="Upload preprocessed data from Milestone 1"
            )
        
        st.divider()
        
        st.subheader("📊 Data Types to Process")
        process_heart_rate = st.checkbox("Heart Rate", value=True)
        process_stress = st.checkbox("Stress Level", value=True)
        process_spo2 = st.checkbox("SpO2 Level", value=True)
        process_steps = st.checkbox("Steps", value=False)
        process_sleep = st.checkbox("Sleep", value=False)
        
        st.divider()
        
        st.subheader("🔧 Feature Extraction")
        window_size = st.slider("Window Size (minutes)", 10, 120, 60, 10)
        
        st.subheader("📊 Prophet Modeling")
        forecast_periods = st.slider("Forecast Periods", 50, 500, 100, 50)
        
        st.subheader("🎯 Clustering")
        clustering_method = st.selectbox("Method", ['kmeans', 'dbscan'])
        
        if clustering_method == 'kmeans':
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            eps, min_samples = None, None
        else:
            n_clusters = None
            eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 5)
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Results Dashboard"])
    
    with tab1:
        # Upload Section
        if not use_sample and uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            for file in uploaded_files:
                st.info(f"📄 {file.name} ({file.size / 1024:.2f} KB)")
        
        # Process Button
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Milestone 2 Pipeline", type="primary", use_container_width=True):
                # Initialize pipeline
                feature_extractor = TSFreshFeatureExtractor(feature_complexity='efficient')
                trend_modeler = ProphetTrendModeler()
                clusterer = BehaviorClusterer()
                
                # Get data
                if use_sample:
                    processed_data = create_sample_data()
                else:
                    if not uploaded_files:
                        st.error("❌ Please upload data files or use sample data")
                        return
                    # Parse uploaded files (implement based on your data format)
                    processed_data = create_sample_data()  # Placeholder
                
                # Filter data types based on user selection
                data_type_filters = {
                    'heart_rate': process_heart_rate,
                    'stress': process_stress,
                    'spo2': process_spo2,
                    'steps': process_steps,
                    'sleep': process_sleep
                }
                
                processed_data = {k: v for k, v in processed_data.items() if data_type_filters.get(k, False)}
                
                # Store results in session state
                st.session_state.results = {}
                
                # Processing pipeline
                for data_type, df in processed_data.items():
                    st.markdown(f"### 📊 Processing {data_type.replace('_', ' ').title()}")
                    
                    # Step 1: Feature Extraction
                    with st.expander("🔵 Step 1: Feature Extraction", expanded=True):
                        feature_matrix, extraction_report = feature_extractor.extract_features(
                            df, data_type, window_size
                        )
                        
                        if not feature_matrix.empty:
                            st.success(f"✅ Extracted {extraction_report['features_extracted']} features from {extraction_report['feature_windows']} windows")
                            
                            top_features = feature_extractor.get_top_features(10)
                            st.dataframe(top_features, use_container_width=True)
                            
                            st.session_state.results[f'{data_type}_features'] = feature_matrix
                            st.session_state.results[f'{data_type}_extraction'] = extraction_report
                    
                    # Step 2: Prophet Modeling
                    with st.expander("🟡 Step 2: Trend Modeling", expanded=True):
                        forecast, modeling_report = trend_modeler.fit_and_predict(
                            df, data_type, forecast_periods
                        )
                        
                        if not forecast.empty:
                            st.success(f"✅ Model trained - MAE: {modeling_report['mae']:.2f}, RMSE: {modeling_report['rmse']:.2f}")
                            
                            st.session_state.results[f'{data_type}_forecast'] = forecast
                            st.session_state.results[f'{data_type}_modeling'] = modeling_report
                    
                    # Step 3: Clustering
                    with st.expander("🟢 Step 3: Clustering", expanded=True):
                        if not feature_matrix.empty:
                            labels, clustering_report = clusterer.cluster_features(
                                feature_matrix, data_type, clustering_method, 
                                n_clusters if n_clusters else 3, eps, min_samples
                            )
                            
                            if len(labels) > 0:
                                st.success(f"✅ Identified {clustering_report['n_clusters']} clusters")
                                
                                if 'silhouette_score' in clustering_report:
                                    col1, col2 = st.columns(2)
                                    col1.metric("Silhouette Score", f"{clustering_report['silhouette_score']:.3f}")
                                    col2.metric("Davies-Bouldin", f"{clustering_report['davies_bouldin_score']:.3f}")
                                
                                st.session_state.results[f'{data_type}_labels'] = labels
                                st.session_state.results[f'{data_type}_clustering'] = clustering_report
                
                st.balloons()
                st.success("🎉 Milestone 2 Pipeline Complete!")
    
    with tab2:
        if 'results' in st.session_state and st.session_state.results:
            st.header("📊 Results Dashboard")
            
            # Get all available data types from results
            available_types = []
            for key in st.session_state.results.keys():
                if '_extraction' in key:
                    data_type = key.replace('_extraction', '')
                    available_types.append(data_type)
            
            if not available_types:
                st.warning("No results available")
                return
            
            # Create dynamic columns based on available data types
            cols = st.columns(min(4, len(available_types)))
            
            for idx, data_type in enumerate(available_types[:4]):
                extraction_report = st.session_state.results.get(f'{data_type}_extraction', {})
                
                with cols[idx]:
                    st.metric(
                        f"{data_type.replace('_', ' ').title()} Features",
                        extraction_report.get('features_extracted', 'N/A'),
                        help=f"Features extracted from {data_type}"
                    )
            
            st.divider()
            
            # Display detailed results for each data type
            for data_type in available_types:
                with st.expander(f"📊 {data_type.replace('_', ' ').title()} Results", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    extraction_report = st.session_state.results.get(f'{data_type}_extraction', {})
                    modeling_report = st.session_state.results.get(f'{data_type}_modeling', {})
                    clustering_report = st.session_state.results.get(f'{data_type}_clustering', {})
                    
                    with col1:
                        st.markdown("**Feature Extraction**")
                        if extraction_report:
                            st.write(f"✅ Features: {extraction_report.get('features_extracted', 'N/A')}")
                            st.write(f"✅ Windows: {extraction_report.get('feature_windows', 'N/A')}")
                            st.write(f"⏱️ Time: {extraction_report.get('extraction_time', 0):.2f}s")
                    
                    with col2:
                        st.markdown("**Prophet Modeling**")
                        if modeling_report:
                            st.write(f"📉 MAE: {modeling_report.get('mae', 0):.2f}")
                            st.write(f"📉 RMSE: {modeling_report.get('rmse', 0):.2f}")
                            st.write(f"📊 MAPE: {modeling_report.get('mape', 0):.2f}%")
                    
                    with col3:
                        st.markdown("**Clustering**")
                        if clustering_report:
                            st.write(f"🎯 Clusters: {clustering_report.get('n_clusters', 'N/A')}")
                            if 'silhouette_score' in clustering_report:
                                st.write(f"📈 Silhouette: {clustering_report.get('silhouette_score', 0):.3f}")
                            if 'davies_bouldin_score' in clustering_report:
                                st.write(f"📊 Davies-Bouldin: {clustering_report.get('davies_bouldin_score', 0):.3f}")
            
            # Summary metrics
            st.divider()
            st.subheader("📈 Overall Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_features = sum(
                st.session_state.results.get(f'{dt}_extraction', {}).get('features_extracted', 0)
                for dt in available_types
            )
            
            total_clusters = sum(
                st.session_state.results.get(f'{dt}_clustering', {}).get('n_clusters', 0)
                for dt in available_types
            )
            
            avg_mae = np.mean([
                st.session_state.results.get(f'{dt}_modeling', {}).get('mae', 0)
                for dt in available_types
                if st.session_state.results.get(f'{dt}_modeling', {}).get('mae')
            ]) if any(st.session_state.results.get(f'{dt}_modeling', {}).get('mae') for dt in available_types) else 0
            
            with col1:
                st.metric("Data Types Processed", len(available_types))
            
            with col2:
                st.metric("Total Features", total_features)
            
            with col3:
                st.metric("Total Clusters", total_clusters)
            
            with col4:
                st.metric("Avg Model MAE", f"{avg_mae:.2f}" if avg_mae > 0 else "N/A")
            
            # Download results
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                results_json = json.dumps({k: str(v) for k, v in st.session_state.results.items()}, indent=2)
                st.download_button(
                    "📥 Download Results (JSON)",
                    results_json,
                    "milestone2_results.json",
                    "application/json",
                    use_container_width=True
                )
            
            st.success("✅ All analyses completed successfully. Ready for Milestone 3: Anomaly Detection!")
            
        else:
            st.info("👈 Run the pipeline from the 'Upload & Process' tab to see results here")


if __name__ == "__main__":
    main()
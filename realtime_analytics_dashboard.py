import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import Dict, List, Any
import sqlite3
from sqlalchemy import create_engine
import redis
from dataclasses import dataclass
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Real-time Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .data-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Data classes for structured data
@dataclass
class KPIMetric:
    name: str
    value: float
    target: float
    trend: str  # 'up', 'down', 'stable'
    change_percent: float

@dataclass
class Alert:
    timestamp: datetime
    severity: str  # 'high', 'medium', 'low'
    message: str
    source: str

# Initialize session state
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'kpi_metrics' not in st.session_state:
    st.session_state.kpi_metrics = {}
if 'streaming_active' not in st.session_state:
    st.session_state.streaming_active = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = generate_historical_data()

class RealTimeDataGenerator:
    """Simulates real-time data generation"""
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.is_running = False
        self.thread = None
    
    def start_streaming(self):
        """Start the data streaming thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._generate_data)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_streaming(self):
        """Stop the data streaming"""
        self.is_running = False
        if self.thread:
            self.thread.join()
    
    def _generate_data(self):
        """Generate simulated real-time data"""
        while self.is_running:
            timestamp = datetime.now()
            
            # Simulate business metrics
            data_point = {
                'timestamp': timestamp,
                'sales': np.random.normal(1000, 200),
                'users_online': np.random.poisson(500),
                'conversion_rate': np.random.uniform(0.02, 0.08),
                'server_cpu': np.random.uniform(20, 95),
                'server_memory': np.random.uniform(30, 90),
                'error_rate': np.random.exponential(0.001),
                'response_time': np.random.gamma(2, 50)
            }
            
            self.data_queue.put(data_point)
            time.sleep(1)  # Generate data every second
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the latest data point"""
        if not self.data_queue.empty():
            return self.data_queue.get()
        return None

def generate_historical_data(days: int = 30) -> pd.DataFrame:
    """Generate historical data for the dashboard"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    # Create realistic business data with trends and seasonality
    base_sales = 1000
    trend = np.linspace(0, 200, len(dates))  # Growing trend
    seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily seasonality
    noise = np.random.normal(0, 50, len(dates))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'sales': base_sales + trend + seasonal + noise,
        'users_online': np.random.poisson(500, len(dates)),
        'conversion_rate': np.random.normal(0.05, 0.01, len(dates)),
        'server_cpu': np.random.uniform(20, 80, len(dates)),
        'server_memory': np.random.uniform(30, 70, len(dates)),
        'error_rate': np.random.exponential(0.001, len(dates)),
        'response_time': np.random.gamma(2, 50, len(dates))
    })
    
    return data

def calculate_kpis(data: pd.DataFrame) -> Dict[str, KPIMetric]:
    """Calculate KPI metrics from data"""
    if data.empty:
        return {}
    
    # Calculate current vs previous period
    current_period = data.tail(24)  # Last 24 hours
    previous_period = data.tail(48).head(24)  # Previous 24 hours
    
    kpis = {}
    
    # Sales KPI
    current_sales = current_period['sales'].sum()
    previous_sales = previous_period['sales'].sum()
    sales_change = ((current_sales - previous_sales) / previous_sales) * 100
    
    kpis['sales'] = KPIMetric(
        name="Total Sales",
        value=current_sales,
        target=25000,  # Daily target
        trend='up' if sales_change > 0 else 'down',
        change_percent=sales_change
    )
    
    # Users Online KPI
    current_users = current_period['users_online'].mean()
    previous_users = previous_period['users_online'].mean()
    users_change = ((current_users - previous_users) / previous_users) * 100
    
    kpis['users'] = KPIMetric(
        name="Avg Users Online",
        value=current_users,
        target=600,
        trend='up' if users_change > 0 else 'down',
        change_percent=users_change
    )
    
    # Conversion Rate KPI
    current_conversion = current_period['conversion_rate'].mean()
    previous_conversion = previous_period['conversion_rate'].mean()
    conversion_change = ((current_conversion - previous_conversion) / previous_conversion) * 100
    
    kpis['conversion'] = KPIMetric(
        name="Conversion Rate",
        value=current_conversion * 100,  # Convert to percentage
        target=5.0,
        trend='up' if conversion_change > 0 else 'down',
        change_percent=conversion_change
    )
    
    # Server Performance KPI
    current_cpu = current_period['server_cpu'].mean()
    cpu_status = 'up' if current_cpu < 70 else 'down'
    
    kpis['server_cpu'] = KPIMetric(
        name="Server CPU",
        value=current_cpu,
        target=70,
        trend=cpu_status,
        change_percent=0  # Simplified for demo
    )
    
    return kpis

def detect_anomalies(data: pd.DataFrame) -> List[Alert]:
    """Detect anomalies and generate alerts"""
    alerts = []
    
    if data.empty:
        return alerts
    
    latest = data.tail(1).iloc[0]
    
    # CPU usage alert
    if latest['server_cpu'] > 90:
        alerts.append(Alert(
            timestamp=latest['timestamp'],
            severity='high',
            message=f"High CPU usage: {latest['server_cpu']:.1f}%",
            source='System Monitor'
        ))
    elif latest['server_cpu'] > 75:
        alerts.append(Alert(
            timestamp=latest['timestamp'],
            severity='medium',
            message=f"Elevated CPU usage: {latest['server_cpu']:.1f}%",
            source='System Monitor'
        ))
    
    # Error rate alert
    if latest['error_rate'] > 0.01:
        alerts.append(Alert(
            timestamp=latest['timestamp'],
            severity='high',
            message=f"High error rate: {latest['error_rate']:.3f}%",
            source='Application Monitor'
        ))
    
    # Response time alert
    if latest['response_time'] > 200:
        alerts.append(Alert(
            timestamp=latest['timestamp'],
            severity='medium',
            message=f"Slow response time: {latest['response_time']:.1f}ms",
            source='Performance Monitor'
        ))
    
    return alerts

def main():
    st.markdown('<h1 class="main-header">📊 Real-time Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Business Intelligence & Performance Monitoring")
    
    # Initialize data generator
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = RealTimeDataGenerator()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Streaming controls
        st.subheader("📡 Data Streaming")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Stream"):
                st.session_state.data_generator.start_streaming()
                st.session_state.streaming_active = True
                st.success("Streaming started!")
        
        with col2:
            if st.button("⏹️ Stop Stream"):
                st.session_state.data_generator.stop_streaming()
                st.session_state.streaming_active = False
                st.info("Streaming stopped!")
        
        # Display streaming status
        status_color = "🟢" if st.session_state.streaming_active else "🔴"
        st.markdown(f"**Status:** {status_color} {'Active' if st.session_state.streaming_active else 'Inactive'}")
        
        # Time range selector
        st.subheader("📅 Time Range")
        time_range = st.selectbox(
            "Select time range",
            ["Last 24 hours", "Last 7 days", "Last 30 days"],
            index=0
        )
        
        # Refresh controls
        st.subheader("🔄 Refresh Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 2)
        
        # Alert settings
        st.subheader("🚨 Alert Settings")
        alert_threshold_cpu = st.slider("CPU Alert Threshold (%)", 50, 100, 80)
        alert_threshold_memory = st.slider("Memory Alert Threshold (%)", 50, 100, 85)
        
        # Data export
        st.subheader("💾 Export Data")
        if st.button("📊 Export CSV"):
            csv = st.session_state.historical_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Get latest data if streaming
    if st.session_state.streaming_active:
        latest_data = st.session_state.data_generator.get_latest_data()
        if latest_data:
            # Add to historical data
            new_row = pd.DataFrame([latest_data])
            st.session_state.historical_data = pd.concat([
                st.session_state.historical_data, new_row
            ], ignore_index=True)
            
            # Keep only last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            st.session_state.historical_data = st.session_state.historical_data[
                st.session_state.historical_data['timestamp'] >= cutoff_date
            ]
    
    # Calculate KPIs
    st.session_state.kpi_metrics = calculate_kpis(st.session_state.historical_data)
    
    # Detect alerts
    new_alerts = detect_anomalies(st.session_state.historical_data)
    for alert in new_alerts:
        if alert not in st.session_state.alerts:
            st.session_state.alerts.append(alert)
    
    # Keep only recent alerts (last 24 hours)
    cutoff_time = datetime.now() - timedelta(hours=24)
    st.session_state.alerts = [
        alert for alert in st.session_state.alerts 
        if alert.timestamp >= cutoff_time
    ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Real-time Metrics", 
        "🚨 Alerts", 
        "📉 Trends", 
        "🔍 Deep Dive"
    ])
    
    with tab1:
        overview_dashboard()
    
    with tab2:
        realtime_metrics_dashboard()
    
    with tab3:
        alerts_dashboard()
    
    with tab4:
        trends_dashboard()
    
    with tab5:
        deep_dive_dashboard()
    
    # Auto-refresh functionality
    if auto_refresh and st.session_state.streaming_active:
        time.sleep(refresh_interval)
        st.rerun()

def overview_dashboard():
    """Main overview dashboard"""
    st.header("📊 Business Overview")
    
    # KPI Cards
    if st.session_state.kpi_metrics:
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            kpi = st.session_state.kpi_metrics.get('sales')
            if kpi:
                trend_icon = "📈" if kpi.trend == 'up' else "📉"
                st.markdown(f'''
                <div class="kpi-card">
                    <h3>{kpi.name}</h3>
                    <h2>${kpi.value:,.0f}</h2>
                    <p>{trend_icon} {kpi.change_percent:+.1f}%</p>
                    <small>Target: ${kpi.target:,.0f}</small>
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            kpi = st.session_state.kpi_metrics.get('users')
            if kpi:
                trend_icon = "📈" if kpi.trend == 'up' else "📉"
                st.markdown(f'''
                <div class="kpi-card">
                    <h3>{kpi.name}</h3>
                    <h2>{kpi.value:,.0f}</h2>
                    <p>{trend_icon} {kpi.change_percent:+.1f}%</p>
                    <small>Target: {kpi.target:,.0f}</small>
                </div>
                ''', unsafe_allow_html=True)
        
        with col3:
            kpi = st.session_state.kpi_metrics.get('conversion')
            if kpi:
                trend_icon = "📈" if kpi.trend == 'up' else "📉"
                st.markdown(f'''
                <div class="kpi-card">
                    <h3>{kpi.name}</h3>
                    <h2>{kpi.value:.2f}%</h2>
                    <p>{trend_icon} {kpi.change_percent:+.1f}%</p>
                    <small>Target: {kpi.target:.1f}%</small>
                </div>
                ''', unsafe_allow_html=True)
        
        with col4:
            kpi = st.session_state.kpi_metrics.get('server_cpu')
            if kpi:
                status_icon = "🟢" if kpi.value < 70 else "🟡" if kpi.value < 85 else "🔴"
                st.markdown(f'''
                <div class="kpi-card">
                    <h3>{kpi.name}</h3>
                    <h2>{kpi.value:.1f}%</h2>
                    <p>{status_icon} System Status</p>
                    <small>Threshold: {kpi.target:.0f}%</small>
                </div>
                ''', unsafe_allow_html=True)
    
    # Quick charts
    st.subheader("📈 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend
        recent_data = st.session_state.historical_data.tail(168)  # Last 7 days
        fig = px.line(
            recent_data, 
            x='timestamp', 
            y='sales',
            title="Sales Trend (Last 7 Days)",
            labels={'sales': 'Sales ($)', 'timestamp': 'Time'}
        )
        fig.update_traces(line_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Users online
        fig = px.area(
            recent_data, 
            x='timestamp', 
            y='users_online',
            title="Users Online (Last 7 Days)",
            labels={'users_online': 'Users Online', 'timestamp': 'Time'}
        )
        fig.update_traces(fill='tonexty', fillcolor='rgba(102, 126, 234, 0.3)')
        st.plotly_chart(fig, use_container_width=True)

def realtime_metrics_dashboard():
    """Real-time metrics dashboard"""
    st.header("📈 Real-time Metrics")
    
    if st.session_state.historical_data.empty:
        st.warning("No data available. Please start data streaming.")
        return
    
    # Latest metrics
    latest_data = st.session_state.historical_data.tail(1).iloc[0]
    
    st.subheader("🔴 Live Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Sales",
            f"${latest_data['sales']:,.0f}",
            delta=f"{np.random.uniform(-100, 100):+.0f}"
        )
    
    with col2:
        st.metric(
            "Users Online",
            f"{latest_data['users_online']:,.0f}",
            delta=f"{np.random.randint(-20, 20):+d}"
        )
    
    with col3:
        st.metric(
            "Conversion Rate",
            f"{latest_data['conversion_rate']*100:.2f}%",
            delta=f"{np.random.uniform(-0.5, 0.5):+.2f}%"
        )
    
    with col4:
        st.metric(
            "Server CPU",
            f"{latest_data['server_cpu']:.1f}%",
            delta=f"{np.random.uniform(-5, 5):+.1f}%"
        )
    
    # Real-time charts
    st.subheader("📊 Live Charts")
    
    # Get recent data for real-time view
    recent_data = st.session_state.historical_data.tail(60)  # Last 60 data points
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Real-time line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['sales'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Real-time Sales",
            xaxis_title="Time",
            yaxis_title="Sales ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # System metrics gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_data['server_cpu'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Server CPU Usage"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def alerts_dashboard():
    """Alerts and notifications dashboard"""
    st.header("🚨 Alerts & Notifications")
    
    if not st.session_state.alerts:
        st.success("🎉 No active alerts!")
        return
    
    # Alert summary
    high_alerts = len([a for a in st.session_state.alerts if a.severity == 'high'])
    medium_alerts = len([a for a in st.session_state.alerts if a.severity == 'medium'])
    low_alerts = len([a for a in st.session_state.alerts if a.severity == 'low'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 High Priority", high_alerts)
    
    with col2:
        st.metric("🟡 Medium Priority", medium_alerts)
    
    with col3:
        st.metric("🟢 Low Priority", low_alerts)
    
    # Alert list
    st.subheader("📋 Recent Alerts")
    
    # Sort alerts by timestamp (newest first)
    sorted_alerts = sorted(st.session_state.alerts, key=lambda x: x.timestamp, reverse=True)
    
    for alert in sorted_alerts[:10]:  # Show last 10 alerts
        alert_class = f"alert-{alert.severity}"
        
        st.markdown(f'''
        <div class="{alert_class}">
            <strong>{alert.severity.upper()}</strong> - {alert.message}<br>
            <small>{alert.source} | {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Alert trends
    st.subheader("📊 Alert Trends")
    
    if st.session_state.alerts:
        # Create alert timeline
        alert_df = pd.DataFrame([
            {
                'timestamp': alert.timestamp,
                'severity': alert.severity,
                'count': 1
            }
            for alert in st.session_state.alerts
        ])
        
        # Group by hour and severity
        alert_df['hour'] = alert_df['timestamp'].dt.floor('H')
        alert_summary = alert_df.groupby(['hour', 'severity']).sum().reset_index()
        
        fig = px.bar(
            alert_summary,
            x='hour',
            y='count',
            color='severity',
            color_discrete_map={'high': '#ff4444', 'medium': '#ffaa00', 'low': '#44ff44'},
            title="Alert Distribution by Hour"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def trends_dashboard():
    """Trends and historical analysis dashboard"""
    st.header("📉 Trends & Historical Analysis")
    
    if st.session_state.historical_data.empty:
        st.warning("No historical data available.")
        return
    
    # Time series analysis
    st.subheader("📈 Time Series Analysis")
    
    # Metric selection
    metrics = ['sales', 'users_online', 'conversion_rate', 'server_cpu', 'server_memory']
    selected_metrics = st.multiselect(
        "Select metrics to analyze",
        metrics,
        default=['sales', 'users_online']
    )
    
    if selected_metrics:
        # Create subplot
        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            subplot_titles=selected_metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.historical_data['timestamp'],
                    y=st.session_state.historical_data[metric],
                    mode='lines',
                    name=metric,
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(selected_metrics),
            title_text="Historical Trends",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = st.session_state.historical_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = st.session_state.historical_data[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(
        st.session_state.historical_data[numeric_cols].describe(),
        use_container_width=True
    )

def deep_dive_dashboard():
    """Deep dive analysis dashboard"""
    st.header("🔍 Deep Dive Analysis")
    
    if st.session_state.historical_data.empty:
        st.warning("No data available for analysis.")
        return
    
    # Advanced filtering
    st.subheader("🎛️ Advanced Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(
                (datetime.now() - timedelta(days=7)).date(),
                datetime.now().date()
            )
        )
    
    with col2:
        # Metric threshold filters
        cpu_range = st.slider(
            "CPU Usage Range (%)",
            0, 100, (0, 100)
        )
    
    with col3:
        # Sales range filter
        sales_range = st.slider(
            "Sales Range ($)",
            0, int(st.session_state.historical_data['sales'].max()),
            (0, int(st.session_state.historical_data['sales'].max()))
        )
    
    # Apply filters
    filtered_data = st.session_state.historical_data.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['timestamp'].dt.date >= start_date) &
            (filtered_data['timestamp'].dt.date <= end_date)
        ]
    
    filtered_data = filtered_data[
        (filtered_data['server_cpu'] >= cpu_range[0]) &
        (filtered_data['server_cpu'] <= cpu_range[1]) &
        (filtered_data['sales'] >= sales_range[0]) &
        (filtered_data['sales'] <= sales_range[1])
    ]
    
    # Display filtered data stats
    st.subheader("📊 Filtered Data Analysis")
    st.write(f"**Filtered records:** {len(filtered_data)} out of {len(st.session_state.historical_data)}")
    
    if not filtered_data.empty:
        # Advanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with regression line
            fig = px.scatter(
                filtered_data,
                x='users_online',
                y='sales',
                size='conversion_rate',
                color='server_cpu',
                title="Sales vs Users Online",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot for performance metrics
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=filtered_data['server_cpu'],
                name='CPU Usage',
                boxpoints='outliers'
            ))
            
            fig.add_trace(go.Box(
                y=filtered_data['server_memory'],
                name='Memory Usage',
                boxpoints='outliers'
            ))
            
            fig.update_layout(
                title="System Performance Distribution",
                yaxis_title="Usage (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced metrics table
        st.subheader("📋 Detailed Metrics Table")
        
        # Add derived columns
        filtered_data['hour'] = filtered_data['timestamp'].dt.hour
        filtered_data['day_of_week'] = filtered_data['timestamp'].dt.day_name()
        
        # Display with formatting
        display_data = filtered_data.tail(100).copy()
        display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_data['sales'] = display_data['sales'].apply(lambda x: f"${x:,.2f}")
        display_data['conversion_rate'] = display_data['conversion_rate'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(display_data, use_container_width=True)
    
    else:
        st.warning("No data matches the selected filters.")

if __name__ == "__main__":
    main() 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import io
import base64
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Advanced ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .model-comparison {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def main():
    st.markdown('<h1 class="main-header">🤖 Advanced ML Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Model Training, Feature Engineering & Interactive Analytics")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Dataset selection
        dataset_option = st.selectbox(
            "Choose Dataset",
            ["Upload Custom", "Sample Dataset", "Generate Synthetic"]
        )
        
        # Model selection
        st.header("🎯 Model Selection")
        models = st.multiselect(
            "Select Models to Train",
            ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
            default=["Random Forest", "Gradient Boosting"]
        )
        
        # Advanced options
        st.header("⚙️ Advanced Options")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", value=42)
        
        # Real-time features
        auto_retrain = st.checkbox("Auto-retrain on data changes")
        show_advanced_metrics = st.checkbox("Show Advanced Metrics", True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer", 
        "🔧 Feature Engineering", 
        "🎯 Model Training", 
        "📈 Performance Analysis", 
        "🚀 Model Deployment"
    ])
    
    # Load or generate dataset
    data = load_dataset(dataset_option)
    
    if data is not None:
        with tab1:
            data_explorer_interface(data)
        
        with tab2:
            data = feature_engineering_interface(data)
        
        with tab3:
            model_training_interface(data, models, test_size, random_state)
        
        with tab4:
            performance_analysis_interface()
        
        with tab5:
            model_deployment_interface()

def load_dataset(option):
    """Load dataset based on user selection"""
    if option == "Upload Custom":
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload a CSV file with your dataset"
        )
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.success(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
    
    elif option == "Sample Dataset":
        # Generate sample regression dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1.5, n_samples),
            'feature_3': np.random.exponential(2, n_samples),
            'feature_4': np.random.uniform(-2, 2, n_samples),
            'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_2': np.random.choice(['X', 'Y'], n_samples)
        })
        
        # Create target variable with some relationship
        data['target'] = (
            2 * data['feature_1'] + 
            1.5 * data['feature_2'] - 
            0.8 * data['feature_3'] + 
            np.random.normal(0, 0.5, n_samples)
        )
        
        st.success("Generated sample dataset with 1000 rows and 7 columns")
        return data
    
    elif option == "Generate Synthetic":
        return generate_synthetic_dataset()
    
    return None

def generate_synthetic_dataset():
    """Generate synthetic dataset with user parameters"""
    st.subheader("🎲 Synthetic Data Generator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.number_input("Number of Samples", 100, 10000, 1000)
        n_features = st.number_input("Number of Features", 2, 20, 5)
    
    with col2:
        noise_level = st.slider("Noise Level", 0.0, 2.0, 0.1)
        correlation = st.slider("Feature Correlation", -1.0, 1.0, 0.0)
    
    with col3:
        target_type = st.selectbox("Target Type", ["Regression", "Classification"])
        if st.button("Generate Dataset"):
            if target_type == "Regression":
                from sklearn.datasets import make_regression
                X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=noise_level,
    random_state=42
)
                data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
                data['target'] = y
                return data
    
    return None

def data_explorer_interface(data):
    """Interface for data exploration"""
    st.header("📊 Data Explorer")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{data.shape[0]}</h3><p>Rows</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{data.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
    
    with col3:
        missing = data.isnull().sum().sum()
        st.markdown(f'<div class="metric-card"><h3>{missing}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
    
    with col4:
        duplicates = data.duplicated().sum()
        st.markdown(f'<div class="metric-card"><h3>{duplicates}</h3><p>Duplicates</p></div>', unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(100), use_container_width=True)
    
    # Interactive visualizations
    st.subheader("📈 Interactive Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y", index=1)
            
            fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

def feature_engineering_interface(data):
    """Interface for feature engineering"""
    st.header("🔧 Feature Engineering")
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    all_columns = data.columns.tolist()
    target_col = st.selectbox("Select Target Column", all_columns)
    feature_cols = st.multiselect(
        "Select Feature Columns", 
        [col for col in all_columns if col != target_col],
        default=[col for col in all_columns if col != target_col]
    )
    
    # Feature transformations
    st.subheader("🔄 Feature Transformations")
    
    transformations = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scaling options
        scaling_method = st.selectbox(
            "Scaling Method",
            ["None", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
        )
        
        # Polynomial features
        poly_degree = st.selectbox("Polynomial Degree", [1, 2, 3], index=0)
    
    with col2:
        # Encoding for categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            encoding_method = st.selectbox(
                "Categorical Encoding",
                ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
            )
    
    # Apply transformations
    if st.button("Apply Feature Engineering"):
        processed_data = apply_feature_engineering(
            data, feature_cols, target_col, scaling_method, 
            poly_degree, encoding_method if categorical_cols else None
        )
        
        st.success("Feature engineering applied successfully!")
        st.subheader("🎉 Processed Dataset")
        st.dataframe(processed_data.head(), use_container_width=True)
        
        return processed_data
    
    return data

def apply_feature_engineering(data, feature_cols, target_col, scaling_method, poly_degree, encoding_method):
    """Apply selected feature engineering techniques"""
    processed_data = data.copy()
    
    # Handle categorical variables
    categorical_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols and encoding_method:
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in categorical_cols:
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        elif encoding_method == "One-Hot Encoding":
            processed_data = pd.get_dummies(processed_data, columns=categorical_cols)
    
    # Feature scaling
    if scaling_method != "None":
        scaler = StandardScaler()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
    
    return processed_data

def model_training_interface(data, models, test_size, random_state):
    """Interface for model training"""
    st.header("🎯 Model Training")
    
    if data is None or data.empty:
        st.warning("Please load a dataset first!")
        return
    
    # Target selection
    target_col = st.selectbox("Select Target Column", data.columns.tolist())
    feature_cols = [col for col in data.columns if col != target_col]
    
    if not feature_cols:
        st.error("No feature columns available!")
        return
    
    # Prepare data
    X = data[feature_cols]
    y = data[target_col]
    
    # Handle categorical variables in features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Model training
    if st.button("🚀 Train Models", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        for i, model_name in enumerate(models):
            status_text.text(f"Training {model_name}...")
            
            # Select model
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'training_time': training_time,
                'predictions': y_pred
            }
            
            # Store in session state
            st.session_state.trained_models[model_name] = results[model_name]
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                st.session_state.feature_importance[model_name] = importance_df
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text("Training completed!")
        st.success("🎉 All models trained successfully!")
        
        # Display results
        display_training_results(results, y_test)

def display_training_results(results, y_test):
    """Display training results"""
    st.subheader("📊 Training Results")
    
    # Metrics comparison
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'R² Score': f"{result['r2']:.4f}",
            'MSE': f"{result['mse']:.4f}",
            'MAE': f"{result['mae']:.4f}",
            'Training Time (s)': f"{result['training_time']:.2f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig = px.bar(
            metrics_df, 
            x='Model', 
            y='R² Score',
            title="Model Performance Comparison (R² Score)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Actual vs Predicted for best model
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        y_pred_best = results[best_model]['predictions']
        
        fig = px.scatter(
            x=y_test, 
            y=y_pred_best,
            title=f"Actual vs Predicted ({best_model})",
            labels={'x': 'Actual', 'y': 'Predicted'}
        )
        fig.add_shape(
            type="line",
            x0=min(y_test), y0=min(y_test),
            x1=max(y_test), y1=max(y_test),
            line=dict(dash="dash", color="red")
        )
        st.plotly_chart(fig, use_container_width=True)

def performance_analysis_interface():
    """Interface for performance analysis"""
    st.header("📈 Performance Analysis")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models first!")
        return
    
    # Model selection for detailed analysis
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(st.session_state.trained_models.keys())
    )
    
    if selected_model:
        model_data = st.session_state.trained_models[selected_model]
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="model-comparison"><h3>{model_data["r2"]:.4f}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="model-comparison"><h3>{model_data["mse"]:.4f}</h3><p>MSE</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="model-comparison"><h3>{model_data["mae"]:.4f}</h3><p>MAE</p></div>', unsafe_allow_html=True)
        
        # Feature importance
        if selected_model in st.session_state.feature_importance:
            st.subheader("🎯 Feature Importance")
            importance_df = st.session_state.feature_importance[selected_model]
            
            fig = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

def model_deployment_interface():
    """Interface for model deployment"""
    st.header("🚀 Model Deployment")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models first!")
        return
    
    # Model selection for deployment
    deploy_model = st.selectbox(
        "Select Model to Deploy",
        list(st.session_state.trained_models.keys())
    )
    
    if deploy_model:
        st.subheader("💾 Model Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Download Model"):
                model = st.session_state.trained_models[deploy_model]['model']
                
                # Save model to bytes
                model_bytes = io.BytesIO()
                joblib.dump(model, model_bytes)
                model_bytes.seek(0)
                
                st.download_button(
                    label="💾 Download Trained Model",
                    data=model_bytes.getvalue(),
                    file_name=f"{deploy_model.lower().replace(' ', '_')}_model.joblib",
                    mime="application/octet-stream"
                )
        
        with col2:
            if st.button("📊 Generate Report"):
                generate_model_report(deploy_model)
        
        # API simulation
        st.subheader("🔌 API Simulation")
        st.info("This section would integrate with FastAPI or Flask for production deployment")
        
        # Sample prediction interface
        st.subheader("🎯 Make Predictions")
        st.write("Enter feature values for prediction:")
        
        # This would be dynamically generated based on trained features
        prediction_input = {}
        for i in range(3):  # Simplified for demo
            prediction_input[f'feature_{i+1}'] = st.number_input(f"Feature {i+1}", value=0.0)
        
        if st.button("🔮 Predict"):
            # Simulate prediction
            result = np.random.normal(10, 2)  # Placeholder
            st.success(f"Predicted value: {result:.4f}")

def generate_model_report(model_name):
    """Generate comprehensive model report"""
    model_data = st.session_state.trained_models[model_name]
    
    report = f"""
# Model Performance Report
**Model:** {model_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
- **R² Score:** {model_data['r2']:.4f}
- **Mean Squared Error:** {model_data['mse']:.4f}
- **Mean Absolute Error:** {model_data['mae']:.4f}
- **Training Time:** {model_data['training_time']:.2f} seconds

## Model Characteristics
- **Type:** Regression Model
- **Algorithm:** {model_name}
- **Status:** Production Ready

## Recommendations
Based on the performance metrics, this model shows {'excellent' if model_data['r2'] > 0.8 else 'acceptable' if model_data['r2'] > 0.6 else 'poor'} performance.
"""
    
    st.markdown(report)
    
    # Download report
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"{model_name}_report.md",
        mime="text/markdown"
    )

if __name__ == "__main__":
    main() 
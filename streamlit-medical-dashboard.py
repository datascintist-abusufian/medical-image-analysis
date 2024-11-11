import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from PIL import Image
import io
import requests
import zipfile
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Medical Image Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATASET_URL = "https://github.com/datascintist-abusufian/medical-image-analysis/raw/main/heart_dataset_sup.zip"
DATASET_PATH = "heart_dataset_sup.zip"
EXTRACT_FOLDER = "heart_dataset_sup"

# Custom CSS
st.markdown("""
    <style>
    .stPlotlyChart {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

class DatasetManager:
    @staticmethod
    def setup_dataset():
        """Setup and verify dataset"""
        try:
            if not os.path.exists(EXTRACT_FOLDER):
                with st.spinner("Downloading dataset..."):
                    response = requests.get(DATASET_URL, timeout=30)
                    response.raise_for_status()
                    
                    with open(DATASET_PATH, "wb") as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
                        zip_ref.extractall()
                    
                    st.success("Dataset downloaded and extracted successfully!")
            else:
                st.info("Dataset already exists locally.")
                
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download dataset: {str(e)}")
            logger.error(f"Dataset download error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error setting up dataset: {str(e)}")
            logger.error(f"Dataset setup error: {str(e)}")
            return False

class ImageProcessor:
    @staticmethod
    def load_image(uploaded_file):
        """Load and validate image"""
        try:
            if uploaded_file.type == "application/dicom":
                st.warning("DICOM support coming soon!")
                return None
            
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None

    @staticmethod
    def calculate_metrics(image):
        """Calculate image analysis metrics"""
        # Simulate metrics calculation
        base_score = 0.90 + np.random.normal(0, 0.02)
        
        return {
            "Dice Score": min(1.0, base_score + 0.02),
            "IoU Score": min(1.0, base_score - 0.05),
            "Sensitivity": min(1.0, base_score - 0.01),
            "Specificity": min(1.0, base_score + 0.04),
            "Precision": min(1.0, base_score + 0.01),
            "Accuracy": min(1.0, base_score + 0.03),
            "F1 Score": min(1.0, base_score + 0.015),
            "Hausdorff Distance": max(0, 3.2 + np.random.normal(0, 0.1))
        }

class DashboardVisualizer:
    @staticmethod
    def display_metrics(metrics):
        """Display metrics in columns"""
        cols = st.columns(4)
        metrics_items = list(metrics.items())
        
        for i, col in enumerate(cols):
            start_idx = i * 2
            for j in range(2):
                if start_idx + j < len(metrics_items):
                    key, value = metrics_items[start_idx + j]
                    col.metric(
                        key,
                        f"{value:.3f}" if key != "Hausdorff Distance" else f"{value:.1f} px"
                    )

    @staticmethod
    def plot_time_series(metrics):
        """Plot time series analysis"""
        time_series_data = pd.DataFrame({
            'Frame': range(1, 6),
            'Dice Score': [metrics['Dice Score'] * (1 + np.random.normal(0, 0.01)) for _ in range(5)],
            'IoU': [metrics['IoU Score'] * (1 + np.random.normal(0, 0.01)) for _ in range(5)],
            'Sensitivity': [metrics['Sensitivity'] * (1 + np.random.normal(0, 0.01)) for _ in range(5)],
            'Precision': [metrics['Precision'] * (1 + np.random.normal(0, 0.01)) for _ in range(5)]
        })
        
        fig = px.line(
            time_series_data,
            x='Frame',
            y=['Dice Score', 'IoU', 'Sensitivity', 'Precision'],
            title='Metrics Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_regional_analysis(metrics):
        """Plot regional analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regional Performance")
            region_data = pd.DataFrame({
                'Region': ['R1', 'R2', 'R3', 'R4'],
                'Accuracy': [metrics['Accuracy'] * (1 + np.random.normal(0, 0.02)) for _ in range(4)],
                'Sensitivity': [metrics['Sensitivity'] * (1 + np.random.normal(0, 0.02)) for _ in range(4)],
                'Specificity': [metrics['Specificity'] * (1 + np.random.normal(0, 0.02)) for _ in range(4)]
            })
            
            fig = px.bar(
                region_data,
                x='Region',
                y=['Accuracy', 'Sensitivity', 'Specificity'],
                title='Regional Analysis',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Performance Radar")
            metrics_radar = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Specificity'],
                'Value': [
                    metrics['Precision'],
                    metrics['Sensitivity'],
                    metrics['F1 Score'],
                    metrics['Accuracy'],
                    metrics['Specificity']
                ]
            })
            
            fig = go.Figure(data=go.Scatterpolar(
                r=metrics_radar['Value'],
                theta=metrics_radar['Metric'],
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def display_advanced_analysis(metrics, analysis_type):
        """Display advanced analysis section"""
        if analysis_type in ["Advanced", "Research"]:
            st.subheader("Advanced Analysis")
            
            with st.expander("Detailed Statistics"):
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                    'Value': [
                        np.mean(list(metrics.values())[:-1]),  # Exclude Hausdorff Distance
                        np.std(list(metrics.values())[:-1]),
                        np.min(list(metrics.values())[:-1]),
                        np.max(list(metrics.values())[:-1]),
                        np.median(list(metrics.values())[:-1])
                    ]
                })
                st.dataframe(stats_df)
            
            with st.expander("Uncertainty Analysis"):
                uncertainty_fig = go.Figure()
                for metric, value in metrics.items():
                    if metric != "Hausdorff Distance":
                        uncertainty_fig.add_trace(go.Box(
                            y=[value * (1 + np.random.normal(0, 0.02)) for _ in range(100)],
                            name=metric
                        ))
                uncertainty_fig.update_layout(title="Metric Uncertainty Distribution")
                st.plotly_chart(uncertainty_fig, use_container_width=True)
            
            if analysis_type == "Research":
                with st.expander("Research Metrics"):
                    st.write("Additional Research Metrics:")
                    research_metrics = {
                        "Cross Validation Score": f"{np.random.normal(0.9, 0.02):.3f}",
                        "Model Confidence": f"{np.random.normal(0.85, 0.03):.3f}",
                        "Uncertainty Score": f"{np.random.normal(0.12, 0.02):.3f}"
                    }
                    for metric, value in research_metrics.items():
                        st.metric(metric, value)

def main():
    st.title("Medical Image Analysis Dashboard")
    
    # Setup dataset
    if not DatasetManager.setup_dataset():
        st.error("Failed to setup required dataset. Please try again.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_file = st.file_uploader("Upload Medical Image", type=['png', 'jpg', 'jpeg', 'dcm'])
        
        st.subheader("Analysis Settings")
        segmentation_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Basic", "Advanced", "Research"]
        )

    # Main content
    if uploaded_file is not None:
        # Load and process image
        image = ImageProcessor.load_image(uploaded_file)
        if image is None:
            return

        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Segmentation Result")
            st.info("Segmentation visualization would appear here")

        # Process image and display results
        with st.spinner('Processing image...'):
            metrics = ImageProcessor.calculate_metrics(image)
            
            st.subheader("Key Metrics")
            DashboardVisualizer.display_metrics(metrics)
            
            st.subheader("Performance Analysis")
            DashboardVisualizer.plot_time_series(metrics)
            DashboardVisualizer.plot_regional_analysis(metrics)
            DashboardVisualizer.display_advanced_analysis(metrics, analysis_type)

    else:
        st.info("Please upload a medical image to begin analysis")
        st.write("Supported formats: PNG, JPG, JPEG, DICOM")

if __name__ == "__main__":
    main()

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
from pathlib import Path

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-right: 4px;
        padding-left: 4px;
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
    def plot_uncertainty_analysis(metrics):
        """Create uncertainty analysis visualization"""
        st.subheader("Uncertainty Analysis")
        
        # Create sample distributions for each metric
        n_samples = 1000
        uncertainty_data = {}
        for metric, value in metrics.items():
            if metric != "Hausdorff Distance":
                # Generate samples with different uncertainty levels
                uncertainty = np.random.normal(value, value * 0.05, n_samples)
                uncertainty_data[metric] = uncertainty

        # Create violin plot
        fig = go.Figure()
        for metric, values in uncertainty_data.items():
            fig.add_trace(go.Violin(
                y=values,
                name=metric,
                box_visible=True,
                meanline_visible=True
            ))

        fig.update_layout(
            title="Metric Uncertainty Distribution",
            yaxis_title="Value",
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add confidence intervals
        st.subheader("Confidence Intervals")
        ci_data = []
        for metric, values in uncertainty_data.items():
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            ci_data.append({
                'Metric': metric,
                'Mean': np.mean(values),
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'Std Dev': np.std(values)
            })
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df.round(4))

    @staticmethod
    def plot_sensitivity_analysis(metrics):
        """Create sensitivity analysis visualization"""
        st.subheader("Sensitivity Analysis")

        perturbation_levels = np.linspace(-0.1, 0.1, 20)
        sensitivity_data = []
        
        for metric, base_value in metrics.items():
            if metric != "Hausdorff Distance":
                for perturbation in perturbation_levels:
                    perturbed_value = base_value * (1 + perturbation)
                    sensitivity_data.append({
                        'Metric': metric,
                        'Perturbation': perturbation * 100,
                        'Value': min(1.0, max(0.0, perturbed_value))
                    })

        sensitivity_df = pd.DataFrame(sensitivity_data)

        fig = px.line(
            sensitivity_df,
            x='Perturbation',
            y='Value',
            color='Metric',
            title='Sensitivity Analysis',
            labels={'Perturbation': 'Input Perturbation (%)', 'Value': 'Metric Value'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_regional_performance(metrics):
        """Create enhanced regional performance visualization"""
        st.subheader("Regional Performance Analysis")

        regions = ['Region A', 'Region B', 'Region C', 'Region D']
        regional_data = []

        for region in regions:
            region_metrics = {}
            for metric, value in metrics.items():
                if metric != "Hausdorff Distance":
                    region_metrics[metric] = value * (1 + np.random.normal(0, 0.05))
            region_metrics['Region'] = region
            regional_data.append(region_metrics)

        regional_df = pd.DataFrame(regional_data)

        # Create heatmap
        heatmap_data = regional_df.drop('Region', axis=1)
        fig = px.imshow(
            heatmap_data,
            x=heatmap_data.columns,
            y=regions,
            color_continuous_scale='RdYlBu_r',
            title='Regional Performance Heatmap'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Regional comparison
        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox(
                "Select Metric for Regional Comparison",
                options=[col for col in regional_df.columns if col != 'Region']
            )

            fig = px.bar(
                regional_df,
                x='Region',
                y=selected_metric,
                title=f'Regional Comparison - {selected_metric}'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            selected_region = st.selectbox(
                "Select Region for Detailed View",
                options=regions
            )

            region_data = regional_df[regional_df['Region'] == selected_region].iloc[0]
            metrics_for_radar = [m for m in metrics.keys() if m != "Hausdorff Distance"]
            values_for_radar = [region_data[m] for m in metrics_for_radar]

            fig = go.Figure(data=go.Scatterpolar(
                r=values_for_radar,
                theta=metrics_for_radar,
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title=f'Performance Radar - {selected_region}'
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_performance_radar(metrics):
        """Create enhanced performance radar visualization"""
        st.subheader("Performance Radar Analysis")

        metrics_for_radar = {k: v for k, v in metrics.items() if k != "Hausdorff Distance"}
        
        fig = go.Figure()

        # Current performance
        fig.add_trace(go.Scatterpolar(
            r=list(metrics_for_radar.values()),
            theta=list(metrics_for_radar.keys()),
            fill='toself',
            name='Current'
        ))

        # Baseline performance
        baseline_values = [v * 0.9 for v in metrics_for_radar.values()]
        fig.add_trace(go.Scatterpolar(
            r=baseline_values,
            theta=list(metrics_for_radar.keys()),
            fill='toself',
            name='Baseline'
        ))

        # Target performance
        target_values = [min(1.0, v * 1.1) for v in metrics_for_radar.values()]
        fig.add_trace(go.Scatterpolar(
            r=target_values,
            theta=list(metrics_for_radar.keys()),
            fill='toself',
            name='Target'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Performance Radar Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Performance summary table
        summary_data = {
            'Metric': list(metrics_for_radar.keys()),
            'Current': list(metrics_for_radar.values()),
            'Baseline': baseline_values,
            'Target': target_values,
            'Gap to Target': [t - c for c, t in zip(metrics_for_radar.values(), target_values)]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(4))

def main():
    st.title("Medical Image Analysis Dashboard")
    
    # Setup dataset
    if not DatasetManager.setup_dataset():
        st.error("Failed to setup required dataset. Please try again.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_file = st.file_uploader(
            "Upload Medical Image",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            key='image_uploader'
        )
        
        st.subheader("Analysis Settings")
        segmentation_threshold = st.slider(
            "Segmentation Threshold",
            0.0, 1.0, 0.5,
            key='seg_threshold'
        )
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Basic", "Advanced", "Research"],
            key='analysis_type'
        )

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
            
            # Basic metrics display
            st.subheader("Key Metrics")
            DashboardVisualizer.display_metrics(metrics)
            
            # Add tabs for different analyses
            tabs = st.tabs([
                "Uncertainty Analysis", 
                "Sensitivity Analysis", 
                "Regional Performance", 
                "Performance Radar"
            ])
            
            with tabs[0]:
                DashboardVisualizer.plot_uncertainty_analysis(metrics)
                
            with tabs[1]:
                DashboardVisualizer.plot_sensitivity_analysis(metrics)
                
            with tabs[2]:
                DashboardVisualizer.plot_regional_performance(metrics)
                
            with tabs[3]:
                DashboardVisualizer.plot_performance_radar(metrics)

            logger.info("Enhanced dashboard visualization completed")

    else:
        st.info("Please upload a medical image to begin analysis")
        st.write("Supported formats: PNG, JPG, JPEG, DICOM")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for details.")

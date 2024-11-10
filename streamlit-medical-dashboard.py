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

# Set page config
st.set_page_config(
    page_title="Medical Image Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve layout
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
    </style>
""", unsafe_allow_html=True)

# Function to download and extract the dataset
def download_and_extract_zip(url, extract_to='.'):
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
        st.success("Dataset downloaded and extracted successfully.")
    else:
        st.error("Failed to download the dataset.")

# Run this only if the dataset is not already downloaded
dataset_path = 'heart_dataset_sup/heart_dataset'
if not os.path.exists(dataset_path):
    download_and_extract_zip(
        'https://github.com/datascintist-abusufian/medical-image-analysis/raw/main/heart_dataset_sup.zip', 
        extract_to='.'
    )

def main():
    st.title("Medical Image Analysis Dashboard")
    
    # Sidebar for upload and settings
    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_file = st.file_uploader("Upload Medical Image", type=['png', 'jpg', 'jpeg', 'dcm'])
        
        # Analysis settings
        st.subheader("Analysis Settings")
        segmentation_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Basic", "Advanced", "Research"]
        )

    # Main content area
    if uploaded_file is not None:
        # Display original and processed images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file) if uploaded_file.type != "application/dicom" else None
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Segmentation Result")
            # Placeholder for segmentation result
            st.info("Segmentation visualization would appear here")
        
        # Simulate processing
        with st.spinner('Processing image...'):
            time.sleep(2)  # Simulate processing time
            
        # Display metrics in columns
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Sample metrics (would come from actual image processing)
        metrics = {
            "Dice Score": 0.92,
            "IoU Score": 0.85,
            "Sensitivity": 0.89,
            "Specificity": 0.94,
            "Precision": 0.91,
            "Accuracy": 0.93,
            "F1 Score": 0.915,
            "Hausdorff Distance": 3.2
        }
        
        with col1:
            st.metric("Dice Score", f"{metrics['Dice Score']:.3f}")
            st.metric("IoU Score", f"{metrics['IoU Score']:.3f}")
        with col2:
            st.metric("Sensitivity", f"{metrics['Sensitivity']:.3f}")
            st.metric("Specificity", f"{metrics['Specificity']:.3f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.3f}")
            st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        with col4:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            st.metric("Hausdorff Dist.", f"{metrics['Hausdorff Distance']:.1f} px")

        # Time series data
        st.subheader("Performance Over Time")
        time_series_data = pd.DataFrame({
            'Frame': range(1, 6),
            'Dice Score': [0.91, 0.92, 0.93, 0.92, 0.91],
            'IoU': [0.84, 0.85, 0.86, 0.85, 0.84],
            'Sensitivity': [0.88, 0.89, 0.90, 0.89, 0.88],
            'Precision': [0.90, 0.91, 0.92, 0.91, 0.90]
        })
        
        fig = px.line(
            time_series_data, 
            x='Frame',
            y=['Dice Score', 'IoU', 'Sensitivity', 'Precision'],
            title='Metrics Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regional Performance")
            region_data = pd.DataFrame({
                'Region': ['R1', 'R2', 'R3', 'R4'],
                'Accuracy': [0.94, 0.91, 0.93, 0.92],
                'Sensitivity': [0.92, 0.89, 0.90, 0.88],
                'Specificity': [0.95, 0.93, 0.94, 0.95]
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
                'Value': [0.91, 0.89, 0.915, 0.93, 0.94]
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

        # Advanced Analysis Section
        if analysis_type in ["Advanced", "Research"]:
            st.subheader("Advanced Analysis")
            
            with st.expander("Detailed Statistics"):
                st.write("Statistical Analysis of Segmentation")
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                    'Value': [0.92, 0.03, 0.88, 0.95, 0.91]
                })
                st.dataframe(stats_df)
            
            with st.expander("Uncertainty Analysis"):
                st.write("Confidence intervals and uncertainty metrics would appear here")
                
            if analysis_type == "Research":
                with st.expander("Research Metrics"):
                    st.write("Additional research-specific metrics and analyses")
                    st.info("Custom analysis parameters can be added here")

    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a medical image to begin analysis")
        st.write("Supported formats: PNG, JPG, JPEG, DICOM")

if __name__ == "__main__":
    main()

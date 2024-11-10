# Medical Image Analysis Dashboard

This repository contains a Streamlit-based web application for analyzing medical images, specifically designed to display metrics and visualizations for medical image segmentation tasks.

![Medical Image Analysis Dashboard](screenshot.png) <!-- Optionally, include a screenshot if you have one -->

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Medical Image Analysis Dashboard** is a tool for medical image analysis, offering functionalities for segmentation visualization, metrics calculation, and performance monitoring. It includes:
- **Upload Options** for medical image files (PNG, JPG, JPEG, DICOM).
- **Metrics Calculation** such as Dice Score, IoU Score, Sensitivity, Specificity, etc.
- **Visualization Tools** for tracking segmentation performance over time.

This dashboard was created by [Md Abu Sufian](https://github.com/datascintist-abusufian) for medical researchers and data scientists to assist in analyzing and interpreting medical image data.

## Features

- **Image Upload**: Supports image formats such as PNG, JPG, JPEG, and DICOM.
- **Segmentation Visualization**: Displays original and segmented images side-by-side.
- **Metrics Display**: Includes Dice Score, IoU Score, Sensitivity, Specificity, Precision, Accuracy, F1 Score, and Hausdorff Distance.
- **Time Series Plot**: Graphs metrics over time.
- **Regional Analysis**: Displays performance across different regions using bar and radar charts.

## Installation

1. **Clone this repository**:
    ```bash
    git clone https://github.com/datascintist-abusufian/medical-image-analysis.git
    cd medical-image-analysis
    ```

2. **Install dependencies**:
    - It is recommended to use a virtual environment:
      ```bash
      python3 -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```
    - Install required packages:
      ```bash
      pip install -r requirements.txt
      ```

3. **Download and Extract Dataset**:
    - The `heart_dataset_sup.zip` file contains sample medical images and should be downloaded programmatically within the script. The code will automatically handle this (see Usage below).

## Usage

1. **Run the Streamlit App**:
    ```bash
    streamlit run streamlit-medical-dashboard.py
    ```

2. **Using the Application**:
   - Upload medical images for analysis by selecting PNG, JPG, JPEG, or DICOM files.
   - Adjust settings like segmentation threshold and analysis type from the sidebar.
   - View original and segmented images, metrics, and analysis visualizations in the main area.

## Folder Structure

This repository includes:
- `streamlit-medical-dashboard.py`: The main Streamlit app script.
- `heart_dataset_sup.zip`: Zipped dataset with a folder structure for sample medical images.
- `requirements.txt`: File listing all necessary Python libraries for the app.

After unzipping, the dataset should follow this structure:

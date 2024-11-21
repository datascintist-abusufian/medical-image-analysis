# Medical Image Analysis Dashboard

A Streamlit-based dashboard for medical image analysis with advanced visualization and analysis capabilities.

## 🌟 Features

- **Image Analysis**
  - Upload and process medical images
  - Support for PNG, JPG, JPEG formats
  - DICOM support coming soon

- **Advanced Analytics**
  - Uncertainty Analysis
  - Sensitivity Analysis
  - Regional Performance Analysis
  - Performance Radar Visualization

- **Key Metrics**
  - Dice Score
  - IoU Score
  - Sensitivity/Specificity
  - Precision/Accuracy
  - F1 Score
  - Hausdorff Distance

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/medical-image-analysis.git
cd medical-image-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

## 📊 Dashboard Components

1. **Image Upload & Display**
   - Original image view
   - Segmentation result visualization
   - Support for multiple image formats

2. **Metrics Display**
   - Real-time calculation of key metrics
   - Interactive visualization
   - Comparative analysis

3. **Advanced Analysis**
   - Uncertainty Analysis with confidence intervals
   - Sensitivity Analysis with perturbation plots
   - Regional Performance Analysis with heatmaps
   - Performance Radar with baseline comparison

## 💻 Usage

1. Launch the application using `streamlit run app.py`
2. Upload a medical image using the sidebar
3. Adjust analysis settings as needed
4. View various analyses through the interactive tabs
5. Export or save results as needed

## 📂 Project Structure

```
medical-image-analysis/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── data/              # Data directory (created automatically)
    └── heart_dataset_sup/
```

## 🛠️ Technical Details

### Built With
- [Streamlit](https://streamlit.io/) - The web framework used
- [Plotly](https://plotly.com/) - Interactive visualizations
- [OpenCV](https://opencv.org/) - Image processing
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) - Data processing

### System Requirements
- RAM: 4GB minimum (8GB recommended)
- Storage: 500MB free space
- CPU: Multi-core processor recommended

## 📋 Data Format

The dashboard currently supports:
- PNG images
- JPG/JPEG images
- DICOM support coming soon

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Your Name - [linkedin](https://www.linkedin.com/in/tacticalbusinessintelligence/)

Project Link: [https://github.com/datascintist-abusufian/medical-image-analysis]

## 🙏 Acknowledgments

- Reference any papers or datasets used
- Credit to contributors
- Mention any inspirations or related projects

## 🔄 Updates & Version History

- v1.0.0 (Current)
  - Initial release
  - Basic image analysis functionality
  - Advanced visualization features

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{medical_image_analysis,
  author = {Your Name},
  title = {Medical Image Analysis Dashboard},
  year = {2024},
  url = {https://github.com/datascintist-abusufian/medical-image-analysis}
}
```

## ⚠️ Disclaimer

This software is for research and educational purposes only. Not intended for clinical use or medical diagnosis.

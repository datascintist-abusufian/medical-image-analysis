# 4D Cardiac MRI Analysis Dashboard

A Streamlit dashboard for exploring public **4D cine cardiac MRI** (3D + time) with expert segmentations, and for computing **real segmentation-quality metrics** and **cardiac function** from them.

**Live app:** https://abusufian-medical-image-analysis-pro.streamlit.app/

> **Related project:** this dashboard is the evaluation companion to [triFuse-pytorch](https://github.com/datascintist-abusufian/triFuse-pytorch), the TriFuse-SRNet reference implementation for scribble-supervised cardiac MRI segmentation. Upload a predicted mask and it is scored against the expert masks.

## 🫀 Data

The app downloads cases on demand from Hugging Face. No imaging data is stored in this repository; `data/` holds only small case catalogues (IDs, diagnostic group, ED/ES frame, height/weight, reported LVEF).

| Source | Cases | What is 4D | Expert masks | Licence |
|---|---|---|---|---|
| **ACDC** (MICCAI 2017) | 150 patients, 5 groups: NOR, MINF, DCM, HCM, RV | Short-axis cine, ~10 slices × 12–35 frames | LV cavity, myocardium, RV cavity at ED and ES | CC BY-NC-SA 4.0 |
| **CMR-MULTI** (MICCAI 2026) | 105 multi-centre patients | Short-axis cine, ~11 slices × 25–30 frames | LV cavity, myocardium, RV cavity on **every frame** | CC BY-NC 4.0, academic use only |

* ACDC: Bernard O., Lalande A., Zotti C., Cervenansky F. *et al.* "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" *IEEE Transactions on Medical Imaging* 37(11):2514–2525, 2018. doi:[10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502). Official site: https://www.creatis.insa-lyon.fr/Challenge/acdc/ (loaded from the unmodified mirror `MedOtter/ACDC`).
* CMR-MULTI: Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation Challenge, 1st Workshop on Medical World Models, MICCAI 2026. BAAI and Beijing Anzhen Hospital. https://mwm2026.github.io/cmr-multi · https://huggingface.co/datasets/TaipingQu/CMR-MULTI

CMR-MULTI stores each cine series as slices × frames stacked along one axis. The app finds the frame count from the drop in image correlation at slice boundaries and reshapes the series to 4D.

## 🌟 Features

- **4D viewer**: pick a patient, browse slices and cardiac phases, play the cine loop of any slice, and see all slices at once with LV/myocardium/RV overlays.
- **Cardiac function from the masks**: LV and RV EDV, ESV, stroke volume and EF; LV mass (1.05 g/mL); BSA-indexed values for ACDC; volume–time curves over the whole heartbeat for CMR-MULTI.
- **Segmentation metrics computed from mask pairs**: Dice, IoU, sensitivity, specificity, precision, volume error, HD95 and ASSD (3D, in mm, using the true voxel spacing).
  - **Regional**: Dice per slice from base to apex.
  - **Uncertainty**: bootstrap 95% confidence interval of per-slice Dice.
  - **Temporal**: Dice across the cardiac cycle when every frame is labelled.
  - CSV export.
- **What gets evaluated**
  - *Upload my predicted mask*: a NIfTI mask for one frame (3D), for all frames (4D), or, for CMR-MULTI, in the original stacked layout. Use the dataset's label IDs, which the app lists in the sidebar.
  - *Classical baseline*: Otsu threshold inside a window around the expert LV. It demonstrates the metrics and is not an automatic model.

## 🚀 Quick start

```bash
git clone https://github.com/datascintist-abusufian/medical-image-analysis.git
cd medical-image-analysis
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Each case downloads once (about 5–20 MB) and is then cached in `/tmp/hf_cardiac_cache`.

## 📂 Project structure

```
medical-image-analysis/
├── app.py                        # Streamlit app
├── requirements.txt
├── data/
│   ├── acdc_catalog.csv          # ACDC case list (no images)
│   └── cmr_multi_catalog.csv     # CMR-MULTI case list (no images)
└── docs/
```

## 🛠️ Built with

Streamlit · Plotly · NiBabel · SciPy · NumPy · pandas · huggingface_hub

## 📄 License

The code is MIT licensed; see [LICENSE](LICENSE). The imaging data stay under their own licences, listed above, and are for non-commercial research use.

## 📞 Contact

Md Abu Sufian - [LinkedIn](https://www.linkedin.com/in/tacticalbusinessintelligence/) | [Google Scholar](https://scholar.google.com/citations?user=8ozStcUAAAAJ&hl=en) | [GitHub](https://github.com/datascintist-abusufian)

## 🔄 Version history

- **v2.0.0**: 4D cine MRI from ACDC and CMR-MULTI, 4D viewer, cardiac function, and metrics computed from real mask pairs, replacing the simulated values of v1.
- **v1.0.0**: prototype dashboard with simulated metrics.

## 📝 Citation

If you use this software, please cite it and the datasets above:

```bibtex
@software{sufian_medical_image_analysis,
  author = {Sufian, Md Abu},
  title  = {4D Cardiac MRI Analysis Dashboard},
  year   = {2026},
  url    = {https://github.com/datascintist-abusufian/medical-image-analysis}
}
```

## ⚠️ Disclaimer

For research and education only. Not for clinical use or medical diagnosis.

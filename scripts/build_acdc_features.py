"""Compute diagnostic features for all 150 ACDC patients from the expert ED/ES masks.

Writes data/acdc_features.csv (one row per patient, no image data). Run from the repo root:
    python scripts/build_acdc_features.py
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

LABELS = {"RV cavity": 1, "Myocardium": 2, "LV cavity": 3}
REPO = "MedOtter/ACDC"

cat = pd.read_csv(ROOT / "data" / "acdc_catalog.csv")
rows = []
for c in cat.itertuples():
    base = f"{c.split}/{c.id}/{c.id}"
    g_ed = nib.load(hf_hub_download(REPO, f"{base}_frame{c.ed:02d}_gt.nii.gz", repo_type="dataset"))
    g_es = nib.load(hf_hub_download(REPO, f"{base}_frame{c.es:02d}_gt.nii.gz", repo_type="dataset"))
    sp = tuple(float(z) for z in g_ed.header.get_zooms()[:3])
    ed, es = (np.asanyarray(g.dataobj).astype(np.uint8) for g in (g_ed, g_es))
    f = A.diagnostic_features(ed, es, LABELS, sp, c.height, c.weight)
    lv, myo, rv = (ed == LABELS[k] for k in ("LV cavity", "Myocardium", "RV cavity"))
    levels = A.slice_levels(lv, myo)
    ins = A.rv_insertion_angles(lv, myo, rv)
    if ins is not None and levels:
        score, _ = A.anterior_score_fn(None)
        wt = A.regional_thickness(ed, LABELS, sp, levels, ins, score)
        f["Segments_over_15mm"] = int(sum(v > 15 for v in wt.values() if v == v))
        f["WT_segment_max"] = float(np.nanmax(list(wt.values())))
    rows.append({"id": c.id, "split": c.split, "group": c.group, **f})
    print(c.id, c.group, round(f["LVEF"], 1), flush=True)

pd.DataFrame(rows).round(2).to_csv(ROOT / "data" / "acdc_features.csv", index=False)
print("wrote", len(rows), "rows")

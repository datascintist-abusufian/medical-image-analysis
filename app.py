"""
4D Cardiac MRI Analysis Dashboard
=================================

Browse public 4D (3D + time) cine cardiac MRI with expert segmentations, and
compute real segmentation-quality metrics and cardiac function.

Data sources (downloaded on demand from Hugging Face, not stored in this repo):
  * ACDC  - Automated Cardiac Diagnosis Challenge (MICCAI 2017), 150 patients,
            5 diagnostic groups, expert masks at end-diastole (ED) and end-systole (ES).
            Bernard et al., IEEE TMI 37(11):2514-2525, 2018. doi:10.1109/TMI.2018.2837502
            Licence: CC BY-NC-SA 4.0
  * CMR-MULTI - Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation
            Challenge (MICCAI 2026), 105 cine short-axis cases, expert masks on every frame.
            BAAI / Beijing Anzhen Hospital. Licence: CC BY-NC 4.0, academic research only.

For research and education only. Not for clinical use.
"""

import gzip
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="4D Cardiac MRI Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
CACHE_DIR = Path("/tmp/hf_cardiac_cache")

ACDC_REPO = "MedOtter/ACDC"            # unmodified mirror of the official ACDC release
CMR_REPO = "TaipingQu/CMR-MULTI"       # official CMR-MULTI release

ACDC_GROUPS = {
    "NOR": "Normal",
    "MINF": "Previous myocardial infarction",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}

# Unified structure names -> label id per dataset
STRUCTURES = ["LV cavity", "Myocardium", "RV cavity"]
LABELS = {
    "ACDC": {"RV cavity": 1, "Myocardium": 2, "LV cavity": 3},
    "CMR-MULTI": {"Myocardium": 1, "LV cavity": 2, "RV cavity": 3},
}
COLOURS = {"LV cavity": (230, 57, 70), "Myocardium": (69, 123, 157), "RV cavity": (244, 162, 97)}
MYO_DENSITY = 1.05  # g/mL


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data
def load_catalog(name):
    return pd.read_csv(APP_DIR / "data" / name).to_dict("records")


def _download(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(CACHE_DIR))


def _to_uint8(vol):
    lo, hi = np.percentile(vol, [0.5, 99.5])
    return np.clip((vol - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def detect_frames(stacked, tmin=8, tmax=60):
    """CMR-MULTI stores slices x frames along one axis (frame fastest).
    Pick the frame count T whose slice boundaries show the sharpest drop in
    correlation between neighbouring images."""
    n = stacked.shape[2]
    v = stacked.reshape(-1, n).astype(np.float32)
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    corr = (v[:, :-1] * v[:, 1:]).mean(0)
    best_t, best_score = None, -np.inf
    for t in range(tmin, min(tmax, n) + 1):
        if n % t or n // t < 2:
            continue
        b = np.arange(t - 1, n - 1, t)
        inner = np.setdiff1d(np.arange(n - 1), b)
        score = corr[inner].mean() - corr[b].mean()
        if score > best_score:
            best_t, best_score = t, score
    return best_t


@st.cache_resource(max_entries=2, show_spinner="Downloading 4D MRI from Hugging Face…")
def load_acdc(pid, split, ed, es):
    base = f"{split}/{pid}/{pid}"
    img = nib.load(_download(ACDC_REPO, f"{base}_4d.nii.gz"))
    vol = _to_uint8(img.get_fdata(dtype=np.float32))                     # (X, Y, S, T)
    zooms = img.header.get_zooms()[:3]
    gt = np.zeros(vol.shape, dtype=np.uint8)
    has_gt = np.zeros(vol.shape[3], dtype=bool)
    for f in sorted({ed, es}):
        g = nib.load(_download(ACDC_REPO, f"{base}_frame{f:02d}_gt.nii.gz"))
        gt[..., f - 1] = np.asanyarray(g.dataobj).astype(np.uint8)
        has_gt[f - 1] = True
    return dict(img=vol, gt=gt, has_gt=has_gt, spacing=tuple(float(z) for z in zooms),
                ed=ed - 1, es=es - 1, stacked=False)


@st.cache_resource(max_entries=2, show_spinner="Downloading 4D MRI from Hugging Face…")
def load_cmr(case_id):
    name = f"CINE_SAX_{case_id:03d}.nii.gz"
    img = nib.load(_download(CMR_REPO, f"CINE_MULTI/SAX_TR/image/{name}"))
    raw = img.get_fdata(dtype=np.float32)
    ann = np.asanyarray(nib.load(_download(CMR_REPO, f"CINE_MULTI/SAX_TR/anno/{name}")).dataobj)
    t = detect_frames(raw)
    s = raw.shape[2] // t
    vol = _to_uint8(raw).reshape(raw.shape[0], raw.shape[1], s, t)
    gt = ann.astype(np.uint8).reshape(raw.shape[0], raw.shape[1], s, t)
    zx, zy, zz = img.header.get_zooms()[:3]
    # The affine steps through the stacked axis, so one slice step = T stacked steps.
    spacing = (float(zx), float(zy), float(zz) * t)
    lv = (gt == LABELS["CMR-MULTI"]["LV cavity"]).sum((0, 1, 2))
    return dict(img=vol, gt=gt, has_gt=np.ones(t, dtype=bool), spacing=spacing,
                ed=int(lv.argmax()), es=int(lv.argmin()), stacked=True)


# --------------------------------------------------------------------------------------
# Baseline segmentation and metrics
# --------------------------------------------------------------------------------------
def otsu_threshold(x):
    hist, edges = np.histogram(x, bins=256, range=(0, 256))
    p = hist.astype(float) / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return edges[int(np.nanargmax(sb))]


def baseline_lv(img3d, gt3d, lv_label, margin=8):
    """Classical LV-cavity baseline: Otsu threshold inside a region of interest
    (the expert LV box enlarged by `margin` px), keep the bright blob nearest the
    ROI centre, fill holes. The ROI uses the expert mask, so this is a demo
    reference for the metrics, not an automatic model."""
    pred = np.zeros(gt3d.shape, dtype=np.uint8)
    for s in range(gt3d.shape[2]):
        ref = gt3d[..., s] == lv_label
        if not ref.any():
            continue
        xs, ys = np.where(ref)
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, ref.shape[0])
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, ref.shape[1])
        roi = img3d[x0:x1, y0:y1, s]
        fg = roi > otsu_threshold(roi)
        fg = ndimage.binary_opening(fg, iterations=1)
        lab, n = ndimage.label(fg)
        if n == 0:
            continue
        cx, cy = (np.array(roi.shape) - 1) / 2
        best, best_d = 0, np.inf
        for k in range(1, n + 1):
            px_, py_ = np.where(lab == k)
            if px_.size < 10:
                continue
            d = np.hypot(px_.mean() - cx, py_.mean() - cy)
            if d < best_d:
                best, best_d = k, d
        if best:
            blob = ndimage.binary_fill_holes(lab == best)
            pred[x0:x1, y0:y1, s][blob] = lv_label
    return pred


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def surface_distances(a, b, spacing):
    if not a.any() or not b.any():
        return None
    sa, sb = _surface(a), _surface(b)
    dt_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    return np.concatenate([dt_b[sa], dt_a[sb]])


def seg_metrics(pred, ref, spacing):
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    tn = pred.size - tp - fp - fn
    vox = np.prod(spacing) / 1000.0
    out = {
        "Dice": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else np.nan,
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) else np.nan,
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "Volume ref (mL)": ref.sum() * vox,
        "Volume pred (mL)": pred.sum() * vox,
    }
    out["Volume error (mL)"] = out["Volume pred (mL)"] - out["Volume ref (mL)"]
    d = surface_distances(pred, ref, spacing)
    out["HD95 (mm)"] = float(np.percentile(d, 95)) if d is not None else np.nan
    out["ASSD (mm)"] = float(d.mean()) if d is not None else np.nan
    return out


def per_slice_dice(pred, ref):
    vals = []
    for s in range(ref.shape[2]):
        p, r = pred[..., s], ref[..., s]
        denom = p.sum() + r.sum()
        vals.append(2 * np.logical_and(p, r).sum() / denom if denom else np.nan)
    return np.array(vals)


def bootstrap_ci(values, n=2000, seed=0):
    v = values[~np.isnan(values)]
    if v.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
def overlay_rgb(img2d, lab2d, labels, alpha, show):
    rgb = np.repeat(img2d[..., None], 3, axis=2).astype(np.float32)
    if show:
        for name, lid in labels.items():
            m = lab2d == lid
            if m.any():
                rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(COLOURS[name], dtype=np.float32)
    return rgb.astype(np.uint8)


def orient(a):
    """Display convention: rows = second image axis."""
    return np.swapaxes(a, 0, 1)


def show_image(rgb, caption):
    st.image(rgb, caption=caption, width="stretch", clamp=True)


def cine_figure(data, slice_idx, labels, alpha, show, step):
    frames = range(0, data["img"].shape[3], step)
    stack = np.stack([
        orient(overlay_rgb(data["img"][:, :, slice_idx, t], data["gt"][:, :, slice_idx, t],
                           labels, alpha, show and data["has_gt"][t]))
        for t in frames
    ])
    fig = px.imshow(stack, animation_frame=0, binary_string=True,
                    labels={"animation_frame": "frame"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    try:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 60
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
    except Exception:
        pass
    return fig


def volumes_over_time(data, labels):
    vox = np.prod(data["spacing"]) / 1000.0
    rows = []
    for t in np.where(data["has_gt"])[0]:
        g = data["gt"][..., t]
        row = {"frame": int(t) + 1}
        for name, lid in labels.items():
            row[name] = (g == lid).sum() * vox
        rows.append(row)
    return pd.DataFrame(rows)


def load_prediction(uploaded, shape, stacked):
    """Accept a 3D mask for one frame, a 4D mask, or (CMR-MULTI) the stacked 3D layout."""
    raw = uploaded.getvalue()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    img = nib.Nifti1Image.from_bytes(raw)
    arr = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    x, y, s, t = shape
    if arr.shape == (x, y, s, t):
        return arr, "4d"
    if stacked and arr.shape == (x, y, s * t):
        return arr.reshape(x, y, s, t), "4d"
    if arr.shape == (x, y, s):
        return arr, "3d"
    raise ValueError(f"Mask shape {arr.shape} does not match the case (expected "
                     f"{(x, y, s)} for one frame or {(x, y, s, t)} for all frames).")


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.header("🫀 4D cardiac MRI")
        source = st.radio(
            "Data source",
            ["ACDC", "CMR-MULTI"],
            captions=["150 patients · 5 diagnostic groups · masks at ED & ES",
                      "105 multi-centre patients · masks on every frame"],
        )
        if source == "ACDC":
            cat = load_catalog("acdc_catalog.csv")
            groups = st.multiselect("Diagnostic group", list(ACDC_GROUPS),
                                    default=list(ACDC_GROUPS),
                                    format_func=lambda g: f"{g} – {ACDC_GROUPS[g]}")
            options = [c for c in cat if c["group"] in groups] or cat
            case = st.selectbox("Patient", options,
                                format_func=lambda c: f"{c['id']} · {c['group']}")
        else:
            cat = load_catalog("cmr_multi_catalog.csv")
            case = st.selectbox("Patient", cat,
                                format_func=lambda c: f"CINE_SAX_{c['id']:03d} · reported LVEF {c['lvef']:.0f}%")

        st.subheader("Display")
        show = st.toggle("Show expert masks", value=True)
        alpha = st.slider("Mask opacity", 0.1, 1.0, 0.45, 0.05)

        st.subheader("Segmentation to evaluate")
        pred_mode = st.radio("Prediction", ["Classical baseline (LV cavity)", "Upload my predicted mask"],
                             help="Upload a NIfTI mask (e.g. a TriFuse-SRNet output) with the same "
                                  "label IDs as the dataset to score it against the expert masks.")
        uploaded = None
        if pred_mode.startswith("Upload"):
            uploaded = st.file_uploader("Predicted mask (.nii / .nii.gz)", type=["nii", "gz"])
            ids = LABELS[source]
            st.caption("Label IDs: " + ", ".join(f"{k} = {v}" for k, v in ids.items()))
    return source, case, show, alpha, pred_mode, uploaded


def header(source, case, data):
    x, y, s, t = data["img"].shape
    sp = data["spacing"]
    c1, c2, c3, c4 = st.columns(4)
    if source == "ACDC":
        c1.metric("Patient", case["id"])
        c2.metric("Diagnosis", case["group"], ACDC_GROUPS[case["group"]], delta_color="off")
    else:
        c1.metric("Patient", f"CINE_SAX_{case['id']:03d}")
        c2.metric("Reported LVEF", f"{case['lvef']:.1f}%")
    c3.metric("4D size (x·y·slices·frames)", f"{x}·{y}·{s}·{t}")
    c4.metric("Voxel (mm)", f"{sp[0]:.2f}·{sp[1]:.2f}·{sp[2]:.1f}")


def tab_viewer(data, labels, show, alpha):
    x, y, s, t = data["img"].shape
    gt_frames = np.where(data["has_gt"])[0]
    c1, c2 = st.columns(2)
    sl = c1.slider("Slice (base → apex order as stored)", 1, s, s // 2) - 1
    default_t = data["ed"] + 1
    fr = c2.slider("Frame (cardiac phase)", 1, t, default_t) - 1
    marks = f"ED = frame {data['ed'] + 1}, ES = frame {data['es'] + 1}"
    if len(gt_frames) < t:
        marks += " · expert masks exist only on ED and ES"
    st.caption(marks)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Cine playback of this slice** (press ▶)")
        step = 1 if x * y * t < 3_000_000 else 2
        st.plotly_chart(cine_figure(data, sl, labels, alpha, show, step), width="stretch")
    with col_b:
        st.markdown(f"**Slice {sl + 1}, frame {fr + 1}**")
        rgb = overlay_rgb(data["img"][:, :, sl, fr], data["gt"][:, :, sl, fr], labels, alpha,
                          show and data["has_gt"][fr])
        show_image(orient(rgb), "Expert mask overlay" if data["has_gt"][fr] and show else "Image")
        legend = " · ".join(f"<span style='color:rgb{COLOURS[n]}'>■</span> {n}" for n in labels)
        st.markdown(legend, unsafe_allow_html=True)

    st.markdown(f"**All slices at frame {fr + 1}**")
    cols = st.columns(min(s, 6))
    for i in range(s):
        rgb = overlay_rgb(data["img"][:, :, i, fr], data["gt"][:, :, i, fr], labels, alpha,
                          show and data["has_gt"][fr])
        with cols[i % len(cols)]:
            show_image(orient(rgb), f"slice {i + 1}")


def tab_function(source, case, data, labels):
    df = volumes_over_time(data, labels)
    ed, es = data["ed"] + 1, data["es"] + 1
    v_ed = df.set_index("frame").loc[ed]
    v_es = df.set_index("frame").loc[es]

    def ef(name):
        if len(df) > 2 and name != "LV cavity":
            # every frame is labelled: use this chamber's own largest and smallest volume
            hi, lo = df[name].max(), df[name].min()
        else:
            hi, lo = v_ed[name], v_es[name]
        return (hi - lo) / hi * 100 if hi else np.nan

    c = st.columns(6)
    c[0].metric("LV EDV", f"{v_ed['LV cavity']:.0f} mL")
    c[1].metric("LV ESV", f"{v_es['LV cavity']:.0f} mL")
    c[2].metric("LV SV", f"{v_ed['LV cavity'] - v_es['LV cavity']:.0f} mL")
    c[3].metric("LVEF (from masks)", f"{ef('LV cavity'):.1f}%")
    c[4].metric("RVEF (from masks)", f"{ef('RV cavity'):.1f}%")
    c[5].metric("LV mass (ED)", f"{v_ed['Myocardium'] * MYO_DENSITY:.0f} g")

    if source == "ACDC":
        bsa = np.sqrt(case["height"] * case["weight"] / 3600)
        st.caption(f"Height {case['height']:.0f} cm · weight {case['weight']:.0f} kg · "
                   f"BSA {bsa:.2f} m² (Mosteller) · LV EDV index {v_ed['LV cavity'] / bsa:.0f} mL/m² · "
                   f"LV mass index {v_ed['Myocardium'] * MYO_DENSITY / bsa:.0f} g/m²")
    else:
        st.caption(f"Reported LVEF in the dataset metadata: {case['lvef']:.1f}%. The value from the "
                   "masks counts every labelled slice, including basal slices that clinical "
                   "reporting may exclude, so the two can differ.")

    if len(df) > 2:
        fig = go.Figure()
        for name in labels:
            fig.add_trace(go.Scatter(x=df["frame"], y=df[name], mode="lines+markers", name=name,
                                     line=dict(color=f"rgb{COLOURS[name]}")))
        for f, lab in [(ed, "ED"), (es, "ES")]:
            fig.add_vline(x=f, line_dash="dot", annotation_text=lab)
        fig.update_layout(title="Volume–time curves from expert masks", xaxis_title="Frame",
                          yaxis_title="Volume (mL)", height=420)
        st.plotly_chart(fig, width="stretch")
    else:
        long = df.melt(id_vars="frame", var_name="Structure", value_name="Volume (mL)")
        long["Phase"] = long["frame"].map({ed: "ED", es: "ES"})
        fig = px.bar(long, x="Structure", y="Volume (mL)", color="Phase", barmode="group",
                     title="Volumes at ED and ES from expert masks")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(df.round(1), hide_index=True, width="stretch")


def tab_metrics(source, data, labels, pred_mode, uploaded):
    sp = data["spacing"]
    lv_id = labels["LV cavity"]
    frames = [data["ed"], data["es"]]

    if pred_mode.startswith("Upload"):
        if uploaded is None:
            st.info("Upload a predicted mask in the sidebar to score it against the expert masks.")
            return
        try:
            pred, kind = load_prediction(uploaded, data["img"].shape, data["stacked"])
        except Exception as e:
            st.error(f"Could not read the mask: {e}")
            return
        if kind == "3d":
            which = st.radio("This 3D mask is for", ["ED", "ES"], horizontal=True)
            f = data["ed"] if which == "ED" else data["es"]
            pred4 = np.zeros_like(data["gt"])
            pred4[..., f] = pred
            frames = [f]
        else:
            pred4 = pred
        structures = list(labels)
        pred_name = f"Uploaded mask ({uploaded.name})"
    else:
        pred4 = np.zeros_like(data["gt"])
        for f in np.where(data["has_gt"])[0]:
            pred4[..., f] = baseline_lv(data["img"][..., f], data["gt"][..., f], lv_id)
        structures = ["LV cavity"]
        pred_name = "Classical baseline (Otsu inside an ROI around the expert LV)"
        st.caption("The baseline only segments the LV cavity and uses the expert mask to place its "
                   "search window, so it shows how the metrics behave; it is not an automatic model. "
                   "Upload your own model's mask in the sidebar for a fair evaluation.")

    rows = []
    for f in frames:
        phase = "ED" if f == data["ed"] else "ES"
        for name in structures:
            lid = labels[name]
            m = seg_metrics(pred4[..., f] == lid, data["gt"][..., f] == lid, sp)
            rows.append({"Phase": phase, "Frame": f + 1, "Structure": name, **m})
    res = pd.DataFrame(rows)
    st.markdown(f"**Evaluated:** {pred_name}")

    head = res[(res.Structure == "LV cavity") & (res.Phase == "ED")]
    if not head.empty:
        h = head.iloc[0]
        c = st.columns(4)
        c[0].metric("Dice (LV, ED)", f"{h['Dice']:.3f}")
        c[1].metric("IoU (LV, ED)", f"{h['IoU']:.3f}")
        c[2].metric("HD95 (LV, ED)", f"{h['HD95 (mm)']:.1f} mm")
        c[3].metric("Volume error (LV, ED)", f"{h['Volume error (mL)']:+.1f} mL")
        c = st.columns(4)
        c[0].metric("Sensitivity", f"{h['Sensitivity']:.3f}")
        c[1].metric("Specificity", f"{h['Specificity']:.4f}")
        c[2].metric("Precision", f"{h['Precision']:.3f}")
        c[3].metric("ASSD", f"{h['ASSD (mm)']:.2f} mm")

    st.dataframe(res.round(3), hide_index=True, width="stretch")

    t1, t2, t3 = st.tabs(["Regional (per slice)", "Uncertainty (bootstrap)", "Temporal"])
    with t1:
        pr = []
        for f in frames:
            for name in structures:
                lid = labels[name]
                d = per_slice_dice(pred4[..., f] == lid, data["gt"][..., f] == lid)
                for i, v in enumerate(d):
                    pr.append({"Slice": i + 1, "Dice": v, "Structure": name,
                               "Phase": "ED" if f == data["ed"] else "ES"})
        pr = pd.DataFrame(pr).dropna()
        if pr.empty:
            st.info("No labelled slices to compare.")
        else:
            fig = px.bar(pr, x="Slice", y="Dice", color="Phase", facet_row="Structure",
                         barmode="group", range_y=[0, 1], height=260 * len(structures))
            st.plotly_chart(fig, width="stretch")
            st.caption("Low Dice on the first and last labelled slices is common: basal and apical "
                       "slices are the hardest to segment.")
    with t2:
        rows = []
        for f in frames:
            for name in structures:
                lid = labels[name]
                d = per_slice_dice(pred4[..., f] == lid, data["gt"][..., f] == lid)
                lo, hi = bootstrap_ci(d)
                rows.append({"Phase": "ED" if f == data["ed"] else "ES", "Structure": name,
                             "Mean slice Dice": np.nanmean(d) if np.isfinite(d).any() else np.nan,
                             "95% CI low": lo, "95% CI high": hi,
                             "Slices": int(np.isfinite(d).sum())})
        u = pd.DataFrame(rows)
        fig = go.Figure()
        for ph in u.Phase.unique():
            q = u[u.Phase == ph]
            fig.add_trace(go.Scatter(
                x=q["Structure"], y=q["Mean slice Dice"], mode="markers", name=ph,
                marker=dict(size=12),
                error_y=dict(type="data", symmetric=False,
                             array=q["95% CI high"] - q["Mean slice Dice"],
                             arrayminus=q["Mean slice Dice"] - q["95% CI low"])))
        fig.update_layout(yaxis=dict(range=[0, 1], title="Mean per-slice Dice"), height=380,
                          title="Bootstrap 95% confidence interval over slices")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(u.round(3), hide_index=True, width="stretch")
    with t3:
        gt_frames = np.where(data["has_gt"])[0]
        avail = [f for f in gt_frames if pred4[..., f].any()]
        if len(avail) <= 2:
            st.info("Dice across the whole heartbeat needs expert masks on every frame "
                    "(CMR-MULTI) and a prediction for every frame.")
        else:
            tr = []
            for f in avail:
                for name in structures:
                    lid = labels[name]
                    m = seg_metrics(pred4[..., f] == lid, data["gt"][..., f] == lid, sp)
                    tr.append({"Frame": f + 1, "Structure": name, "Dice": m["Dice"],
                               "HD95 (mm)": m["HD95 (mm)"]})
            tr = pd.DataFrame(tr)
            fig = px.line(tr, x="Frame", y="Dice", color="Structure", markers=True,
                          range_y=[0, 1], title="Dice across the cardiac cycle")
            fig.add_vline(x=data["ed"] + 1, line_dash="dot", annotation_text="ED")
            fig.add_vline(x=data["es"] + 1, line_dash="dot", annotation_text="ES")
            st.plotly_chart(fig, width="stretch")

    st.download_button("Download metrics (CSV)", res.to_csv(index=False).encode(),
                       file_name="segmentation_metrics.csv", mime="text/csv")


def tab_about():
    st.markdown("""
**Data sources** (downloaded on demand from Hugging Face; not redistributed in this repository)

* **ACDC** – Automated Cardiac Diagnosis Challenge, MICCAI 2017. 150 patients from University
  Hospital of Dijon in 5 equal groups (normal, previous infarction, dilated cardiomyopathy,
  hypertrophic cardiomyopathy, abnormal RV). 4D cine short-axis MRI with expert LV, RV and
  myocardium masks at end-diastole and end-systole. Licence CC BY-NC-SA 4.0.
  *Citation required:* O. Bernard, A. Lalande, C. Zotti, F. Cervenansky et al., "Deep Learning
  Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem
  Solved?", IEEE Transactions on Medical Imaging 37(11):2514–2525, 2018. doi:10.1109/TMI.2018.2837502.
  Official site: https://www.creatis.insa-lyon.fr/Challenge/acdc/
* **CMR-MULTI** – Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation Challenge
  (MICCAI 2026, 1st Workshop on Medical World Models), curated by BAAI and Beijing Anzhen Hospital.
  105 cine short-axis cases with LV, RV and myocardium masks on every frame. Licence CC BY-NC 4.0;
  academic research only, no clinical or commercial use. https://mwm2026.github.io/cmr-multi ·
  https://huggingface.co/datasets/TaipingQu/CMR-MULTI

**Methods**

* Volumes = labelled voxels × voxel volume. EF = (EDV − ESV) / EDV. Myocardial mass = volume × 1.05 g/mL.
* For CMR-MULTI, ED and ES are the frames with the largest and smallest LV cavity volume.
* HD95 and ASSD use 3D surface distances in millimetres with the true voxel spacing.
* The bootstrap interval resamples per-slice Dice values (2,000 resamples).

*For research and education only. Not for clinical use or diagnosis.*
""")


def main():
    st.title("🫀 4D Cardiac MRI Analysis Dashboard")
    st.caption("Public 4D cine cardiac MRI (3D + time) with expert segmentations · real Dice, IoU, "
               "Hausdorff distance and cardiac function")
    source, case, show, alpha, pred_mode, uploaded = sidebar()
    labels = LABELS[source]
    try:
        if source == "ACDC":
            data = load_acdc(case["id"], case["split"], int(case["ed"]), int(case["es"]))
        else:
            data = load_cmr(int(case["id"]))
    except Exception as e:
        logger.exception("load failed")
        st.error(f"Could not load this case from Hugging Face: {e}. Please try again in a minute.")
        return

    header(source, case, data)
    tabs = st.tabs(["🎞️ 4D viewer", "❤️ Cardiac function", "📏 Segmentation metrics", "ℹ️ Data & methods"])
    with tabs[0]:
        tab_viewer(data, labels, show, alpha)
    with tabs[1]:
        tab_function(source, case, data, labels)
    with tabs[2]:
        tab_metrics(source, data, labels, pred_mode, uploaded)
    with tabs[3]:
        tab_about()


if __name__ == "__main__":
    main()

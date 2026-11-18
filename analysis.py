"""
Advanced cardiac MRI analysis used by app.py (and by scripts/build_acdc_features.py).

Methods
-------
* Shape-based interpolation + marching cubes      -> 3D/4D surface reconstruction
* AHA 16-segment regional wall thickness / thickening (RV-insertion-point referenced)
* Dense TV-L1 optical-flow motion tracking and label propagation through the cardiac cycle
* Segmentation-derived global circumferential and radial strain
* Anatomical quality-control checks on segmentations
* Hand-crafted diagnostic features for ACDC disease classification

All functions work on label volumes shaped (X, Y, S) or (X, Y, S, T) with voxel spacing in mm.
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

AHA_NAMES = {
    1: "Basal anterior", 2: "Basal anteroseptal", 3: "Basal inferoseptal", 4: "Basal inferior",
    5: "Basal inferolateral", 6: "Basal anterolateral", 7: "Mid anterior", 8: "Mid anteroseptal",
    9: "Mid inferoseptal", 10: "Mid inferior", 11: "Mid inferolateral", 12: "Mid anterolateral",
    13: "Apical anterior", 14: "Apical septal", 15: "Apical inferior", 16: "Apical lateral",
}


# --------------------------------------------------------------------------------------
# 3D / 4D reconstruction
# --------------------------------------------------------------------------------------
def shape_based_interpolation(mask, factor):
    """Upsample a binary stack along the slice axis by interpolating signed distance maps
    (Raya & Udupa 1990) so that thick-slice masks give smooth surfaces."""
    if factor <= 1 or not mask.any():
        return mask.astype(bool)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    sdf = outside - inside  # negative inside
    up = ndimage.zoom(sdf, (1, 1, factor), order=1)
    return up < 0


def surface_mesh(mask, spacing, factor=None, step=1):
    """Marching-cubes triangle mesh (vertices in mm) of a binary 3D mask."""
    mask = mask.astype(bool)
    if mask.sum() < 20:
        return None
    if factor is None:
        factor = max(1, int(round(spacing[2] / max(spacing[0], spacing[1]) / 2)))
    vol = shape_based_interpolation(mask, factor)
    vol = np.pad(vol, 1)
    sp = (spacing[0], spacing[1], spacing[2] / factor)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), 0.5, spacing=sp,
                                                    step_size=step)
    except (ValueError, RuntimeError):
        return None
    return verts, faces


# --------------------------------------------------------------------------------------
# Regional (AHA) wall thickness
# --------------------------------------------------------------------------------------
def _circ_mean(a):
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _ang_diff_pos(a, b):
    """Positive (counter-clockwise) angular distance from a to b in [0, 2pi)."""
    return float((b - a) % (2 * np.pi))


def rv_insertion_angles(lv, myo, rv):
    """Return (theta_a, theta_b, theta_septum) angles of the two RV insertion points
    and the septal mid-point around the LV centre, averaged over slices.
    Angles are in array coordinates: theta = atan2(d_axis1, d_axis0)."""
    a_list, b_list, s_list = [], [], []
    for s in range(lv.shape[2]):
        if not (lv[..., s].any() and myo[..., s].any() and rv[..., s].any()):
            continue
        c = np.array(ndimage.center_of_mass(lv[..., s] | myo[..., s]))
        contact = myo[..., s] & ndimage.binary_dilation(rv[..., s], iterations=2)
        if contact.sum() < 5:
            continue
        pts = np.argwhere(contact) - c
        ang = np.sort(np.arctan2(pts[:, 1], pts[:, 0]))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        k = int(np.argmax(gaps))                 # the largest gap is the free wall
        start, end = ang[(k + 1) % len(ang)], ang[k]
        a_list.append(start)
        b_list.append(end)
        s_list.append(start + _ang_diff_pos(start, end) / 2)
    if not a_list:
        return None
    return _circ_mean(np.array(a_list)), _circ_mean(np.array(b_list)), _circ_mean(np.array(s_list))


def assign_aha(theta, level, ins, anterior_score):
    """Map angles (array) on a slice at level 'basal'|'mid'|'apical' to AHA segment ids.
    ins = (theta_a, theta_b, theta_septum); anterior_score(theta) > 0 means anterior."""
    ta, tb, ts = ins
    # decide which insertion is anterior
    if anterior_score(ta) < anterior_score(tb):
        ta, tb = tb, ta
    # travel direction from anterior insertion through the septum
    direction = 1.0 if _ang_diff_pos(ta, ts) < _ang_diff_pos(ta, tb) else -1.0
    rel = ((theta - ta) * direction) % (2 * np.pi)          # 0 at anterior insertion, septum first
    sept_arc = ((tb - ta) * direction) % (2 * np.pi)
    seg = np.zeros(theta.shape, dtype=int)
    if level in ("basal", "mid"):
        off = 0 if level == "basal" else 6
        in_sept = rel < sept_arc
        half = sept_arc / 2
        seg[in_sept & (rel < half)] = 2 + off
        seg[in_sept & (rel >= half)] = 3 + off
        lat = (rel - sept_arc) / max(2 * np.pi - sept_arc, 1e-6)   # 0..1 over lateral arc
        q = np.clip((lat * 4).astype(int), 0, 3)
        lateral_ids = np.array([4, 5, 6, 1]) + off
        seg[~in_sept] = lateral_ids[q[~in_sept]]
    else:
        # four 90-degree sectors centred on the septum
        rel_s = ((theta - ts) * direction + np.pi / 4) % (2 * np.pi)
        q = (rel_s // (np.pi / 2)).astype(int)            # 0 septal, 1 inferior, 2 lateral, 3 anterior
        seg = np.array([14, 15, 16, 13])[q]
    return seg


def slice_levels(lv, myo):
    """Label labelled slices as basal / mid / apical (thirds), base = larger LV area end."""
    idx = [s for s in range(lv.shape[2]) if lv[..., s].any() and myo[..., s].any()]
    if len(idx) < 3:
        return {}
    area = [lv[..., s].sum() for s in idx]
    if area[0] < area[-1]:
        idx = idx[::-1]                       # order base -> apex
    n = len(idx)
    b, m = int(np.ceil(n / 3)), int(np.ceil(2 * n / 3))
    return {s: ("basal" if i < b else "mid" if i < m else "apical") for i, s in enumerate(idx)}


def wall_thickness_slice(lv2d, myo2d, spacing2d):
    """Thickness (mm) at each endocardial boundary pixel = distance to the epicardium."""
    if not (lv2d.any() and myo2d.any()):
        return None, None
    epi_region = ndimage.binary_fill_holes(lv2d | myo2d)
    epi_border = epi_region & ~ndimage.binary_erosion(epi_region)
    endo_border = lv2d & ~ndimage.binary_erosion(lv2d)
    dist = ndimage.distance_transform_edt(~epi_border, sampling=spacing2d)
    pts = np.argwhere(endo_border)
    return pts, dist[endo_border]


def regional_thickness(lab3d, labels, spacing, levels, ins, anterior_score):
    """Mean wall thickness (mm) per AHA segment for one frame."""
    lv = lab3d == labels["LV cavity"]
    myo = lab3d == labels["Myocardium"]
    acc = {k: [] for k in range(1, 17)}
    for s, level in levels.items():
        pts, th = wall_thickness_slice(lv[..., s], myo[..., s], spacing[:2])
        if pts is None or len(th) == 0:
            continue
        c = np.array(ndimage.center_of_mass(lv[..., s]))
        d = pts - c
        theta = np.arctan2(d[:, 1], d[:, 0])
        seg = assign_aha(theta, level, ins, anterior_score)
        for k in np.unique(seg):
            if k:
                acc[int(k)].extend(th[seg == k].tolist())
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}


def anterior_score_fn(affine=None):
    """Return f(theta)->score, larger = more anterior. Uses the RAS affine when it carries
    orientation; otherwise assumes the usual short-axis display with anterior at the top."""
    if affine is not None:
        a0, a1 = np.asarray(affine)[:3, 0], np.asarray(affine)[:3, 1]
        if not (np.allclose(np.abs(a0), np.abs(np.eye(3)[0]) * np.abs(a0).max())
                and np.allclose(np.abs(a1), np.abs(np.eye(3)[1]) * np.abs(a1).max())):
            return lambda t: np.cos(t) * a0[1] + np.sin(t) * a1[1], True
    return lambda t: -np.sin(t), False


# --------------------------------------------------------------------------------------
# Motion tracking, label propagation, strain
# --------------------------------------------------------------------------------------
def _roi(mask3d, margin=20):
    xs, ys = np.where(mask3d.any(axis=2))
    if xs.size == 0:
        return slice(None), slice(None)
    return (slice(max(xs.min() - margin, 0), xs.max() + margin + 1),
            slice(max(ys.min() - margin, 0), ys.max() + margin + 1))


def propagate_labels(img4d, lab_ed, ed_index, margin=20, num_warp=5, num_iter=10, attachment=30,
                     progress=None):
    """Track the ED segmentation through the whole cycle with dense TV-L1 optical flow.

    Flow is estimated between consecutive frames (forward from ED, wrapping round the cycle).
    The frame-to-frame displacement fields are composed into one mapping from every frame
    back to ED, and the ED labels are warped once per frame, so nearest-neighbour resampling
    errors do not accumulate. Returns (labels4d, mean_displacement_px_per_frame)."""
    x, y, s, t = img4d.shape
    rx, ry = _roi(lab_ed > 0, margin)
    crop = img4d[rx, ry].astype(np.float32) / 255.0
    h, w = crop.shape[:2]
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    out = np.zeros((x, y, s, t), dtype=np.uint8)
    out[..., ed_index] = lab_ed
    slices = [k for k in range(s) if lab_ed[..., k].any()]
    phi = {k: (rr.astype(np.float32), cc.astype(np.float32)) for k in slices}
    mags = np.zeros(t)
    prev = ed_index
    for step in range(1, t):
        f = (ed_index + step) % t
        m = []
        for k in slices:
            v, u = optical_flow_tvl1(crop[..., k, f], crop[..., k, prev], num_warp=num_warp,
                                     num_iter=num_iter, attachment=attachment)
            coords = np.array([rr + v, cc + u])
            pr, pc = phi[k]
            phi[k] = (ndimage.map_coordinates(pr, coords, order=1, mode="nearest"),
                      ndimage.map_coordinates(pc, coords, order=1, mode="nearest"))
            out[rx, ry, k, f] = warp(lab_ed[rx, ry, k], np.array(phi[k]), order=0, mode="edge",
                                     preserve_range=True).astype(np.uint8)
            mask = lab_ed[rx, ry, k] > 0
            m.append(np.hypot(v, u)[mask].mean())
        mags[f] = float(np.mean(m)) if m else 0.0
        prev = f
        if progress:
            progress(step / (t - 1))
    return out, mags


def global_strain(lab4d, labels, spacing, frames, levels_mid):
    """Segmentation-based global strains over time (relative to ED = first in `frames`):
    circumferential from mid-wall perimeter, radial from mean wall thickness."""
    lv_id, myo_id = labels["LV cavity"], labels["Myocardium"]
    per, thick = [], []
    for f in frames:
        P, Tt = [], []
        for s in levels_mid:
            lv = lab4d[..., s, f] == lv_id
            myo = lab4d[..., s, f] == myo_id
            if not (lv.any() and myo.any()):
                continue
            epi = ndimage.binary_fill_holes(lv | myo)
            c_endo = measure.find_contours(lv.astype(float), 0.5)
            c_epi = measure.find_contours(epi.astype(float), 0.5)
            if not c_endo or not c_epi:
                continue
            sc = np.array(spacing[:2])
            le = max(np.sum(np.linalg.norm(np.diff(c * sc, axis=0), axis=1)) for c in c_endo)
            lp = max(np.sum(np.linalg.norm(np.diff(c * sc, axis=0), axis=1)) for c in c_epi)
            P.append((le + lp) / 2)
            _, th = wall_thickness_slice(lv, myo, spacing[:2])
            if th is not None and len(th):
                Tt.append(th.mean())
        per.append(np.mean(P) if P else np.nan)
        thick.append(np.mean(Tt) if Tt else np.nan)
    per, thick = np.array(per), np.array(thick)
    return (per - per[0]) / per[0] * 100, (thick - thick[0]) / thick[0] * 100


# --------------------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------------------
def qc_checks(lab3d, labels):
    """Anatomical plausibility checks for one frame. Returns list of (check, ok, detail)."""
    lv = lab3d == labels["LV cavity"]
    myo = lab3d == labels["Myocardium"]
    rv = lab3d == labels["RV cavity"]
    out = []
    frag, enclosed_bad, holes, slices = [], [], [], 0
    for s in range(lab3d.shape[2]):
        for name, m in (("LV cavity", lv[..., s]), ("Myocardium", myo[..., s]), ("RV cavity", rv[..., s])):
            if m.any() and ndimage.label(m)[1] > 1:
                frag.append(f"{name} s{s + 1}")
        if lv[..., s].any():
            slices += 1
            ring = ndimage.binary_dilation(lv[..., s], iterations=1) & ~lv[..., s]
            frac = (ring & myo[..., s]).sum() / max(ring.sum(), 1)
            if frac < 0.7:
                enclosed_bad.append(f"s{s + 1} ({frac:.0%})")
        if lv[..., s].any() and (ndimage.binary_fill_holes(lv[..., s]) & ~lv[..., s]).any():
            holes.append(f"s{s + 1}")
    out.append(("Each structure is one connected piece per slice", not frag,
                ", ".join(frag[:6]) + (" …" if len(frag) > 6 else "") if frag else "ok"))
    out.append(("LV cavity surrounded by myocardium (≥70% of its border)", not enclosed_bad,
                ", ".join(enclosed_bad[:6]) if enclosed_bad else "ok"))
    out.append(("No holes inside the LV cavity", not holes, ", ".join(holes) if holes else "ok"))
    out.append(("LV present on at least 4 slices", slices >= 4, f"{slices} slices"))
    out.append(("Myocardium and RV both present", bool(myo.any() and rv.any()),
                "ok" if myo.any() and rv.any() else "missing structure"))
    return out


# --------------------------------------------------------------------------------------
# Diagnostic features (ACDC)
# --------------------------------------------------------------------------------------
def diagnostic_features(lab_ed, lab_es, labels, spacing, height, weight):
    vox = np.prod(spacing) / 1000.0
    bsa = np.sqrt(height * weight / 3600.0)
    v = {}
    for ph, lab in (("ED", lab_ed), ("ES", lab_es)):
        for name, lid in labels.items():
            v[(ph, name)] = (lab == lid).sum() * vox
    lv_ed, lv_es = v[("ED", "LV cavity")], v[("ES", "LV cavity")]
    rv_ed, rv_es = v[("ED", "RV cavity")], v[("ES", "RV cavity")]
    myo_ed, myo_es = v[("ED", "Myocardium")], v[("ES", "Myocardium")]
    lv = lab_ed == labels["LV cavity"]
    myo = lab_ed == labels["Myocardium"]
    levels = slice_levels(lv, myo)
    th_ed, th_es = [], []
    for s in levels:
        _, t1 = wall_thickness_slice(lv[..., s], myo[..., s], spacing[:2])
        _, t2 = wall_thickness_slice(lab_es[..., s] == labels["LV cavity"],
                                     lab_es[..., s] == labels["Myocardium"], spacing[:2])
        if t1 is not None and len(t1):
            th_ed.append(t1)
        if t2 is not None and len(t2):
            th_es.append(t2)
    all_ed = np.concatenate(th_ed) if th_ed else np.array([np.nan])
    slice_max_ed = np.array([np.percentile(t, 95) for t in th_ed]) if th_ed else np.array([np.nan])
    mean_es = np.mean([t.mean() for t in th_es]) if th_es else np.nan
    return {
        "LVEDVi": lv_ed / bsa, "LVESVi": lv_es / bsa,
        "LVEF": (lv_ed - lv_es) / lv_ed * 100 if lv_ed else np.nan,
        "RVEDVi": rv_ed / bsa, "RVESVi": rv_es / bsa,
        "RVEF": (rv_ed - rv_es) / rv_ed * 100 if rv_ed else np.nan,
        "LVMi": myo_ed * 1.05 / bsa,
        "RV_LV_EDV_ratio": rv_ed / lv_ed if lv_ed else np.nan,
        "Myo_ES_ED_ratio": myo_es / myo_ed if myo_ed else np.nan,
        "WT_ED_mean": float(np.nanmean(all_ed)),
        "WT_ED_max": float(np.nanmax(slice_max_ed)),
        "WT_ED_sd": float(np.nanstd(all_ed)),
        "Thickening": (mean_es - np.nanmean(all_ed)) / np.nanmean(all_ed) * 100,
        "BSA": bsa,
    }


FEATURE_COLUMNS = ["LVEDVi", "LVESVi", "LVEF", "RVEDVi", "RVESVi", "RVEF", "LVMi",
                   "RV_LV_EDV_ratio", "Myo_ES_ED_ratio", "WT_ED_mean", "WT_ED_max", "WT_ED_sd",
                   "Thickening"]


def acdc_rule_based(f):
    """The published ACDC group definitions applied as simple rules (Bernard et al. 2018).
    The 'several segments thicker than 15 mm' HCM criterion is approximated by the
    thickest slice exceeding 15 mm."""
    if f["LVEF"] < 40 and f["LVEDVi"] > 100:
        return "DCM"
    if f["LVEF"] < 40:
        return "MINF"
    if f["LVMi"] > 110 and f["WT_ED_max"] > 15 and f["LVEF"] >= 50:
        return "HCM"
    if f["RVEDVi"] > 110 or f["RVEF"] < 40:
        return "RV"
    return "NOR"

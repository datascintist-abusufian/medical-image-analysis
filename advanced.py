"""
Advanced analysis sections of the 4D Cardiac MRI Analysis Dashboard.

Each section is a function called from app.py with the loaded case. Heavy computations are
cached per case so that moving sliders does not recompute them.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analysis as A

APP_DIR = Path(__file__).parent
GROUP_ORDER = ["NOR", "MINF", "DCM", "HCM", "RV"]
AHA_THETA = {**{k: (90 + 60 * (k - 1)) % 360 for k in range(1, 7)},
             **{k: (90 + 60 * (k - 7)) % 360 for k in range(7, 13)},
             13: 90, 14: 180, 15: 270, 16: 0}


def _key(source, case):
    return f"{source}:{case['id']}"


# --------------------------------------------------------------------------------------
# Shared cached computations
# --------------------------------------------------------------------------------------
def _geometry(data, labels):
    """Slice levels and RV insertion points from the ED expert mask."""
    g = data["gt"][..., data["ed"]]
    lv, myo, rv = (g == labels[k] for k in ("LV cavity", "Myocardium", "RV cavity"))
    return A.slice_levels(lv, myo), A.rv_insertion_angles(lv, myo, rv)


def regional(key, data, labels, frame):
    store = st.session_state.setdefault("_regional", {})
    k = (key, frame)
    if k not in store:
        levels, ins = _geometry(data, labels)
        score, oriented = A.anterior_score_fn(data.get("affine"))
        if not levels or ins is None:
            store[k] = None
        else:
            store[k] = A.regional_thickness(data["gt"][..., frame], labels, data["spacing"],
                                            levels, ins, score), oriented
    return store[k]


def tracked(key):
    return st.session_state.get("_tracking", {}).get(key)


def labels_over_cycle(key, data):
    """Expert masks on every frame (CMR-MULTI) or tracked masks (ACDC, after tracking)."""
    if data["has_gt"].all():
        return data["gt"], "expert masks"
    tr = tracked(key)
    if tr is not None:
        return tr["labels"], "motion-tracked masks"
    return None, None


# --------------------------------------------------------------------------------------
# 3D / 4D reconstruction
# --------------------------------------------------------------------------------------
def _mesh_traces(lab3d, labels, spacing, colours, opacity, step=1):
    traces = []
    for name in ("LV cavity", "Myocardium", "RV cavity"):
        m = A.surface_mesh(lab3d == labels[name], spacing, step=step)
        if m is None:
            continue
        v, f = m
        v = np.round(v, 1).astype(np.float32)
        traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=f"rgb{colours[name]}", opacity=opacity[name], name=name,
                                showlegend=True, flatshading=False,
                                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3)))
    return traces


def section_reconstruction(source, case, data, labels, colours):
    st.subheader("3D / 4D surface reconstruction")
    st.caption("Short-axis slices are 5–10 mm apart, so each mask is first upsampled through the "
               "slices by shape-based interpolation of signed distance maps (Raya & Udupa 1990). "
               "Marching cubes then extracts triangle meshes in millimetres. This is the same "
               "3D+t mesh representation that Imperial's MeshHeart model (Qiao et al., "
               "Nature Machine Intelligence 2025) learns from.")
    key = _key(source, case)
    lab4d, origin = labels_over_cycle(key, data)
    frames = list(np.where(data["has_gt"])[0]) if lab4d is None else list(range(data["img"].shape[3]))
    src = data["gt"] if lab4d is None else lab4d
    c1, c2 = st.columns([2, 1])
    fr = c1.select_slider("Cardiac frame", options=[f + 1 for f in frames],
                          value=data["ed"] + 1,
                          format_func=lambda f: f"{f}" + (" (ED)" if f == data["ed"] + 1 else
                                                         " (ES)" if f == data["es"] + 1 else "")) - 1
    myo_op = c2.slider("Myocardium opacity", 0.1, 1.0, 0.35, 0.05)
    op = {"LV cavity": 1.0, "Myocardium": myo_op, "RV cavity": 0.85}
    fig = go.Figure(_mesh_traces(src[..., fr], labels, data["spacing"], colours, op))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False), height=560, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    if lab4d is None:
        st.info("Expert masks for ACDC exist only at ED and ES. Run **Motion & strain → motion "
                "tracking** to reconstruct every frame of the heartbeat in 4D.")
    else:
        st.caption(f"Frames from {origin}.")
        if st.toggle("Animate the whole heartbeat (4D)", value=False):
            with st.spinner("Building 4D mesh animation…"):
                step = max(1, int(np.ceil(len(frames) / 12)))
                sel = frames[::step]
                op2 = {"LV cavity": 1.0, "Myocardium": 0.3, "RV cavity": 0.85}
                fr_objs = [go.Frame(data=_mesh_traces(src[..., f], labels, data["spacing"], colours,
                                                      op2, step=3), name=str(f + 1)) for f in sel]
                anim = go.Figure(data=fr_objs[0].data, frames=fr_objs)
                anim.update_layout(
                    scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                               zaxis_visible=False),
                    height=560, margin=dict(l=0, r=0, t=0, b=0),
                    updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05,
                                      buttons=[dict(label="▶ Play", method="animate",
                                                    args=[None, dict(frame=dict(duration=120, redraw=True),
                                                                     fromcurrent=True, mode="immediate")])])],
                    sliders=[dict(steps=[dict(method="animate", label=fo.name,
                                              args=[[fo.name], dict(mode="immediate",
                                                                    frame=dict(duration=0, redraw=True))])
                                         for fo in fr_objs], currentvalue=dict(prefix="frame "))])
                st.plotly_chart(anim, width="stretch")

    vox = np.prod(data["spacing"]) / 1000.0
    rows = []
    for name in labels:
        m = src[..., fr] == labels[name]
        mesh = A.surface_mesh(m, data["spacing"])
        area = 0.0
        if mesh is not None:
            v, f = mesh
            tri = v[f]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        vol = m.sum() * vox
        sph = (np.pi ** (1 / 3) * (6 * vol * 1000) ** (2 / 3) / area) if area else np.nan
        rows.append({"Structure": name, "Volume (mL)": vol, "Surface area (cm²)": area / 100,
                     "Sphericity": sph})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    st.caption("Sphericity = π^(1/3)·(6V)^(2/3) / A (1 for a perfect sphere). A more spherical LV "
               "is a recognised sign of adverse remodelling, for example in dilated cardiomyopathy.")


# --------------------------------------------------------------------------------------
# Regional AHA analysis
# --------------------------------------------------------------------------------------
def bullseye(values, title, colorscale, cmin, cmax, unit):
    fig = go.Figure()
    rings = {"basal": (range(1, 7), 2, 60), "mid": (range(7, 13), 1, 60), "apical": (range(13, 17), 0, 90)}
    for _, (segs, base, width) in rings.items():
        segs = list(segs)
        vals = [values.get(s, np.nan) for s in segs]
        fig.add_trace(go.Barpolar(
            r=[1] * len(segs), base=[base] * len(segs), theta=[AHA_THETA[s] for s in segs],
            width=[width] * len(segs),
            marker=dict(color=vals, colorscale=colorscale, cmin=cmin, cmax=cmax,
                        line=dict(color="white", width=2),
                        colorbar=dict(title=unit, len=0.7) if base == 2 else None,
                        showscale=base == 2),
            customdata=[[s, A.AHA_NAMES[s], v] for s, v in zip(segs, vals)],
            hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{customdata[2]:.1f} " + unit + "<extra></extra>",
            showlegend=False))
    for s, th in AHA_THETA.items():
        r = 2.5 if s <= 6 else 1.5 if s <= 12 else 0.55
        v = values.get(s, np.nan)
        label = f"{s}<br>{v:.0f}" if v == v else str(s)
        fig.add_trace(go.Scatterpolar(r=[r], theta=[th], mode="text", text=[label],
                                      textfont=dict(color="black", size=12), showlegend=False,
                                      hoverinfo="skip"))
    fig.update_layout(title=title, height=460, margin=dict(l=20, r=20, t=50, b=20),
                      polar=dict(radialaxis=dict(visible=False, range=[0, 3]),
                                 angularaxis=dict(visible=False, direction="counterclockwise",
                                                  rotation=0)))
    return fig


def section_regional(source, case, data, labels, metric_row):
    st.subheader("Regional myocardial analysis (AHA 16-segment model)")
    st.caption("This follows the regional-phenotype approach of the Imperial UK Biobank pipeline "
               "(Bai et al., JCMR 2018; Nature Medicine 2020). Wall thickness is the distance from "
               "each endocardial boundary point to the epicardium. Segments are anchored at the RV "
               "insertion points and split into basal, mid and apical thirds of the labelled slices.")
    key = _key(source, case)
    r_ed = regional(key, data, labels, data["ed"])
    r_es = regional(key, data, labels, data["es"])
    if r_ed is None or r_es is None:
        st.warning("Could not find RV insertion points or enough labelled slices for this case.")
        return
    wt_ed, oriented = r_ed
    wt_es, _ = r_es
    thick = {k: (wt_es[k] - wt_ed[k]) / wt_ed[k] * 100 if wt_ed[k] == wt_ed[k] and wt_ed[k] else np.nan
             for k in wt_ed}
    n15 = sum(v > 15 for v in wt_ed.values() if v == v)
    metric_row([("Max segmental WT (ED)", f"{np.nanmax(list(wt_ed.values())):.1f} mm"),
                ("Mean WT (ED)", f"{np.nanmean(list(wt_ed.values())):.1f} mm"),
                ("Segments > 15 mm", f"{n15}"),
                ("Mean systolic thickening", f"{np.nanmean(list(thick.values())):.0f}%")])
    maps = {"Wall thickness at ED": (wt_ed, "YlOrRd", 3, 20, "mm"),
            "Wall thickness at ES": (wt_es, "YlOrRd", 3, 25, "mm"),
            "Systolic thickening": (thick, "RdBu", -20, 120, "%")}
    which = st.radio("Bullseye map", list(maps), horizontal=True)
    vals, cs, lo, hi, unit = maps[which]
    st.plotly_chart(bullseye(vals, which, cs, lo, hi, unit), width="stretch")
    if not oriented:
        st.caption("ACDC files carry no patient orientation, so segment names assume the standard "
                   "short-axis view (anterior at the top). The thickness values do not depend on "
                   "this; only the anterior/inferior naming could be mirrored.")
    df = pd.DataFrame({"Segment": list(A.AHA_NAMES), "Name": list(A.AHA_NAMES.values()),
                       "WT ED (mm)": [wt_ed[k] for k in A.AHA_NAMES],
                       "WT ES (mm)": [wt_es[k] for k in A.AHA_NAMES],
                       "Thickening (%)": [thick[k] for k in A.AHA_NAMES]})
    st.dataframe(df.round(1), hide_index=True, width="stretch")
    st.caption("Diastolic thickness above 15 mm in several segments is one of the ACDC criteria for "
               "hypertrophic cardiomyopathy. Reduced thickening in a coronary territory suggests "
               "previous infarction.")


# --------------------------------------------------------------------------------------
# Motion tracking and strain
# --------------------------------------------------------------------------------------
def _dice(a, b):
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def section_motion(source, case, data, labels, colours, metric_row):
    st.subheader("Motion tracking, label propagation and strain")
    st.caption("Dense TV-L1 optical flow (Zach, Pock & Bischof 2007) is estimated between "
               "consecutive frames. The displacement fields are composed back to end-diastole, and "
               "the ED segmentation is propagated through the whole heartbeat. This is a classical "
               "version of the registration-based motion tracking behind the strain measures of "
               "the UK Biobank pipeline (Bai et al. 2020) and the motion correction in Oxford's "
               "automated CMR reporting projects.")
    key = _key(source, case)
    t = data["img"].shape[3]
    frames = [(data["ed"] + k) % t for k in range(t)]
    store = st.session_state.setdefault("_tracking", {})
    if key not in store:
        st.info(f"Motion tracking takes about 10–40 s for this case ({data['img'].shape[2]} slices × {t} frames).")
        if st.button("▶ Run motion tracking", type="primary"):
            bar = st.progress(0.0, text="Tracking myocardial motion…")
            lab, mags = A.propagate_labels(data["img"], data["gt"][..., data["ed"]], data["ed"],
                                           progress=lambda p: bar.progress(min(p, 1.0),
                                                                           text="Tracking myocardial motion…"))
            store[key] = {"labels": lab, "mags": mags}
            bar.empty()
            st.rerun()
    tr = store.get(key)
    vox = np.prod(data["spacing"]) / 1000.0
    levels, _ = _geometry(data, labels)
    mid = [s for s, l in levels.items() if l == "mid"]

    if tr is not None:
        lab = tr["labels"]
        es = data["es"]
        dice_es = {n: _dice(lab[..., es] == labels[n], data["gt"][..., es] == labels[n]) for n in labels}
        metric_row([(f"Dice at ES – {n}", f"{d:.3f}") for n, d in dice_es.items()] +
                   [("Peak motion", f"{tr['mags'].max() * data['spacing'][0]:.1f} mm/frame")])
        st.caption("The ES expert mask is never used by the tracker, so Dice at ES is an independent "
                   "check of the propagated segmentation.")
        vt = pd.DataFrame({"Frame": [f + 1 for f in frames],
                           **{n: [(lab[..., f] == labels[n]).sum() * vox for f in frames] for n in labels}})
        fig = go.Figure()
        for n in labels:
            fig.add_trace(go.Scatter(x=list(range(1, t + 1)), y=vt[n], name=f"{n} (tracked)",
                                     line=dict(color=f"rgb{colours[n]}")))
            gtf = np.where(data["has_gt"])[0]
            fig.add_trace(go.Scatter(x=[(f - data["ed"]) % t + 1 for f in gtf],
                                     y=[(data["gt"][..., f] == labels[n]).sum() * vox for f in gtf],
                                     mode="markers", marker=dict(symbol="x", size=10,
                                                                  color=f"rgb{colours[n]}"),
                                     name=f"{n} (expert)"))
        fig.update_layout(title="Volume–time curves from tracked masks (× = expert)",
                          xaxis_title="Frames after ED", yaxis_title="Volume (mL)", height=400)
        st.plotly_chart(fig, width="stretch")
        if data["has_gt"].all():
            dl = [_dice(lab[..., f] == labels["LV cavity"], data["gt"][..., f] == labels["LV cavity"])
                  for f in frames]
            dm = [_dice(lab[..., f] == labels["Myocardium"], data["gt"][..., f] == labels["Myocardium"])
                  for f in frames]
            f2 = px.line(pd.DataFrame({"Frames after ED": range(1, t + 1), "LV cavity": dl,
                                       "Myocardium": dm}).melt("Frames after ED", var_name="Structure",
                                                               value_name="Dice"),
                         x="Frames after ED", y="Dice", color="Structure", range_y=[0, 1],
                         title="Tracked vs expert masks on every frame")
            st.plotly_chart(f2, width="stretch")
        st.caption("Intensity-based optical flow tends to underestimate contraction, especially in "
                   "vigorously contracting hearts with trabeculated walls. Learned motion models "
                   "(e.g. Qin et al., MICCAI 2018) reduce this drift.")

    lab4d, origin = labels_over_cycle(key, data)
    if lab4d is not None and mid:
        gcs, grs = A.global_strain(lab4d, labels, data["spacing"], frames, mid)
        metric_row([("Peak circumferential strain", f"{np.nanmin(gcs):.1f}%"),
                    ("Peak radial strain", f"{np.nanmax(grs):.1f}%")])
        sf = pd.DataFrame({"Frames after ED": range(1, t + 1), "Circumferential (mid-wall)": gcs,
                           "Radial (wall thickness)": grs}).melt("Frames after ED", var_name="Strain",
                                                                   value_name="%")
        fig = px.line(sf, x="Frames after ED", y="%", color="Strain", markers=True,
                      title=f"Global mid-ventricular strain from {origin}")
        fig.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")
        st.caption("These are segmentation-based strains: circumferential from the change in mid-wall "
                   "perimeter, radial from the change in mean wall thickness, both relative to ED. "
                   "They approximate, but are not identical to, tissue-tracking or tagging strain.")
    elif tr is None:
        st.caption("Strain curves need masks on every frame: run motion tracking first.")


# --------------------------------------------------------------------------------------
# Diagnosis (ACDC)
# --------------------------------------------------------------------------------------
@st.cache_resource
def diagnosis_model():
    df = pd.read_csv(APP_DIR / "data" / "acdc_features.csv")
    cols = A.FEATURE_COLUMNS
    tr, te = df[df.split == "training"], df[df.split == "testing"]
    rf = RandomForestClassifier(n_estimators=500, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(rf, tr[cols], tr.group, cv=cv, method="predict_proba")
    rf.fit(tr[cols], tr.group)
    classes = list(rf.classes_)
    probs = pd.DataFrame(index=df.id, columns=classes, dtype=float)
    probs.loc[tr.id] = oof
    probs.loc[te.id] = rf.predict_proba(te[cols])
    pred = probs.idxmax(axis=1)
    res = dict(
        df=df, probs=probs, pred=pred, classes=classes,
        cv_acc=accuracy_score(tr.group, pred.loc[tr.id]),
        test_acc=accuracy_score(te.group, pred.loc[te.id]),
        cm_test=confusion_matrix(te.group, pred.loc[te.id], labels=GROUP_ORDER),
        importance=pd.Series(rf.feature_importances_, cols).sort_values(),
        rule=df.apply(A.acdc_rule_based, axis=1),
    )
    emb = make_pipeline(StandardScaler(), PCA(2)).fit_transform(df[cols])
    res["pca"] = pd.DataFrame({"PC1": emb[:, 0], "PC2": emb[:, 1], "group": df.group, "id": df.id})
    return res


def section_diagnosis(source, case, data, labels, metric_row):
    st.subheader("Automated diagnosis from segmentation-derived features")
    st.caption("This follows the ACDC diagnosis approach (Isensee et al. 2017; Khened et al., "
               "MedIA 2019). Clinically motivated features are computed from the masks: indexed "
               "volumes, EF, LV mass, RV/LV ratio, wall-thickness statistics and thickening. A random "
               "forest then classifies the 5 groups. Training patients are scored out-of-fold (5-fold "
               "CV), and the 50 official test patients by a model trained only on the 100 training "
               "patients.")
    if source != "ACDC":
        st.info("CMR-MULTI provides no diagnostic labels, height or weight, so the classifier runs on "
                "ACDC cases only.")
        return
    m = diagnosis_model()
    df = m["df"]
    me = df[df.id == case["id"]].iloc[0]
    p = m["probs"].loc[case["id"]]
    pred = p.idxmax()
    metric_row([("True group", f"{case['group']}"),
                ("Model prediction", f"{pred} ({p.max():.0%})"),
                ("Rule-based (ACDC criteria)", m["rule"][df.id == case["id"]].iloc[0]),
                ("CV accuracy (100 train)", f"{m['cv_acc']:.0%}"),
                ("Held-out test accuracy (50)", f"{m['test_acc']:.0%}")])
    c1, c2 = st.columns(2)
    with c1:
        pr = p.reindex(GROUP_ORDER).reset_index()
        pr.columns = ["Group", "Probability"]
        st.plotly_chart(px.bar(pr, x="Group", y="Probability", range_y=[0, 1],
                               title=f"Class probabilities – {case['id']} ({me.split})"), width="stretch")
    with c2:
        fig = px.scatter(m["pca"], x="PC1", y="PC2", color="group", hover_name="id",
                         category_orders={"group": GROUP_ORDER}, title="All 150 patients in feature space (PCA)")
        row = m["pca"][m["pca"].id == case["id"]]
        fig.add_trace(go.Scatter(x=row.PC1, y=row.PC2, mode="markers", name="this patient",
                                 marker=dict(size=18, symbol="star", color="black")))
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        cm = m["cm_test"]
        st.plotly_chart(px.imshow(cm, x=GROUP_ORDER, y=GROUP_ORDER, text_auto=True,
                                  color_continuous_scale="Blues",
                                  labels=dict(x="Predicted", y="True", color="n"),
                                  title="Confusion matrix – 50 held-out test patients"), width="stretch")
    with c4:
        imp = m["importance"].reset_index()
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Random-forest feature importance"), width="stretch")
    med = df.groupby("group")[A.FEATURE_COLUMNS].median().reindex(GROUP_ORDER).T
    med.insert(0, "This patient", me[A.FEATURE_COLUMNS].astype(float))
    st.markdown("**This patient vs group medians**")
    st.dataframe(med.round(1), width="stretch")
    st.caption(f"Rule-based accuracy over all 150 patients: {(m['rule'] == df.group).mean():.0%}. "
               "Features here come from expert masks; a fully automatic system would compute them "
               "from predicted segmentations, so its errors would add to these.")


# --------------------------------------------------------------------------------------
# Quality control and automated report
# --------------------------------------------------------------------------------------
def section_qc_report(source, case, data, labels, volumes_df, metric_row):
    st.subheader("Quality control and automated report")
    st.caption("Oxford's 2026 MRC iCASE DPhil project on accountable AI for CMR uses quality-control "
               "networks that warn when an automated result is unreliable. Here simple anatomical "
               "plausibility checks are applied to the masks before the measurements are reported.")
    key = _key(source, case)
    rows = []
    for ph, f in (("ED", data["ed"]), ("ES", data["es"])):
        for check, ok, detail in A.qc_checks(data["gt"][..., f], labels):
            rows.append({"Phase": ph, "Check": check, "Pass": "✅" if ok else "⚠️", "Detail": detail})
    qc = pd.DataFrame(rows)
    st.dataframe(qc, hide_index=True, width="stretch")

    v = volumes_df.set_index("frame")
    ved, ves = v.loc[data["ed"] + 1], v.loc[data["es"] + 1]
    lvef = (ved["LV cavity"] - ves["LV cavity"]) / ved["LV cavity"] * 100
    rvef = (ved["RV cavity"] - ves["RV cavity"]) / ved["RV cavity"] * 100
    mass = ved["Myocardium"] * 1.05
    reg = regional(key, data, labels, data["ed"])
    name = case["id"] if source == "ACDC" else "CINE_SAX_%03d" % int(case["id"])
    lines = [f"# Automated CMR report – {name}",
             f"*Generated {date.today().isoformat()} by the 4D Cardiac MRI Analysis Dashboard. "
             "Research use only – not a clinical report.*", "",
             f"**Dataset:** {source}  ",
             f"**Matrix:** {' × '.join(map(str, data['img'].shape))} · voxel "
             f"{data['spacing'][0]:.2f} × {data['spacing'][1]:.2f} × {data['spacing'][2]:.1f} mm  ",
             f"**ED / ES frames:** {data['ed'] + 1} / {data['es'] + 1}", "",
             "## Left ventricle",
             f"- EDV {ved['LV cavity']:.0f} mL, ESV {ves['LV cavity']:.0f} mL, "
             f"SV {ved['LV cavity'] - ves['LV cavity']:.0f} mL, EF {lvef:.1f}%",
             f"- Myocardial mass (ED) {mass:.0f} g"]
    flags = []
    if source == "ACDC":
        bsa = np.sqrt(case["height"] * case["weight"] / 3600)
        lines.append(f"- Indexed: EDV {ved['LV cavity'] / bsa:.0f} mL/m², mass {mass / bsa:.0f} g/m² "
                     f"(BSA {bsa:.2f} m²)")
        if ved["LV cavity"] / bsa > 100:
            flags.append("LV EDV index above 100 mL/m² (ACDC DCM criterion)")
        if mass / bsa > 110:
            flags.append("LV mass index above 110 g/m² (ACDC HCM criterion)")
        if ved["RV cavity"] / bsa > 110:
            flags.append("RV EDV index above 110 mL/m² (ACDC abnormal-RV criterion)")
    if lvef < 40:
        flags.append("LVEF below 40%")
    if rvef < 40:
        flags.append("RVEF below 40% (ACDC abnormal-RV criterion)")
    if reg is not None:
        wt = reg[0]
        n15 = sum(x > 15 for x in wt.values() if x == x)
        lines.append(f"- Regional wall thickness (ED): max {np.nanmax(list(wt.values())):.1f} mm, "
                     f"{n15} AHA segment(s) > 15 mm")
        if n15 >= 2:
            flags.append(f"{n15} segments thicker than 15 mm at ED (ACDC HCM criterion)")
    lines += ["", "## Right ventricle",
              f"- EDV {ved['RV cavity']:.0f} mL, ESV {ves['RV cavity']:.0f} mL, EF {rvef:.1f}%", ""]
    if source == "ACDC":
        m = diagnosis_model()
        p = m["probs"].loc[case["id"]]
        lines += ["## Automated classification",
                  f"- Random forest: **{p.idxmax()}** (probability {p.max():.0%}); "
                  f"dataset label: {case['group']}", ""]
    lines += ["## Flags", *(f"- ⚠️ {x}" for x in flags)] if flags else ["## Flags", "- None"]
    nfail = int((qc["Pass"] == "⚠️").sum())
    lines += ["", "## Quality control",
              f"- {len(qc) - nfail}/{len(qc)} plausibility checks passed" +
              (f"; review: {', '.join(qc[qc.Pass == '⚠️'].Check.unique())}" if nfail else "")]
    report = "\n".join(lines)
    metric_row([("QC checks passed", f"{len(qc) - nfail}/{len(qc)}"), ("Flags", str(len(flags)))])
    with st.container(border=True):
        st.markdown(report)
    st.download_button("Download report (Markdown)", report.encode(), file_name="cmr_report.md",
                       mime="text/markdown")


# --------------------------------------------------------------------------------------
# Imaging guide and research context
# --------------------------------------------------------------------------------------
GUIDE = """
### 1. How cine cardiac MRI is acquired
* **Sequence:** balanced steady-state free precession (bSSFP/SSFP) cine gives bright blood and
  dark myocardium, so the cavities stand out without contrast agent.
* **ECG gating:** each image line is acquired over several heartbeats and sorted by cardiac phase,
  so one breath-hold gives a movie of a whole (average) heartbeat. Retrospective gating covers the
  full cycle; prospective gating can miss its last 5–10%.
* **Short-axis stack:** slices are planned perpendicular to the LV long axis from the mitral-valve
  plane (base) to the apex. ACDC used 5–8 mm slices, sometimes with gaps, so slices are 5–10 mm
  apart, with 1.37–1.68 mm pixels and 28–40 frames per cycle.
* **4D = 3D + time:** a stack of ~10 slices × 25–35 frames. This dashboard shows it as
  *slice × frame*.

### 2. Reading the viewer
* **End-diastole (ED)** is the most filled frame (largest LV) and **end-systole (ES)** the most
  contracted.
* Colours: red = LV cavity, blue = LV myocardium, orange = RV cavity.
* Basal slices contain the valve planes and outflow tracts, which are the hardest to segment
  consistently. Apical slices suffer most from partial-volume blurring.

### 3. What each measurement means
| Measurement | How it is computed here | Clinical meaning |
|---|---|---|
| EDV / ESV | voxels × voxel volume at ED / ES | chamber size; dilatation |
| Ejection fraction | (EDV − ESV) / EDV | global systolic function |
| LV mass | myocardial volume × 1.05 g/mL | hypertrophy |
| Indexed values | divided by body-surface area (Mosteller) | compare across body sizes |
| Wall thickness | endocardium → epicardium distance, AHA segments | focal / asymmetric hypertrophy |
| Thickening | (WT_ES − WT_ED) / WT_ED | regional contraction |
| Circumferential / radial strain | change of mid-wall perimeter / wall thickness | myocardial deformation |
| Sphericity | shape of the 3D surface | adverse remodelling |

### 4. ACDC diagnostic groups (as defined by the dataset organisers)
| Group | Definition |
|---|---|
| NOR | normal cardiac anatomy and function |
| MINF | LVEF < 40% with several abnormally contracting segments |
| DCM | LV diastolic volume > 100 mL/m² and LVEF < 40% |
| HCM | LV mass > 110 g/m², several segments > 15 mm thick at diastole, normal EF |
| RV | RV cavity volume > 110 mL/m² or RVEF < 40% |

### 5. Reference ranges
Normal values depend on age, sex, ethnicity, field strength and contouring convention (for example
whether papillary muscles count as myocardium or blood pool). Use published age- and sex-specific
ranges rather than single cut-offs:
* Kawel-Boehm N. *et al.* Reference ranges ("normal values") for CMR in adults and children: 2020
  update. *JCMR* 22:87 (2020). doi:10.1186/s12968-020-00683-3
* Petersen S.E. *et al.* Reference ranges for cardiac structure and function using CMR in
  Caucasians from the UK Biobank population cohort. *JCMR* 19:18 (2017).
* Raisi-Estabragh Z. *et al.* Variation in LV CMR normal reference ranges: systematic review and
  meta-analysis. *Eur Heart J Cardiovasc Imaging* (2020).

### 6. Pitfalls to keep in mind
* **Basal slice selection** changes volumes and EF the most. This is why the CMR-MULTI EF from
  masks can differ from the reported LVEF.
* **Through-plane motion:** the base moves towards the apex in systole, so a fixed slice sees
  different anatomy at ED and ES.
* **Thick slices:** 3D surfaces and apical measurements depend on interpolation between slices.
* **Domain shift:** models trained on one scanner or centre (ACDC) can fail on another (CMR-MULTI).
  That is the reason for testing on both.
"""

RESEARCH = [
    ("Oxford – Zhang group DPhil 2026 (RDM Cardiovascular Medicine)",
     "Automated CMR analysis and reporting with machine-learning pipelines; generative AI for contrast "
     "(Virtual Native Enhancement); AI imaging biomarkers for population studies. Supervisors "
     "Q. Zhang, S. Piechnik, V. Ferreira.",
     "Automated report, cardiac function, regional analysis",
     "https://www.rdm.ox.ac.uk/graduate-study/dphil-in-medical-sciences/dphil-projects/2026-zhuang"),
    ("Oxford – MRC iCASE DPhil 2026: Accountable AI for automated CMR processing",
     "Motion correction, segmentation and landmark detection networks plus quality-control-driven "
     "networks that warn when confidence is low; industrial partner AI4Med.",
     "QC checks, motion tracking, report",
     "https://www.medsci.ox.ac.uk/study/graduateschool/mrcdtp/how-to-apply/icase-2026/developing-accountable-ai-for-automated-cardiovascular-mr-processing-towards-precision-diagnostic-reporting"),
    ("Oxford – Big Data Institute DPhil 2026: multimorbidity trajectories in CVD",
     "AI-enhanced cardiac imaging phenotypes in UK Biobank linked to multi-organ disease.",
     "Feature extraction for population-scale phenotyping",
     "https://www.ndph.ox.ac.uk/study-with-us/dphil-population-health/choose-a-dphil-project-2026/deep-learning-of-multimorbidity-trajectories-in-cardiovascular-disease-capturing-cross-talk-between-diseases-with-ai-and-big-data"),
    ("Imperial – Bai lab UK Biobank CMR pipeline (ukbb_cardiac)",
     "Automated segmentation; global volumes; regional AHA wall thickness; circumferential, radial "
     "and longitudinal strain from motion tracking.",
     "AHA bullseye, strain, volume curves",
     "https://github.com/baiwenjia/ukbb_cardiac"),
    ("Imperial – MeshHeart (Qiao et al., Nature Machine Intelligence 2025)",
     "Generative model of personalised 3D+t heart meshes learned from 38,000 UK Biobank participants; "
     "'latent delta' measures deviation from a personalised normal heart.",
     "3D/4D surface reconstruction",
     "https://www.imperial.ac.uk/news/264275/ai-model-generates-personalised-heart-animations/"),
    ("Imperial – O'Regan Computational Cardiac Imaging group (MRC LMS)",
     "Machine learning on cardiac imaging and genetics, including predicting survival from cardiac "
     "motion and 3D+time heart models; BHF programme on risk in hypertrophic cardiomyopathy.",
     "Wall thickness / HCM criteria, 4D motion",
     "https://lms.mrc.ac.uk/research/computational-cardiac-imaging"),
    ("Imperial – Niederer: scaling cardiac digital twins",
     "Methods to build patient-specific cardiac digital twins at scale from imaging.",
     "Meshes are the geometric starting point of a digital twin",
     "https://imperial.ac.uk/media/imperial-college/medicine/nhli/funding/Scaling-Cardiac-Digital-Twins.pdf"),
    ("ACDC diagnosis benchmarks (Isensee et al. 2017; Khened et al. MedIA 2019)",
     "Segmentation-derived features with ensemble classifiers. Isensee reported 94% 5-fold CV "
     "accuracy on ACDC training data.",
     "Diagnosis tab (random forest on the same feature families)",
     "https://arxiv.org/abs/1707.00587"),
]


def section_guide():
    st.subheader("Imaging guide")
    st.markdown(GUIDE)


def section_research():
    st.subheader("Research context: current DPhil/PhD projects and the methods implemented here")
    st.caption("Projects and groups at Oxford and Imperial whose techniques this dashboard implements in "
               "simplified, CPU-friendly form on the public ACDC and CMR-MULTI data.")
    for title, what, here, url in RESEARCH:
        with st.container(border=True):
            st.markdown(f"**[{title}]({url})**  \n{what}  \n*Implemented here:* {here}")
    st.markdown("""
**Natural next steps for a PhD-level extension**
* Replace the classical baseline with an nnU-Net or a TriFuse-SRNet model and upload its masks in
  *Segmentation metrics*, testing on ACDC → CMR-MULTI to measure domain shift.
* Learn motion with a deep registration network and compare its strain with the TV-L1 tracker here.
* Fit a statistical shape model or a mesh autoencoder (MeshHeart-style) to the 3D+t surfaces and
  test whether a latent-space distance separates the ACDC groups.
* Add uncertainty estimates (test-time augmentation or ensembles) and feed them into the QC checks.
""")

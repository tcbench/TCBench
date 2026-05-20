# ==== CONFIG ====
EVAL_DIR = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/TCBench Results"
GT_FILE = "ibtracs_RI_gt.csv"  # ground truth file saved by you
TREAT_MISSING_AS_NEGATIVE = (
    True  # if a model lacks a (SID, t0, vt) prediction, count as 'no-RI'
)

# ==== IMPORTS ====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== HELPERS ====
KEYS = ["SID", "Initial Time", "Valid Time"]


def _to_dt(df, cols=("Initial Time", "Valid Time")):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _to_bool(series: pd.Series):
    """Robustly coerce a series of RI flags to booleans, preserving NaN."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    mapdict = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    out = s.map(mapdict)
    # Keep original NaN as NaN
    out[series.isna()] = np.nan
    return out


def _pretty_name(fn_no_ext: str):
    # Tidy up filename into a short model label
    name = fn_no_ext
    for tok in ["_RI", "_results", "_clean", "_corrected", "(downloaded)"]:
        name = name.replace(tok, "")
    name = name.replace("2023_", "")
    return name


def _load_gt(path):
    gt = pd.read_csv(path)
    gt = _to_dt(gt)
    # pick RI true column name
    ri_cols = [c for c in gt.columns if c.lower() in {"ri", "ri_true"}]
    if not ri_cols:
        raise ValueError(
            f"No RI column found in {path} (looked for 'RI' or 'RI_true')."
        )
    gt = gt[KEYS + [ri_cols[0]]].rename(columns={ri_cols[0]: "RI_true"})
    gt["RI_true"] = _to_bool(gt["RI_true"]).fillna(
        False
    )  # GT should be defined; default False if missing
    return gt


def _load_model_ri(path):
    df = pd.read_csv(path)
    df = _to_dt(df)
    # predicted RI column
    ri_cols = [
        c
        for c in df.columns
        if c.lower() in {"ri", "ri_pred", "ri_hat", "ri_prediction"}
    ]
    if not ri_cols:
        raise ValueError(f"No RI prediction column in {path}")
    return df[KEYS + [ri_cols[0]]].rename(columns={ri_cols[0]: "RI_pred"})


def _confusion_and_scores(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    # align to ground truth keys, keep all GT rows
    merged = gt_df.merge(pred_df, on=KEYS, how="left")
    # coerce preds to bool
    merged["RI_pred"] = _to_bool(merged["RI_pred"])
    coverage = merged["RI_pred"].notna().mean()  # fraction of GT keys with a prediction
    if TREAT_MISSING_AS_NEGATIVE:
        merged["RI_pred"] = merged["RI_pred"].fillna(False)
    else:
        # drop rows without prediction if not penalizing missing
        merged = merged.dropna(subset=["RI_pred"])

    y = merged["RI_true"].astype(bool).values
    yhat = merged["RI_pred"].astype(bool).values

    TP = int(np.sum((y == True) & (yhat == True)))
    FP = int(np.sum((y == False) & (yhat == True)))
    FN = int(np.sum((y == True) & (yhat == False)))
    TN = int(np.sum((y == False) & (yhat == False)))
    N = int(len(merged))
    RI_pos = int(np.sum(y))

    # CSI = TP / (TP+FP+FN)
    denom_csi = TP + FP + FN
    CSI = TP / denom_csi if denom_csi > 0 else np.nan

    # Peirce Skill Score (Hanssen–Kuipers): TPR - FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    PSS = (TPR - FPR) if (not np.isnan(TPR) and not np.isnan(FPR)) else np.nan

    return {
        "CSI": CSI,
        "PSS": PSS,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "N": N,
        "RI_pos": RI_pos,
        "coverage": coverage,
    }


# ==== LOAD DATA ====
gt_path = os.path.join(EVAL_DIR, GT_FILE)
gt = _load_gt(gt_path)

model_files = [
    f
    for f in os.listdir(EVAL_DIR)
    if f.endswith("_RI.csv") and f != GT_FILE and "ibtracs_ri_gt" not in f.lower()
]
if not model_files:
    raise FileNotFoundError("No *_RI.csv model files found.")

rows = []
for fn in sorted(model_files):
    try:
        pred = _load_model_ri(os.path.join(EVAL_DIR, fn))
        scores = _confusion_and_scores(gt, pred)
        label = _pretty_name(os.path.splitext(fn)[0])
        rows.append({"model": label, **scores})
    except Exception as e:
        print(f"⚠️ Skipping {fn}: {e}")

scores_df = pd.DataFrame(rows)
if scores_df.empty:
    raise RuntimeError("No valid model RI files after loading.")

# ==== RI SKILL BY LEAD TIME (CSI & PSS curves, no coverage panel) ====


def _lead_hours(df):
    """Compute integer lead time in hours from Initial/Valid Time."""
    return (
        ((df["Valid Time"] - df["Initial Time"]).dt.total_seconds() / 3600.0)
        .round()
        .astype("Int64")
    )


def _per_lead_scores(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align prediction to GT keys, coerce RI to booleans, optionally treat missing as negative,
    and compute TP/FP/FN/TN, CSI, PSS per lead_hours.
    Returns DataFrame indexed by lead_hours with columns [CSI, PSS, N, RI_pos].
    """
    merged = gt_df.merge(pred_df, on=KEYS, how="left")  # keep all GT keys
    merged["RI_pred"] = _to_bool(merged["RI_pred"])
    if TREAT_MISSING_AS_NEGATIVE:
        merged["RI_pred"] = merged["RI_pred"].fillna(False)
    else:
        merged = merged.dropna(subset=["RI_pred"])

    # compute lead hours and drop rows without times
    merged = merged.dropna(subset=["Initial Time", "Valid Time"]).copy()
    merged["lead_hours"] = _lead_hours(merged)

    # boolean arrays
    m = merged["RI_pred"].astype(bool)
    t = merged["RI_true"].astype(bool)

    # precompute masks
    TPm = m & t
    FPm = m & ~t
    FNm = ~m & t
    TNm = ~m & ~t

    # group by lead
    g = merged.groupby("lead_hours", dropna=False)
    TP = g.apply(
        lambda d: int(((d["RI_pred"].astype(bool)) & (d["RI_true"].astype(bool))).sum())
    )
    FP = g.apply(
        lambda d: int(
            ((d["RI_pred"].astype(bool)) & (~d["RI_true"].astype(bool))).sum()
        )
    )
    FN = g.apply(
        lambda d: int((~d["RI_pred"].astype(bool) & (d["RI_true"].astype(bool))).sum())
    )
    TN = g.apply(
        lambda d: int((~d["RI_pred"].astype(bool) & (~d["RI_true"].astype(bool))).sum())
    )
    N = g.size().astype(int)
    RI_pos = g["RI_true"].sum().astype(int)

    # metrics
    CSI = TP / (TP + FP + FN).replace(0, np.nan)
    TPR = TP / (TP + FN).replace(0, np.nan)
    FPR = FP / (FP + TN).replace(0, np.nan)
    PSS = TPR - FPR

    out = pd.DataFrame({"CSI": CSI, "PSS": PSS, "N": N, "RI_pos": RI_pos}).sort_index()
    out.index.name = "lead_hours"
    return out


# Build per-model curves
curves = {}  # model -> per-lead DataFrame
for fn in sorted(model_files):
    try:
        pred = _load_model_ri(os.path.join(EVAL_DIR, fn))
        label = _pretty_name(os.path.splitext(fn)[0])
        curves[label] = _per_lead_scores(gt, pred)
    except Exception as e:
        print(f"⚠️ Skipping {fn}: {e}")

# ===================== FIGURE A — OVERALL SKILL BY MODEL (BARS) =====================
# Uses `scores_df` built above

import matplotlib as mpl

# Order by CSI and set consistent colors
disp = scores_df.sort_values("CSI", ascending=False).reset_index(drop=True)
labels_bar = disp["model"].tolist()
ypos = np.arange(len(disp))

cmap = mpl.cm.get_cmap("tab10")
color_map = {m: cmap(i % 10) for i, m in enumerate(labels_bar)}
bar_colors = [color_map[m] for m in labels_bar]

figA, (ax_csi_bar, ax_pss_bar) = plt.subplots(
    1, 2, figsize=(12, max(4.8, 0.6 * len(disp))), sharey=True
)

# CSI bars
ax_csi_bar.barh(ypos, disp["CSI"].values, color=bar_colors, edgecolor="none")
ax_csi_bar.set_yticks(ypos)
ax_csi_bar.set_yticklabels(labels_bar, fontsize=10)
ax_csi_bar.invert_yaxis()
ax_csi_bar.set_xlabel("Critical Success Index (CSI)")
ax_csi_bar.set_title("RI — CSI (overall)")
xmax_csi = (
    float(np.nanmax(disp["CSI"].values))
    if np.isfinite(np.nanmax(disp["CSI"].values))
    else 0.0
)
ax_csi_bar.set_xlim(0, max(0.05, xmax_csi * 1.15))
for y, val in enumerate(disp["CSI"].values):
    if np.isfinite(val):
        ax_csi_bar.text(
            val + ax_csi_bar.get_xlim()[1] * 0.01,
            y,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
ax_csi_bar.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.8)

# PSS bars
ax_pss_bar.axvline(0.0, color="0.35", lw=1)
ax_pss_bar.barh(ypos, disp["PSS"].values, color=bar_colors, edgecolor="none")
ax_pss_bar.set_yticks(ypos)
ax_pss_bar.set_yticklabels([])  # names on CSI panel
ax_pss_bar.invert_yaxis()
ax_pss_bar.set_xlabel("Peirce Skill Score (PSS = TPR − FPR)")
ax_pss_bar.set_title("RI — PSS (overall)")
vmax = (
    float(np.nanmax(np.abs(disp["PSS"].values)))
    if np.isfinite(np.nanmax(np.abs(disp["PSS"].values)))
    else 0.1
)
ax_pss_bar.set_xlim(-vmax * 1.15, vmax * 1.15)
for y, val in enumerate(disp["PSS"].values):
    if np.isfinite(val):
        ha = "left" if val >= 0 else "right"
        pad = ax_pss_bar.get_xlim()[1] * 0.01 * (1 if val >= 0 else -1)
        ax_pss_bar.text(val + pad, y, f"{val:+.3f}", va="center", ha=ha, fontsize=9)
ax_pss_bar.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.8)

legend_handles = [mpl.patches.Patch(color=color_map[m], label=m) for m in labels_bar]
figA.legend(
    legend_handles,
    [m for m in labels_bar],
    loc="lower center",
    ncols=min(5, len(labels_bar)),
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=9,
)

figA.suptitle("Rapid Intensification skill — overall by model", fontsize=14, y=0.98)
# figA.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# ===================== FIGURE B — SKILL BY LEAD TIME (CURVES, WITH CAPS) =====================
# Uses `curves` dict that we already built above

if not curves:
    raise RuntimeError("No per-lead RI curves were computed.")

# Consistent colors per model
labels_curve = list(curves.keys())
cmap = mpl.cm.get_cmap("tab10")
color_map_curve = {lab: cmap(i % 10) for i, lab in enumerate(labels_curve)}

figB, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
ax_csi, ax_pss = axes

# CSI by lead (cap at 0.2)
for lab in labels_curve:
    df = curves[lab]
    x = df.index.values.astype(float)
    y = df["CSI"].values
    ax_csi.plot(
        x,
        y,
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        label=lab,
        color=color_map_curve[lab],
    )
ax_csi.set_title("RI — Critical Success Index (CSI) by lead")
ax_csi.set_xlabel("Lead time (hours)")
ax_csi.set_ylabel("CSI")
ax_csi.set_xlim(6, 120)
ax_csi.set_ylim(0, 0.2)  # << cap as requested
ax_csi.grid(True, alpha=0.3)

# PSS by lead (cap at ±0.2)
for lab in labels_curve:
    df = curves[lab]
    x = df.index.values.astype(float)
    y = df["PSS"].values
    ax_pss.plot(
        x,
        y,
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        label=lab,
        color=color_map_curve[lab],
    )
ax_pss.axhline(0.0, color="0.35", lw=1)
ax_pss.set_title("RI — Peirce Skill Score (PSS) by lead")
ax_pss.set_xlabel("Lead time (hours)")
ax_pss.set_ylabel("PSS = TPR − FPR")
ax_pss.set_xlim(6, 120)
ax_pss.set_ylim(-0.05, 0.2)  
ax_pss.grid(True, alpha=0.3)

# Shared legend (moved above)
handles, labels_ = ax_csi.get_legend_handles_labels()
figB.legend(
    handles,
    labels_,
    loc="lower center",
    ncols=min(5, len(labels_curve)),
    bbox_to_anchor=(0.5, 1.08),
    frameon=False,
    fontsize=9,
)

figB.suptitle(
    "Rapid Intensification skill by lead time (vs ibtracs_RI_gt.csv)",
    fontsize=14,
    y=0.99,
)
# figB.tight_layout(rect=[0, 0.02, 1, 0.88])
plt.show()

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTPUT_DIR = Path("outputs/metrics/exp3_python_reasoning_split")

# Must match Experiments 1, 2, 5
LABEL_COL = "gt_binary"
RANDOM_STATE = 42

# Fixed Experiment 3 settings (same as static fixed-gate baseline)
GATE_MAX_TTI = 3.5
GATE_SYNC_TTI = 0.2
TTC_THR = 1.347
DRAC_THR = 3.8371696339999786

BASE_IRI = "http://example.org/avrisk#"

def safe_div(a, b):
    return a / b if b else 0.0

def predict_static_fixed_gate(df):
    active = (
        df["ego_tti_s"].notna() &
        df["other_tti_s"].notna() &
        df["TTC_conflict"].notna() &
        df["DRAC"].notna() &
        (df["ego_tti_s"] >= 0) &
        (df["other_tti_s"] >= 0) &
        (df["TTC_conflict"] > 0) &
        (np.maximum(df["ego_tti_s"], df["other_tti_s"]) <= GATE_MAX_TTI) &
        (np.abs(df["ego_tti_s"] - df["other_tti_s"]) <= GATE_SYNC_TTI)
    )
    pred = active & (
        (df["TTC_conflict"] <= TTC_THR) |
        (df["DRAC"] >= DRAC_THR)
    )
    return pred.astype(np.int8)

def compute_metrics(df, pred_col="exp3_pred"):
    y = (df[LABEL_COL].astype(str).str.lower() == "high").astype(np.int8).to_numpy()
    yhat = df[pred_col].to_numpy().astype(np.int8)
    p, r, f2, _ = precision_recall_fscore_support(y, yhat, average="binary", beta=2, zero_division=0)
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tp = int(((y == 1) & (yhat == 1)).sum())
    spec = safe_div(tn, tn + fp)

    ep = df.groupby("scenario_id", dropna=False).agg(
        has_collision=(LABEL_COL, lambda s: (s.astype(str).str.lower() == "high").any()),
        warned=(pred_col, "max")
    ).reset_index()
    pos = ep[ep["has_collision"]]
    neg = ep[~ep["has_collision"]]
    episode_warning_recall = float(pos["warned"].mean()) if len(pos) else 0.0
    episode_false_alarm_rate = float(neg["warned"].mean()) if len(neg) else 0.0

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(p), "recall": float(r), "f2": float(f2),
        "specificity": float(spec),
        "episode_warning_recall": episode_warning_recall,
        "episode_false_alarm_rate": episode_false_alarm_rate,
    }

def sanitize(text):
    s = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "x"

def write_ttl(df, path: Path):
    lines = []
    lines.append(f'@prefix : <{BASE_IRI}> .')
    lines.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    lines.append('')
    lines.append(':highRisk a :HighRisk .')
    lines.append(':safeRisk a :Safe .')
    lines.append('')

    for row in df.itertuples(index=False):
        sid = getattr(row, "scenario_id")
        fid = int(getattr(row, "frame"))
        enc = f'enc_{sanitize(sid)}_{fid}'
        risk = "highRisk" if int(getattr(row, "exp3_pred")) == 1 else "safeRisk"
        lines.append(f':{enc} a :Encounter ;')
        lines.append(f'  :frameId "{fid}"^^xsd:int ;')
        lines.append(f'  :timestampS "{float(getattr(row, "time"))}"^^xsd:double ;')
        lines.append(f'  :scenarioId "{sid}"^^xsd:string ;')
        lines.append(f'  :hasRiskLevel :{risk} .')
        lines.append('')
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required = ["frame","time","scenario_id","ego_tti_s","other_tti_s","TTC_conflict","DRAC",LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keep = (
        df[LABEL_COL].notna() &
        df["ego_tti_s"].notna() &
        df["other_tti_s"].notna() &
        df["TTC_conflict"].notna() &
        df["DRAC"].notna() &
        (df["ego_tti_s"] >= 0) &
        (df["other_tti_s"] >= 0) &
        (df["TTC_conflict"] > 0)
    )
    df = df.loc[keep].copy()

    groups = df["scenario_id"].astype(str)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, tmp_idx = next(gss1.split(df, groups=groups))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp["scenario_id"].astype(str)))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()

    # No training needed for Exp 3 static ontology equivalent; predict on all splits for completeness
    train["exp3_pred"] = predict_static_fixed_gate(train)
    val["exp3_pred"] = predict_static_fixed_gate(val)
    test["exp3_pred"] = predict_static_fixed_gate(test)

    summary = pd.DataFrame([
        {"split": "train", "model": "exp3_python_static_ontology_equivalent", **compute_metrics(train)},
        {"split": "val", "model": "exp3_python_static_ontology_equivalent", **compute_metrics(val)},
        {"split": "test", "model": "exp3_python_static_ontology_equivalent", **compute_metrics(test)},
    ])
    summary.to_csv(OUTPUT_DIR / "exp3_split_summary.csv", index=False)

    # Main apples-to-apples result for comparison with Exp1/Exp2/Exp5
    test_summary = summary[summary["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    test_summary.to_csv(OUTPUT_DIR / "exp3_test_summary.csv", index=False)

    keep_cols = [
        "frame","time","scenario_id","weather","ego_type","other_type",
        "ego_tti_s","other_tti_s","TTC_conflict","DRAC",LABEL_COL,"exp3_pred"
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    test[keep_cols].to_csv(OUTPUT_DIR / "exp3_test_inferred_labels.csv", index=False)
    write_ttl(test, OUTPUT_DIR / "exp3_test_inferred_risk.ttl")

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['scenario_id'].astype(str).nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")
    print("\n=== Experiment 3 Python reasoning summary by split ===")
    print(summary.to_string(index=False))
    print(f"\nSaved files in: {OUTPUT_DIR.resolve()}")
    print("- exp3_split_summary.csv")
    print("- exp3_test_summary.csv")
    print("- exp3_test_inferred_labels.csv")
    print("- exp3_test_inferred_risk.ttl")

if __name__ == "__main__":
    main()
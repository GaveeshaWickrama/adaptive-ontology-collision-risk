#!/usr/bin/env python3
"""
Experiment 4: Fuzzy ontology-inspired reasoning on the final labeled dataset.

Usage:
    py exp4_fuzzy_ontology_inspired_reasoning.py

Expected input file in the same folder:
    type22_final_labeled_dataset.csv

What it does:
- Uses the same grouped train/validation/test split style as Experiments 1–3
- Tunes fuzzy membership breakpoints and final decision threshold on validation
- Evaluates on train / val / test
- Saves split summary, test inferred labels, test TTL, and config
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# =========================
# CONFIG
# =========================
INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTDIR = Path("outputs/metrics/exp4_fuzzy_ontology_inspired_reasoning")
RANDOM_STATE = 42
LABEL_COL = "gt_binary"

# =========================
# HELPERS
# =========================
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def ensure_binary_ground_truth(series: pd.Series) -> pd.Series:
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    positive_tokens = {
        "1", "true", "yes", "high", "highrisk", "collisionlikely",
        "positive", "risk", "warn", "warning"
    }
    return lowered.isin(positive_tokens).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion(y_true, y_pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    beta2 = 4.0
    f2 = safe_div((1 + beta2) * precision * recall, beta2 * precision + recall)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "specificity": specificity,
    }


def episode_metrics(df: pd.DataFrame, episode_col: str, gt_col: str, pred_col: str) -> Dict[str, float]:
    grouped = df.groupby(episode_col, sort=False)
    ep_gt = grouped[gt_col].max().astype(int)
    ep_pred = grouped[pred_col].max().astype(int)

    pos = ep_gt == 1
    neg = ep_gt == 0

    episode_warning_recall = safe_div(int(((ep_pred == 1) & pos).sum()), int(pos.sum()))
    episode_false_alarm_rate = safe_div(int(((ep_pred == 1) & neg).sum()), int(neg.sum()))

    return {
        "episode_warning_recall": episode_warning_recall,
        "episode_false_alarm_rate": episode_false_alarm_rate,
        "n_episodes": int(len(ep_gt)),
    }


# =========================
# FUZZY MEMBERSHIPS
# =========================
def decreasing_trapezoid(x: np.ndarray, full_at: float, zero_at: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=float)
    out[x >= zero_at] = 0.0
    mask = (x > full_at) & (x < zero_at)
    out[mask] = (zero_at - x[mask]) / (zero_at - full_at)
    return np.clip(out, 0.0, 1.0)


def increasing_trapezoid(x: np.ndarray, zero_at: float, full_at: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    out[x >= full_at] = 1.0
    mask = (x > zero_at) & (x < full_at)
    out[mask] = (x[mask] - zero_at) / (full_at - zero_at)
    return np.clip(out, 0.0, 1.0)


# =========================
# FUZZY REASONING
# =========================
def compute_fuzzy_score(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()

    ttc = out["TTC_conflict"].astype(float).to_numpy()
    drac = out["DRAC"].astype(float).to_numpy()
    ego_tti = out["ego_tti_s"].astype(float).to_numpy()
    other_tti = out["other_tti_s"].astype(float).to_numpy()

    max_tti = np.maximum(ego_tti, other_tti)
    sync_gap = np.abs(ego_tti - other_tti)

    # Gate memberships
    mu_near_zone = decreasing_trapezoid(max_tti, params["gate_time_full"], params["gate_time_zero"])
    mu_sync = decreasing_trapezoid(sync_gap, params["sync_full"], params["sync_zero"])
    mu_gate = np.minimum(mu_near_zone, mu_sync)

    # Primary risk memberships
    mu_low_ttc = decreasing_trapezoid(ttc, params["ttc_full"], params["ttc_zero"])
    mu_high_drac = increasing_trapezoid(drac, params["drac_zero"], params["drac_full"])

    # OR-like aggregation for severity
    mu_primary = 1.0 - (1.0 - mu_low_ttc) * (1.0 - mu_high_drac)

    # AND-like aggregation with gate
    risk_score = np.minimum(mu_gate, mu_primary)

    out["max_tti"] = max_tti
    out["sync_gap"] = sync_gap
    out["mu_near_zone"] = mu_near_zone
    out["mu_sync"] = mu_sync
    out["mu_gate"] = mu_gate
    out["mu_low_ttc"] = mu_low_ttc
    out["mu_high_drac"] = mu_high_drac
    out["fuzzy_risk_score"] = np.clip(risk_score, 0.0, 1.0)

    return out


def evaluate_split(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    pred = (df["fuzzy_risk_score"].to_numpy() >= threshold).astype(int)
    y_true = df["gt_binary_num"].to_numpy().astype(int)

    fm = frame_metrics(y_true, pred)

    tmp = df.copy()
    tmp["pred_binary"] = pred
    em = episode_metrics(tmp, "episode_id", "gt_binary_num", "pred_binary")

    return {**fm, **em}


# =========================
# TTL EXPORT
# =========================
def ttl_safe(s: object) -> str:
    text = str(s)
    for ch in ['\\', '"', ' ', '/', ':', '#', '.', '-']:
        text = text.replace(ch, "_")
    return text


def export_test_ttl(test_df: pd.DataFrame, outpath: Path) -> None:
    lines = []
    lines.append('@prefix ex: <http://example.org/exp4#> .')
    lines.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    lines.append("")
    lines.append("ex:FuzzyInteractionFrame a ex:OntologyClass .")
    lines.append("ex:LowTTC a ex:FuzzyConcept .")
    lines.append("ex:HighDRAC a ex:FuzzyConcept .")
    lines.append("ex:StrongInteraction a ex:FuzzyConcept .")
    lines.append("ex:ElevatedCollisionRisk a ex:FuzzyConcept .")
    lines.append("")

    for _, row in test_df.iterrows():
        ep = ttl_safe(row["episode_id"])
        fr = ttl_safe(row["frame"])
        node = f"ex:frame_{ep}_{fr}"

        lines.append(f"{node} a ex:FuzzyInteractionFrame ;")
        lines.append(f'    ex:muLowTTC "{float(row["mu_low_ttc"]):.6f}"^^xsd:double ;')
        lines.append(f'    ex:muHighDRAC "{float(row["mu_high_drac"]):.6f}"^^xsd:double ;')
        lines.append(f'    ex:muGate "{float(row["mu_gate"]):.6f}"^^xsd:double ;')
        lines.append(f'    ex:riskScore "{float(row["fuzzy_risk_score"]):.6f}"^^xsd:double ;')
        lines.append(f'    ex:predictedWarning "{int(row["pred_binary"])}"^^xsd:int .')
        lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# =========================
# MAIN
# =========================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = [
        "scenario_id", "frame", "TTC_conflict", "DRAC",
        "ego_tti_s", "other_tti_s", LABEL_COL
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["episode_id"] = df["scenario_id"].astype(str)
    df["gt_binary_num"] = ensure_binary_ground_truth(df[LABEL_COL])

    keep = (
        df["TTC_conflict"].notna() &
        df["DRAC"].notna() &
        df["ego_tti_s"].notna() &
        df["other_tti_s"].notna() &
        (df["TTC_conflict"] > 0) &
        (df["ego_tti_s"] >= 0) &
        (df["other_tti_s"] >= 0)
    )
    df = df.loc[keep].copy()

    # Same grouped split style as Experiments 1–3
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, tmp_idx = next(gss1.split(df, groups=df["episode_id"]))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp["episode_id"]))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")

    # Parameter search space
    grid = {
        "gate_time_full": [1.0, 1.5],
        "gate_time_zero": [3.0, 3.5, 4.0],
        "sync_full": [0.10, 0.15],
        "sync_zero": [0.30, 0.40],
        "ttc_full": [0.8, 1.0],
        "ttc_zero": [1.8, 2.0, 2.2],
        "drac_zero": [2.8, 3.2, 3.6],
        "drac_full": [4.2, 4.6, 5.0],
    }
    threshold_candidates = np.round(np.arange(0.10, 0.96, 0.05), 2)

    keys = list(grid.keys())
    best = None
    best_obj = -1.0

    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))

        if not (params["gate_time_zero"] > params["gate_time_full"]):
            continue
        if not (params["sync_zero"] > params["sync_full"]):
            continue
        if not (params["ttc_zero"] > params["ttc_full"]):
            continue
        if not (params["drac_full"] > params["drac_zero"]):
            continue

        val_scored = compute_fuzzy_score(val, params)

        for thr in threshold_candidates:
            metrics = evaluate_split(val_scored, float(thr))
            objective = (
                metrics["f2"]
                + 1e-6 * metrics["episode_warning_recall"]
                + 1e-9 * metrics["specificity"]
            )

            if objective > best_obj:
                best_obj = objective
                best = {
                    "params": params,
                    "threshold": float(thr),
                    "val_metrics": metrics,
                }

    if best is None:
        raise RuntimeError("No valid fuzzy configuration found.")

    print("\n=== Best validation configuration ===")
    for k, v in best["params"].items():
        print(f"{k}: {v}")
    print(f"threshold: {best['threshold']:.2f}")

    rows = []
    test_final = None

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        scored = compute_fuzzy_score(split_df, best["params"])
        metrics = evaluate_split(scored, best["threshold"])

        row = {
            "split": split_name,
            "model": "exp4_fuzzy_ontology_inspired_reasoning",
            **best["params"],
            "threshold": best["threshold"],
            **metrics,
        }
        rows.append(row)

        print(f"\n=== {split_name.upper()} summary ===")
        print(
            f"tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']} "
            f"precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
            f"f2={metrics['f2']:.6f} specificity={metrics['specificity']:.6f} "
            f"episode_warning_recall={metrics['episode_warning_recall']:.6f} "
            f"episode_false_alarm_rate={metrics['episode_false_alarm_rate']:.6f}"
        )

        if split_name == "test":
            scored = scored.copy()
            scored["pred_binary"] = (scored["fuzzy_risk_score"] >= best["threshold"]).astype(int)
            test_final = scored

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTDIR / "exp4_split_summary.csv", index=False)

    with open(OUTDIR / "exp4_best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": INPUT_CSV,
                "random_state": RANDOM_STATE,
                "label_col": LABEL_COL,
                "best_params": best["params"],
                "threshold": best["threshold"],
                "validation_metrics": best["val_metrics"],
            },
            f,
            indent=2,
        )

    if test_final is not None:
        test_final.to_csv(OUTDIR / "exp4_test_inferred_labels.csv", index=False)
        export_test_ttl(test_final, OUTDIR / "exp4_test_inferred_risk.ttl")

    print(f"\nSaved files in: {OUTDIR.resolve()}")
    print("- exp4_best_config.json")
    print("- exp4_split_summary.csv")
    print("- exp4_test_inferred_labels.csv")
    print("- exp4_test_inferred_risk.ttl")


if __name__ == "__main__":
    main()
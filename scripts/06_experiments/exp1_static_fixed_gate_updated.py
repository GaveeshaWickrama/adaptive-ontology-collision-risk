import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

# =========================
# CONFIG
# =========================
INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTPUT_DIR = Path("outputs/metrics/exp1_static_fixed_gate")

LABEL_COL = "gt_binary"
RANDOM_STATE = 42

# Fixed interaction gate
GATE_MAX_TTI = 3.5
GATE_SYNC_TTI = 0.2

# Fixed thresholds (from your earlier best static run)
BASE_TTC_THR = 1.347
BASE_DRAC_THR = 3.8371696339999786


# =========================
# HELPERS
# =========================
def safe_div(a, b):
    return a / b if b else 0.0


def build_episode_id(df: pd.DataFrame) -> pd.Series:
    if "scenario_id" in df.columns:
        return df["scenario_id"].astype(str)
    return df["ego_id"].astype(str) + "_" + df["other_id"].astype(str)


def gate_active(ego_tti: np.ndarray, other_tti: np.ndarray) -> np.ndarray:
    valid = np.isfinite(ego_tti) & np.isfinite(other_tti) & (ego_tti >= 0) & (other_tti >= 0)
    return (
        valid &
        (np.maximum(ego_tti, other_tti) <= GATE_MAX_TTI) &
        (np.abs(ego_tti - other_tti) <= GATE_SYNC_TTI)
    )


def predict_static_fixed_gate(df: pd.DataFrame) -> np.ndarray:
    active = gate_active(df["ego_tti_s"].to_numpy(), df["other_tti_s"].to_numpy())
    pred = active & (
        (df["TTC_conflict"].to_numpy() <= BASE_TTC_THR) |
        (df["DRAC"].to_numpy() >= BASE_DRAC_THR)
    )
    return pred.astype(np.int8)


def compute_frame_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f2, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", beta=2, zero_division=0
    )
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    specificity = safe_div(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(p),
        "recall": float(r),
        "f2": float(f2),
        "specificity": float(specificity),
    }


def build_episode_arrays(df_eval: pd.DataFrame):
    ep_codes, ep_uniques = pd.factorize(df_eval["episode_id"], sort=False)
    ep_count = len(ep_uniques)
    y = df_eval["y"].to_numpy().astype(np.int8)
    ep_pos = np.bincount(ep_codes, weights=y, minlength=ep_count) > 0
    return ep_codes.astype(np.int32), ep_pos.astype(bool), ep_count


def compute_episode_metrics(pred: np.ndarray, ep_codes: np.ndarray, ep_pos: np.ndarray, ep_count: int):
    warned = np.bincount(ep_codes, weights=pred.astype(np.int8), minlength=ep_count) > 0
    pos = ep_pos
    neg = ~ep_pos
    return {
        "episode_warning_recall": float(warned[pos].mean() if pos.any() else 0.0),
        "episode_false_alarm_rate": float(warned[neg].mean() if neg.any() else 0.0),
    }


def evaluate(df_eval: pd.DataFrame, pred: np.ndarray, model_name: str):
    y = df_eval["y"].to_numpy().astype(np.int8)
    ep_codes, ep_pos, ep_count = build_episode_arrays(df_eval)
    fm = compute_frame_metrics(y, pred)
    em = compute_episode_metrics(pred, ep_codes, ep_pos, ep_count)
    return {"model": model_name, **fm, **em}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["episode_id"] = build_episode_id(df)

    keep = (
        df[LABEL_COL].notna() &
        df["TTC_conflict"].notna() &
        df["DRAC"].notna() &
        df["ego_tti_s"].notna() &
        df["other_tti_s"].notna() &
        (df["TTC_conflict"] > 0) &
        (df["ego_tti_s"] >= 0) &
        (df["other_tti_s"] >= 0)
    )
    df = df.loc[keep].copy()
    df["y"] = (df[LABEL_COL].astype(str).str.lower() == "high").astype(np.int8)

    # Same grouped split logic used elsewhere
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, tmp_idx = next(gss1.split(df, groups=df["episode_id"]))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp["episode_id"]))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()

    pred_train = predict_static_fixed_gate(train)
    pred_val = predict_static_fixed_gate(val)
    pred_test = predict_static_fixed_gate(test)

    train_summary = evaluate(train, pred_train, "exp1_static_fixed_gate")
    val_summary = evaluate(val, pred_val, "exp1_static_fixed_gate")
    test_summary = evaluate(test, pred_test, "exp1_static_fixed_gate")

    summary = pd.DataFrame([train_summary, val_summary, test_summary], index=["train", "val", "test"]).reset_index()
    summary = summary.rename(columns={"index": "split"})
    summary.to_csv(OUTPUT_DIR / "exp1_split_summary.csv", index=False)

    test_out = test.copy()
    test_out["pred"] = pred_test
    test_out.to_csv(OUTPUT_DIR / "exp1_test_predictions.csv", index=False)

    with open(OUTPUT_DIR / "exp1_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_csv": INPUT_CSV,
            "label_col": LABEL_COL,
            "random_state": RANDOM_STATE,
            "gate_max_tti": GATE_MAX_TTI,
            "gate_sync_tti": GATE_SYNC_TTI,
            "base_ttc_thr": BASE_TTC_THR,
            "base_drac_thr": BASE_DRAC_THR,
        }, f, indent=2)

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")
    print(f"Fixed gate: max_tti <= {GATE_MAX_TTI}, |ego_tti-other_tti| <= {GATE_SYNC_TTI}")
    print(f"Base thresholds: TTC <= {BASE_TTC_THR:.3f}, DRAC >= {BASE_DRAC_THR:.3f}")

    print("\n=== Split summary ===")
    print(summary.to_string(index=False))

    print(f"\nSaved files in: {OUTPUT_DIR.resolve()}")
    print("- exp1_config.json")
    print("- exp1_split_summary.csv")
    print("- exp1_test_predictions.csv")


if __name__ == "__main__":
    main()
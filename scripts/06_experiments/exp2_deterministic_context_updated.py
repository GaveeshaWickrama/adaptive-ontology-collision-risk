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
OUTPUT_DIR = Path("outputs/metrics/exp2_deterministic_context")

LABEL_COL = "gt_binary"
RANDOM_STATE = 42

# Same fixed interaction gate
GATE_MAX_TTI = 3.5
GATE_SYNC_TTI = 0.2

# Base thresholds from Exp 1
BASE_TTC_THR = 1.347
BASE_DRAC_THR = 3.8371696339999786

# =====================================================
# STUDY-SPECIFIC DETERMINISTIC CONTEXT PRIORS
# These are not exact paper values.
# They are directional priors:
# - more adverse weather / lower visibility -> earlier warning
# - larger / heavier vehicles -> earlier warning
# =====================================================
MODIFIERS = {
    "weather_ctx": {
        "clear": {"ttc": 0.00, "drac": 0.00},
        "rain":  {"ttc": 0.10, "drac": -0.15},
        "fog":   {"ttc": 0.20, "drac": -0.25},
    },
    "lighting_ctx": {
        "day":   {"ttc": 0.00, "drac": 0.00},
        "night": {"ttc": 0.10, "drac": -0.10},
    },
    "ego_cls": {
        "PassengerCar":     {"ttc": 0.00, "drac": 0.00},
        "SUV":              {"ttc": 0.03, "drac": -0.03},
        "Van":              {"ttc": 0.05, "drac": -0.05},
        "HeavyVehicle":     {"ttc": 0.12, "drac": -0.12},
        "Motorcycle":       {"ttc": 0.00, "drac": 0.00},
        "EmergencyVehicle": {"ttc": 0.00, "drac": 0.00},
    },
    "other_cls": {
        "PassengerCar":     {"ttc": 0.00, "drac": 0.00},
        "SUV":              {"ttc": 0.03, "drac": -0.03},
        "Van":              {"ttc": 0.05, "drac": -0.05},
        "HeavyVehicle":     {"ttc": 0.12, "drac": -0.12},
        "Motorcycle":       {"ttc": 0.00, "drac": 0.00},
        "EmergencyVehicle": {"ttc": 0.00, "drac": 0.00},
    },
}

# Global clipping to keep deterministic adjustments bounded
TTC_DELTA_MIN, TTC_DELTA_MAX = 0.00, 0.40
DRAC_DELTA_MIN, DRAC_DELTA_MAX = -0.50, 0.00


# =========================
# HELPERS
# =========================
def safe_div(a, b):
    return a / b if b else 0.0


def map_vehicle(v: str) -> str:
    v = str(v).lower()
    if any(k in v for k in ["truck", "bus", "carlacola"]):
        return "HeavyVehicle"
    if any(k in v for k in ["bike", "motorcycle", "yamaha", "harley"]):
        return "Motorcycle"
    if any(k in v for k in ["ambulance", "firetruck", "police"]):
        return "EmergencyVehicle"
    if any(k in v for k in ["van", "sprinter"]):
        return "Van"
    if any(k in v for k in ["suv", "jeep"]):
        return "SUV"
    return "PassengerCar"


def build_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = df["weather"].astype(str).str.lower().str.strip()
    df["lighting_ctx"] = np.where(w.eq("night"), "night", "day")
    df["weather_ctx"] = w.replace({"night": "clear"})
    df["ego_cls"] = df["ego_type"].apply(map_vehicle)
    df["other_cls"] = df["other_type"].apply(map_vehicle)
    return df


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


def get_delta(table: dict, key: str):
    return table.get(key, {"ttc": 0.0, "drac": 0.0})


def compute_context_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unique_contexts = df[["weather_ctx", "lighting_ctx", "ego_cls", "other_cls"]].drop_duplicates()

    for r in unique_contexts.itertuples(index=False):
        weather_ctx, lighting_ctx, ego_cls, other_cls = r

        parts = [
            get_delta(MODIFIERS["weather_ctx"], weather_ctx),
            get_delta(MODIFIERS["lighting_ctx"], lighting_ctx),
            get_delta(MODIFIERS["ego_cls"], ego_cls),
            get_delta(MODIFIERS["other_cls"], other_cls),
        ]

        ttc_delta = sum(p["ttc"] for p in parts)
        drac_delta = sum(p["drac"] for p in parts)

        ttc_delta = float(np.clip(ttc_delta, TTC_DELTA_MIN, TTC_DELTA_MAX))
        drac_delta = float(np.clip(drac_delta, DRAC_DELTA_MIN, DRAC_DELTA_MAX))

        rows.append({
            "weather_ctx": weather_ctx,
            "lighting_ctx": lighting_ctx,
            "ego_cls": ego_cls,
            "other_cls": other_cls,
            "ttc_delta": ttc_delta,
            "drac_delta": drac_delta,
            "ttc_thr": BASE_TTC_THR + ttc_delta,
            "drac_thr": BASE_DRAC_THR + drac_delta,
        })

    return pd.DataFrame(rows)


def predict_contextual_fixed_gate(df: pd.DataFrame, thr_df: pd.DataFrame) -> np.ndarray:
    active = gate_active(df["ego_tti_s"].to_numpy(), df["other_tti_s"].to_numpy())
    pred = np.zeros(len(df), dtype=np.int8)

    thr_map = {
        (r.weather_ctx, r.lighting_ctx, r.ego_cls, r.other_cls): (r.ttc_thr, r.drac_thr)
        for r in thr_df.itertuples(index=False)
    }

    ttc = df["TTC_conflict"].to_numpy()
    drac = df["DRAC"].to_numpy()

    for i, r in enumerate(df[["weather_ctx", "lighting_ctx", "ego_cls", "other_cls"]].itertuples(index=False)):
        if not active[i]:
            continue
        ttc_thr, drac_thr = thr_map[(r.weather_ctx, r.lighting_ctx, r.ego_cls, r.other_cls)]
        pred[i] = int((ttc[i] <= ttc_thr) or (drac[i] >= drac_thr))

    return pred


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
    df = build_context(df)
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

    # Same grouped split strategy as Exp 1
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, tmp_idx = next(gss1.split(df, groups=df["episode_id"]))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp["episode_id"]))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()

    thresholds = compute_context_thresholds(df)
    thresholds.to_csv(OUTPUT_DIR / "deterministic_context_thresholds.csv", index=False)

    pred_train = predict_contextual_fixed_gate(train, thresholds)
    pred_val = predict_contextual_fixed_gate(val, thresholds)
    pred_test = predict_contextual_fixed_gate(test, thresholds)

    train_summary = evaluate(train, pred_train, "exp2_deterministic_context_fixed_gate")
    val_summary = evaluate(val, pred_val, "exp2_deterministic_context_fixed_gate")
    test_summary = evaluate(test, pred_test, "exp2_deterministic_context_fixed_gate")

    summary = pd.DataFrame([train_summary, val_summary, test_summary], index=["train", "val", "test"]).reset_index()
    summary = summary.rename(columns={"index": "split"})
    summary.to_csv(OUTPUT_DIR / "exp2_split_summary.csv", index=False)

    test_out = test.copy()
    test_out["pred"] = pred_test
    test_out.to_csv(OUTPUT_DIR / "exp2_test_predictions.csv", index=False)

    with open(OUTPUT_DIR / "deterministic_context_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_csv": INPUT_CSV,
            "label_col": LABEL_COL,
            "random_state": RANDOM_STATE,
            "gate_max_tti": GATE_MAX_TTI,
            "gate_sync_tti": GATE_SYNC_TTI,
            "base_ttc_thr": BASE_TTC_THR,
            "base_drac_thr": BASE_DRAC_THR,
            "modifiers": MODIFIERS,
            "ttc_delta_clip": [TTC_DELTA_MIN, TTC_DELTA_MAX],
            "drac_delta_clip": [DRAC_DELTA_MIN, DRAC_DELTA_MAX],
        }, f, indent=2)

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")
    print(f"Fixed gate: max_tti <= {GATE_MAX_TTI}, |ego_tti-other_tti| <= {GATE_SYNC_TTI}")
    print(f"Base thresholds: TTC <= {BASE_TTC_THR:.3f}, DRAC >= {BASE_DRAC_THR:.3f}")

    print("\n=== Test thresholds preview ===")
    print(thresholds.head(10).to_string(index=False))

    print("\n=== Split summary ===")
    print(summary.to_string(index=False))

    print(f"\nSaved files in: {OUTPUT_DIR.resolve()}")
    print("- deterministic_context_config.json")
    print("- deterministic_context_thresholds.csv")
    print("- exp2_split_summary.csv")
    print("- exp2_test_predictions.csv")


if __name__ == "__main__":
    main()
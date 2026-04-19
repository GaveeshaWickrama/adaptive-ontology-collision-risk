from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import GroupShuffleSplit

#INPUT_CSV = "type22_outcome_ground_truth.csv"
INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTDIR = Path("outputs/metrics/exp5_binary_results")
RANDOM_STATE = 42
BETA = 2.0
N_GRID = 20
MIN_ROWS_PER_CONTEXT = 150
MIN_POS_PER_CONTEXT = 15
LAMBDA_CANDIDATES = [50, 150, 300, 600, 1000]
TEST_SIZE = 0.15
VAL_SIZE_WITHIN_TRAINVAL = 0.17647058823529413
DOWNSAMPLE_EVERY = 1


# def map_vehicle(v: str) -> str:
#     v = str(v).lower()
#     if any(k in v for k in ["ambulance", "firetruck", "police"]):
#         return "EmergencyVehicle"
#     if any(k in v for k in ["truck", "bus"]):
#         return "HeavyVehicle"
#     if any(k in v for k in ["motorcycle", "bike"]):
#         return "Motorcycle"
#     if "van" in v:
#         return "Van"
#     return "PassengerCar"

def map_vehicle(v: str) -> str:
    v = str(v).lower()
    if any(k in v for k in ["ambulance", "firetruck", "police"]):
        return "EmergencyVehicle"
    if any(k in v for k in ["truck", "bus", "carlacola"]):
        return "HeavyVehicle"
    if any(k in v for k in ["motorcycle", "bike", "yamaha", "harley"]):
        return "Motorcycle"
    if any(k in v for k in ["van", "sprinter"]):
        return "Van"
    if any(k in v for k in ["suv", "jeep", "patrol"]):
        return "SUV"
    return "PassengerCar"



def build_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    if "episode_id" in df.columns:
        df["episode_id"] = df["episode_id"].astype(str)
        return df
    if "scenario_id" not in df.columns:
        raise ValueError("Need either 'episode_id' or 'scenario_id'.")
    df = df.sort_values(["scenario_id", "time"]).copy()
    time_reset = df.groupby("scenario_id")["time"].diff().lt(0).fillna(False)
    df["episode_idx_within_scenario"] = (
        time_reset.groupby(df["scenario_id"]).cumsum().astype(int)
    )
    df["episode_id"] = (
        df["scenario_id"].astype(str)
        + "__ep"
        + df["episode_idx_within_scenario"].astype(str)
    )
    return df



def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "post_collision" in df.columns:
        df = df.loc[~df["post_collision"]].copy()

    required = ["gt_binary", "TTC_conflict", "DRAC", "weather", "ego_type", "other_type", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[
        df["gt_binary"].notna()
        & df["TTC_conflict"].notna()
        & df["DRAC"].notna()
        & (df["TTC_conflict"] > 0)
    ].copy()

    df["y"] = df["gt_binary"].astype(str).str.lower().map({"low": 0, "high": 1})
    df = df.loc[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)

    df = build_episode_id(df)

    if DOWNSAMPLE_EVERY > 1:
        df = (
            df.sort_values(["episode_id", "time"])
              .groupby("episode_id", group_keys=False)
              .apply(lambda g: g.iloc[::DOWNSAMPLE_EVERY])
              .reset_index(drop=True)
        )

    if "lighting" in df.columns:
        df["weather_ctx"] = df["weather"].astype(str).str.lower().str.strip()
        df["lighting_ctx"] = df["lighting"].astype(str).str.lower().str.strip()
    else:
        w = df["weather"].astype(str).str.lower().str.strip()
        df["lighting_ctx"] = np.where(w.eq("night"), "night", "day")
        df["weather_ctx"] = np.where(w.eq("night"), "clear", w)

    df["ego_cls"] = df["ego_type"].apply(map_vehicle)
    df["other_cls"] = df["other_type"].apply(map_vehicle)
    df["context"] = (
        df["weather_ctx"] + "|" + df["lighting_ctx"] + "|" + df["ego_cls"] + "|" + df["other_cls"]
    )

    df["ttc_used"] = df["TTC_conflict"].clip(lower=0)
    df["drac_used"] = df["DRAC"].clip(lower=0)

    return df.reset_index(drop=True)



def make_splits(df: pd.DataFrame):
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss_outer.split(df, groups=df["episode_id"]))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    gss_inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE_WITHIN_TRAINVAL,
        random_state=RANDOM_STATE,
    )
    train_idx_rel, val_idx_rel = next(gss_inner.split(train_val, groups=train_val["episode_id"]))
    train = train_val.iloc[train_idx_rel].copy()
    val = train_val.iloc[val_idx_rel].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True), train_val.reset_index(drop=True), test.reset_index(drop=True)



def search_threshold_pair(df_sub: pd.DataFrame, beta: float = BETA, n_grid: int = N_GRID):
    y = df_sub["y"].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return None

    ttc = df_sub["ttc_used"].to_numpy()
    drac = df_sub["drac_used"].to_numpy()

    ttc_grid = np.unique(np.quantile(ttc, np.linspace(0.02, 0.98, n_grid)))
    drac_grid = np.unique(np.quantile(drac, np.linspace(0.02, 0.98, n_grid)))

    best = {"score": -1.0, "ttc_thr": float(np.median(ttc)), "drac_thr": float(np.median(drac))}

    for t in ttc_grid:
        ttc_mask = ttc <= t
        for d in drac_grid:
            pred = (ttc_mask | (drac >= d)).astype(int)
            score = fbeta_score(y, pred, beta=beta, zero_division=0)
            if score > best["score"]:
                best = {"score": float(score), "ttc_thr": float(t), "drac_thr": float(d)}

    return best



def fit_context_thresholds(train_df: pd.DataFrame, global_thr: dict, lam: float) -> pd.DataFrame:
    rows = []
    for ctx, g in train_df.groupby("context"):
        positives = int(g["y"].sum())
        if len(g) < MIN_ROWS_PER_CONTEXT or positives < MIN_POS_PER_CONTEXT or positives == len(g):
            continue
        raw = search_threshold_pair(g)
        if raw is None:
            continue
        w = len(g) / (len(g) + lam)
        rows.append(
            {
                "context": ctx,
                "n": len(g),
                "positives": positives,
                "positive_rate": positives / len(g),
                "ttc_thr": w * raw["ttc_thr"] + (1 - w) * global_thr["ttc_thr"],
                "drac_thr": w * raw["drac_thr"] + (1 - w) * global_thr["drac_thr"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["n", "positives", "positive_rate", "ttc_thr", "drac_thr"])
    return pd.DataFrame(rows).set_index("context").sort_index()



def predict_binary(df_sub: pd.DataFrame, global_thr: dict, ctx_thr: pd.DataFrame | None = None) -> pd.Series:
    work = df_sub[["context", "ttc_used", "drac_used"]].copy()
    if ctx_thr is not None and not ctx_thr.empty:
        work = work.join(ctx_thr[["ttc_thr", "drac_thr"]], on="context")
    else:
        work["ttc_thr"] = np.nan
        work["drac_thr"] = np.nan

    work["ttc_thr"] = work["ttc_thr"].fillna(global_thr["ttc_thr"])
    work["drac_thr"] = work["drac_thr"].fillna(global_thr["drac_thr"])

    pred = ((work["ttc_used"] <= work["ttc_thr"]) | (work["drac_used"] >= work["drac_thr"])).astype(int)
    return pred



def frame_metrics(y_true, y_pred, prefix: str = "") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        f"{prefix}tn": int(tn),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tp": int(tp),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f2": float(fbeta_score(y_true, y_pred, beta=BETA, zero_division=0)),
        f"{prefix}specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    return out



def episode_metrics(df_sub: pd.DataFrame, pred_col: str, prefix: str = "") -> dict:
    episode_df = df_sub.groupby("episode_id").agg(
        y_ep=("y", "max"),
        pred_ep=(pred_col, "max"),
    )

    coll_mask = episode_df["y_ep"] == 1
    noncoll_mask = episode_df["y_ep"] == 0

    out = {
        f"{prefix}episode_warning_recall": float(episode_df.loc[coll_mask, "pred_ep"].mean()) if coll_mask.any() else np.nan,
        f"{prefix}episode_false_alarm_rate": float(episode_df.loc[noncoll_mask, "pred_ep"].mean()) if noncoll_mask.any() else np.nan,
    }

    if "dt_to_collision" in df_sub.columns:
        hits = df_sub.loc[(df_sub["y"] == 1) & (df_sub[pred_col] == 1) & df_sub["dt_to_collision"].notna()].copy()
        if len(hits) > 0:
            lead = hits.groupby("episode_id")["dt_to_collision"].max()
            out[f"{prefix}mean_correct_warning_lead_s"] = float(lead.mean())
            out[f"{prefix}median_correct_warning_lead_s"] = float(lead.median())
        else:
            out[f"{prefix}mean_correct_warning_lead_s"] = np.nan
            out[f"{prefix}median_correct_warning_lead_s"] = np.nan
    return out



def per_context_metrics(df_sub: pd.DataFrame, pred_col: str, model_name: str) -> pd.DataFrame:
    rows = []
    for ctx, g in df_sub.groupby("context"):
        rows.append(
            {
                "model": model_name,
                "context": ctx,
                "n": len(g),
                "positives": int(g["y"].sum()),
                "positive_rate": float(g["y"].mean()),
                "precision": float(precision_score(g["y"], g[pred_col], zero_division=0)),
                "recall": float(recall_score(g["y"], g[pred_col], zero_division=0)),
                "f2": float(fbeta_score(g["y"], g[pred_col], beta=BETA, zero_division=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "context"]).reset_index(drop=True)

def compute_adaptive_score(df_sub: pd.DataFrame, global_thr: dict, ctx_thr: pd.DataFrame | None = None) -> pd.Series:
    work = df_sub[["context", "ttc_used", "drac_used", "ego_tti_s", "other_tti_s"]].copy()

    if ctx_thr is not None and not ctx_thr.empty:
        work = work.join(ctx_thr[["ttc_thr", "drac_thr"]], on="context")
    else:
        work["ttc_thr"] = np.nan
        work["drac_thr"] = np.nan

    work["ttc_thr"] = work["ttc_thr"].fillna(global_thr["ttc_thr"])
    work["drac_thr"] = work["drac_thr"].fillna(global_thr["drac_thr"])

    # Gate score: 1 when strongly inside gate, 0 when outside
    max_tti = np.maximum(work["ego_tti_s"], work["other_tti_s"])
    sync_gap = np.abs(work["ego_tti_s"] - work["other_tti_s"])

    gate_time_score = np.clip((3.5 - max_tti) / 3.5, 0.0, 1.0)
    gate_sync_score = np.clip((0.2 - sync_gap) / 0.2, 0.0, 1.0)
    gate_score = np.minimum(gate_time_score, gate_sync_score)

    # TTC risk component: higher when TTC is below threshold
    ttc_score = np.clip((work["ttc_thr"] - work["ttc_used"]) / work["ttc_thr"], 0.0, 1.0)

    # DRAC risk component: higher when DRAC is above threshold
    drac_score = np.clip((work["drac_used"] - work["drac_thr"]) / work["drac_thr"], 0.0, 1.0)

    # Final score aligned with OR rule inside gate
    risk_score = gate_score * np.maximum(ttc_score, drac_score)

    return pd.Series(risk_score, index=df_sub.index, name="adaptive_score")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = prepare_dataframe(df)

    train, val, train_val, test = make_splits(df)

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")
    print(f"Positive rate train: {train['y'].mean():.4f} | val: {val['y'].mean():.4f} | test: {test['y'].mean():.4f}")

    global_train = search_threshold_pair(train)
    if global_train is None:
        raise RuntimeError("Global threshold search failed. Check label diversity in training split.")

    baseline_val_pred = predict_binary(val, global_train)
    baseline_val_metrics = frame_metrics(val["y"], baseline_val_pred, prefix="val_")

    tuning_rows = []
    best_lambda = None
    best_val_f2 = -1.0

    for lam in LAMBDA_CANDIDATES:
        ctx_thr = fit_context_thresholds(train, global_train, lam)
        val_pred = predict_binary(val, global_train, ctx_thr)
        m = frame_metrics(val["y"], val_pred)
        tuning_rows.append({
            "lambda": lam,
            "contexts_learned": int(len(ctx_thr)),
            "val_precision": m["precision"],
            "val_recall": m["recall"],
            "val_f2": m["f2"],
        })
        if m["f2"] > best_val_f2:
            best_val_f2 = m["f2"]
            best_lambda = lam

    tuning_df = pd.DataFrame(tuning_rows).sort_values("lambda")
    tuning_df.to_csv(OUTDIR / "lambda_tuning.csv", index=False)

    print("\nValidation results for lambda:")
    print(tuning_df)
    print(f"\nChosen lambda: {best_lambda}")
    print(f"Baseline val F2: {baseline_val_metrics['val_f2']:.4f}")
    print(f"Adaptive best val F2: {best_val_f2:.4f}")

    global_final = search_threshold_pair(train_val)
    ctx_final = fit_context_thresholds(train_val, global_final, best_lambda)

    with open(OUTDIR / "exp5_global_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(global_final, f, indent=2)

    ctx_final.reset_index().to_csv(OUTDIR / "exp5_binary_context_thresholds.csv", index=False)

    test = test.copy()
    test["pred_static"] = predict_binary(test, global_final)
    test["pred_adaptive"] = predict_binary(test, global_final, ctx_final)
    test["adaptive_score"] = compute_adaptive_score(test, global_final, ctx_final)
    test.to_csv(OUTDIR / "binary_test_predictions.csv", index=False)

    summary_rows = []

    static_summary = {"model": "static_global"}
    static_summary.update(frame_metrics(test["y"], test["pred_static"]))
    static_summary.update(episode_metrics(test, "pred_static"))
    summary_rows.append(static_summary)

    adaptive_summary = {"model": "adaptive_context"}
    adaptive_summary.update(frame_metrics(test["y"], test["pred_adaptive"]))
    adaptive_summary.update(episode_metrics(test, "pred_adaptive"))
    summary_rows.append(adaptive_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "binary_eval_summary.csv", index=False)

    context_metrics = pd.concat(
        [
            per_context_metrics(test, "pred_static", "static_global"),
            per_context_metrics(test, "pred_adaptive", "adaptive_context"),
        ],
        axis=0,
        ignore_index=True,
    )
    context_metrics.to_csv(OUTDIR / "binary_context_metrics.csv", index=False)

    print("\n=== Test summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved files in: {OUTDIR.resolve()}")
    print("- exp5_global_thresholds.json")
    print("- exp5_binary_context_thresholds.csv")
    print("- lambda_tuning.csv")
    print("- binary_eval_summary.csv")
    print("- binary_context_metrics.csv")
    print("- binary_test_predictions.csv")


if __name__ == "__main__":
    main()

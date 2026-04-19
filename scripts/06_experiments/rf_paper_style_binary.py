#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


# ==========================================
# CONFIG
# ==========================================
INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTDIR = Path("outputs/metrics/rf_paper_style_binary")

RANDOM_STATE = 42
OBS_WINDOW_S = 0.5
PRED_WINDOW_S = 0.7

# paper-inspired RF defaults
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1
RF_CRITERION = "gini"

USE_SMOTE = True   # train only

# if your simulator is 20 Hz, dt = 0.05
# if unsure, infer from median time diff
DEFAULT_DT = 0.05


# ==========================================
# HELPERS
# ==========================================
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def normalize_binary_label(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    pos = {"1", "true", "high", "highrisk", "collisionlikely", "risk", "yes"}
    return x.isin(pos).astype(int)


def ensure_scenario_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scenario_id" not in out.columns:
        if "ego_id" in out.columns and "other_id" in out.columns:
            out["scenario_id"] = out["ego_id"].astype(str) + "_" + out["other_id"].astype(str)
        else:
            raise ValueError("Need scenario_id, or both ego_id and other_id.")
    return out


def infer_dt(df: pd.DataFrame) -> float:
    if "time" not in df.columns:
        return DEFAULT_DT
    diffs = (
        df.sort_values(["scenario_id", "time"])
          .groupby("scenario_id")["time"]
          .diff()
          .dropna()
    )
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) == 0:
        return DEFAULT_DT
    return float(np.median(diffs))


def map_weather_lighting(raw_weather: str):
    w = str(raw_weather).strip().lower()
    lighting = "night" if w == "night" else "day"
    weather = "clear" if w == "night" else w
    return weather, lighting


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


def slope_of_regression(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # speeds
    if "ego_speed_mps" in out.columns and "ego_speed" not in out.columns:
        out["ego_speed"] = out["ego_speed_mps"]
    if "other_speed_mps" in out.columns and "other_speed" not in out.columns:
        out["other_speed"] = out["other_speed_mps"]

    # headway-style distance
    if "ego_dist_to_conflict_m" in out.columns:
        out["headway_distance"] = out["ego_dist_to_conflict_m"]
    elif "dist_between_m" in out.columns:
        out["headway_distance"] = out["dist_between_m"]
    elif "dist_between" in out.columns:
        out["headway_distance"] = out["dist_between"]
    else:
        out["headway_distance"] = np.nan

    # speed difference
    if "rel_speed_towards_mps" in out.columns:
        out["speed_difference"] = out["rel_speed_towards_mps"].abs()
    elif {"ego_speed", "other_speed"}.issubset(out.columns):
        out["speed_difference"] = (out["ego_speed"] - out["other_speed"]).abs()
    else:
        out["speed_difference"] = np.nan

    # acceleration from ego speed derivative within scenario
    if "ego_speed" in out.columns and "time" in out.columns:
        out = out.sort_values(["scenario_id", "time"]).copy()
        dt = out.groupby("scenario_id")["time"].diff()
        ds = out.groupby("scenario_id")["ego_speed"].diff()
        out["acceleration"] = ds / dt
        out["acceleration"] = out["acceleration"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["acceleration"] = 0.0

    # jerk from acceleration derivative
    if "time" in out.columns:
        dt2 = out.groupby("scenario_id")["time"].diff()
        da = out.groupby("scenario_id")["acceleration"].diff()
        out["jerk"] = da / dt2
        out["jerk"] = out["jerk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["jerk"] = 0.0

    # context
    weather_ctx = []
    lighting_ctx = []
    for w in out["weather"].astype(str):
        wc, lc = map_weather_lighting(w)
        weather_ctx.append(wc)
        lighting_ctx.append(lc)
    out["weather_ctx"] = weather_ctx
    out["lighting_ctx"] = lighting_ctx
    out["ego_cls"] = out["ego_type"].apply(map_vehicle)
    out["other_cls"] = out["other_type"].apply(map_vehicle)

    return out


def build_window_dataset(df: pd.DataFrame, obs_window_s: float, pred_window_s: float) -> pd.DataFrame:
    rows = []
    dt = infer_dt(df)
    obs_n = max(2, int(round(obs_window_s / dt)))
    pred_n = max(1, int(round(pred_window_s / dt)))

    df = df.sort_values(["scenario_id", "time"]).copy()

    for sid, g in df.groupby("scenario_id", sort=False):
        g = g.reset_index(drop=True)

        y = normalize_binary_label(g["gt_binary"])
        g = g.copy()
        g["y_true"] = y

        for end_obs in range(obs_n - 1, len(g) - pred_n):
            obs = g.iloc[end_obs - obs_n + 1 : end_obs + 1]
            fut = g.iloc[end_obs + 1 : end_obs + 1 + pred_n]

            row = {
                "scenario_id": sid,
                "time_anchor": float(obs["time"].iloc[-1]),
                "gt_future_binary": int(fut["y_true"].max()),

                # risk/history variables analogous to paper's RM/RL/RT
                "rm": float(obs["R_outcome"].mean()) if "R_outcome" in obs.columns else float(obs["y_true"].mean()),
                "rl": int(obs["y_true"].iloc[-1]),
                "rt": float(obs["y_true"].iloc[-1] - obs["y_true"].iloc[0]),

                # behavioral variables: mean/std/slope
                "speed_mean": float(obs["ego_speed"].mean()) if "ego_speed" in obs.columns else np.nan,
                "speed_std": float(obs["ego_speed"].std(ddof=0)) if "ego_speed" in obs.columns else np.nan,
                "speed_srf": slope_of_regression(obs["ego_speed"].to_numpy()) if "ego_speed" in obs.columns else np.nan,

                "acc_mean": float(obs["acceleration"].mean()),
                "acc_std": float(obs["acceleration"].std(ddof=0)),
                "acc_srf": slope_of_regression(obs["acceleration"].to_numpy()),

                "jerk_mean": float(obs["jerk"].mean()),
                "jerk_std": float(obs["jerk"].std(ddof=0)),
                "jerk_srf": slope_of_regression(obs["jerk"].to_numpy()),

                "headway_mean": float(obs["headway_distance"].mean()),
                "headway_std": float(obs["headway_distance"].std(ddof=0)),
                "headway_srf": slope_of_regression(obs["headway_distance"].to_numpy()),

                "speeddiff_mean": float(obs["speed_difference"].mean()),
                "speeddiff_std": float(obs["speed_difference"].std(ddof=0)),
                "speeddiff_srf": slope_of_regression(obs["speed_difference"].to_numpy()),

                # contextual variables
                "weather_ctx": str(obs["weather_ctx"].iloc[-1]),
                "lighting_ctx": str(obs["lighting_ctx"].iloc[-1]),
                "ego_cls": str(obs["ego_cls"].iloc[-1]),
                "other_cls": str(obs["other_cls"].iloc[-1]),
            }

            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def split_grouped(df: pd.DataFrame, group_col: str, seed: int = 42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, tmp_idx = next(gss1.split(df, groups=df[group_col]))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp[group_col]))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()
    return train, val, test


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = safe_div(tn, tn + fp)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "specificity": float(specificity),
    }


def episode_metrics(df: pd.DataFrame, pred_col: str, gt_col: str = "y_true", group_col: str = "scenario_id"):
    g = df.groupby(group_col, sort=False)
    ep_gt = g[gt_col].max().astype(int)
    ep_pred = g[pred_col].max().astype(int)

    pos_mask = ep_gt == 1
    neg_mask = ep_gt == 0

    ewr = safe_div(int(((ep_pred == 1) & pos_mask).sum()), int(pos_mask.sum()))
    efar = safe_div(int(((ep_pred == 1) & neg_mask).sum()), int(neg_mask.sum()))
    return ewr, efar


def build_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        criterion=RF_CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("rf", model)])


def maybe_smote(X_train, y_train):
    if not USE_SMOTE:
        return X_train, y_train
    if SMOTE is None:
        print("SMOTE not available. Install imbalanced-learn or set USE_SMOTE=False.")
        return X_train, y_train
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = ensure_scenario_id(df)
    df = add_basic_features(df)

    keep = (
        df["gt_binary"].notna()
        & df["TTC_conflict"].notna()
        & df["DRAC"].notna()
        & (df["TTC_conflict"] > 0)
    )
    if "post_collision" in df.columns:
        keep &= ~df["post_collision"].fillna(False)

    df = df.loc[keep].copy()

    win_df = build_window_dataset(df, OBS_WINDOW_S, PRED_WINDOW_S)

    numeric_cols = [
        "rm", "rl", "rt",
        "speed_mean", "speed_std", "speed_srf",
        "acc_mean", "acc_std", "acc_srf",
        "jerk_mean", "jerk_std", "jerk_srf",
        "headway_mean", "headway_std", "headway_srf",
        "speeddiff_mean", "speeddiff_std", "speeddiff_srf",
    ]
    categorical_cols = ["weather_ctx", "lighting_ctx", "ego_cls", "other_cls"]

    train_df, val_df, test_df = split_grouped(win_df, "scenario_id", seed=RANDOM_STATE)

    print(f"Total window samples: {len(win_df):,}")
    print(f"Total episodes: {win_df['scenario_id'].nunique():,}")
    print(f"Train windows: {len(train_df):,} | Val windows: {len(val_df):,} | Test windows: {len(test_df):,}")
    print(
        f"Positive rate train: {train_df['gt_future_binary'].mean():.4f} | "
        f"val: {val_df['gt_future_binary'].mean():.4f} | "
        f"test: {test_df['gt_future_binary'].mean():.4f}"
    )

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df["gt_future_binary"].astype(int).values
    X_val = val_df[numeric_cols + categorical_cols]
    y_val = val_df["gt_future_binary"].astype(int).values
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df["gt_future_binary"].astype(int).values

    pipe = build_pipeline(numeric_cols, categorical_cols)

    # preprocess separately for SMOTE
    pre = pipe.named_steps["pre"]
    X_train_pre = pre.fit_transform(X_train)
    X_val_pre = pre.transform(X_val)
    X_test_pre = pre.transform(X_test)

    X_train_res, y_train_res = maybe_smote(X_train_pre, y_train)

    rf = pipe.named_steps["rf"]
    rf.fit(X_train_res, y_train_res)

    rows = []
    for split_name, split_df, X_split_pre, y_split in [
        ("train", train_df, X_train_pre, y_train),
        ("val", val_df, X_val_pre, y_val),
        ("test", test_df, X_test_pre, y_test),
    ]:
        pred = rf.predict(X_split_pre)

        m = frame_metrics(y_split, pred)
        tmp = split_df.copy()
        tmp["y_true"] = y_split
        tmp["pred"] = pred
        ewr, efar = episode_metrics(tmp, pred_col="pred", gt_col="y_true", group_col="scenario_id")

        rows.append({
            "split": split_name,
            "model": "rf_paper_style_binary",
            "obs_window_s": OBS_WINDOW_S,
            "pred_window_s": PRED_WINDOW_S,
            **m,
            "episode_warning_recall": ewr,
            "episode_false_alarm_rate": efar,
        })

        if split_name == "test":
            tmp.to_csv(OUTDIR / "rf_paper_style_test_predictions.csv", index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTDIR / "rf_paper_style_summary.csv", index=False)

    feature_names = (
        list(pre.get_feature_names_out())
        if hasattr(pre, "get_feature_names_out")
        else [f"f{i}" for i in range(X_train_pre.shape[1])]
    )
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(OUTDIR / "rf_paper_style_feature_importance.csv", index=False)

    with open(OUTDIR / "rf_paper_style_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": INPUT_CSV,
                "obs_window_s": OBS_WINDOW_S,
                "pred_window_s": PRED_WINDOW_S,
                "rf_n_estimators": RF_N_ESTIMATORS,
                "rf_max_depth": RF_MAX_DEPTH,
                "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "rf_criterion": RF_CRITERION,
                "use_smote": USE_SMOTE,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            },
            f,
            indent=2,
        )

    print("\n=== Final summary ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved files in: {OUTDIR.resolve()}")
    print("- rf_paper_style_summary.csv")
    print("- rf_paper_style_test_predictions.csv")
    print("- rf_paper_style_feature_importance.csv")
    print("- rf_paper_style_config.json")


if __name__ == "__main__":
    main()
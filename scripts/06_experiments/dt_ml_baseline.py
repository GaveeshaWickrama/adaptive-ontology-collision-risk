#!/usr/bin/env python3
"""
Decision-tree ML baseline for binary frame-level risk prediction.

Paper inspiration:
- Hu et al. (2017): Decision tree-based maneuver prediction
- Uses CART-style tree with Gini splitting and class balancing via undersampling

This script:
- reads the final labeled dataset
- builds the same grouped train/val/test split style used in the thesis
- trains a paper-inspired decision tree baseline
- tunes hyperparameters on validation using F2
- evaluates on train / val / test
- saves predictions, summary, feature importances, and readable tree rules
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


# =========================
# CONFIG
# =========================
INPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"
OUTDIR = Path("outputs/metrics/ml_decision_tree_baseline")
RANDOM_STATE = 42

# Keep this True to mimic the paper's imbalance handling idea
USE_TRAIN_UNDERSAMPLING = True

# Same grouped split style as Exp 1-4
OUTER_TEST_SIZE = 0.30
INNER_TEST_SIZE = 0.50  # splits held-out 30% into val/test equally

LABEL_COL = "gt_binary"

NUMERIC_FEATURES = [
    "TTC_conflict",
    "DRAC",
    "ego_tti_s",
    "other_tti_s",
]

CATEGORICAL_FEATURES = [
    "weather_ctx",
    "lighting_ctx",
    "ego_cls",
    "other_cls",
]


# =========================
# HELPERS
# =========================
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


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


def build_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = out["weather"].astype(str).str.lower().str.strip()
    out["lighting_ctx"] = np.where(w.eq("night"), "night", "day")
    out["weather_ctx"] = np.where(w.eq("night"), "clear", w)
    out["ego_cls"] = out["ego_type"].apply(map_vehicle)
    out["other_cls"] = out["other_type"].apply(map_vehicle)
    return out


def build_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "episode_id" in out.columns:
        out["episode_id"] = out["episode_id"].astype(str)
        return out

    if "scenario_id" in out.columns:
        out["episode_id"] = out["scenario_id"].astype(str)
        return out

    if {"ego_id", "other_id"}.issubset(out.columns):
        out["episode_id"] = out["ego_id"].astype(str) + "_" + out["other_id"].astype(str)
        return out

    raise ValueError("Need one of: episode_id, scenario_id, or ego_id/other_id")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "post_collision" in df.columns:
        df = df.loc[~df["post_collision"].fillna(False)].copy()

    required = set(NUMERIC_FEATURES + ["weather", "ego_type", "other_type", LABEL_COL])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = (
        df[LABEL_COL].notna()
        & df["TTC_conflict"].notna()
        & df["DRAC"].notna()
        & df["ego_tti_s"].notna()
        & df["other_tti_s"].notna()
        & (df["TTC_conflict"] > 0)
        & (df["ego_tti_s"] >= 0)
        & (df["other_tti_s"] >= 0)
    )
    df = df.loc[keep].copy()

    df["y"] = df[LABEL_COL].astype(str).str.lower().map({"low": 0, "high": 1})
    df = df.loc[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)

    df = build_episode_id(df)
    df = build_context(df)

    return df.reset_index(drop=True)


def grouped_split(df: pd.DataFrame):
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, tmp_idx = next(gss1.split(df, groups=df["episode_id"]))
    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, groups=tmp["episode_id"]))
    val = tmp.iloc[val_rel_idx].copy()
    test = tmp.iloc[test_rel_idx].copy()

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def undersample_train(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df.copy()

    n = min(len(pos), len(neg))
    pos_s = pos.sample(n=n, random_state=RANDOM_STATE)
    neg_s = neg.sample(n=n, random_state=RANDOM_STATE)

    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    return out.reset_index(drop=True)


def make_pipeline(max_depth, min_samples_leaf, ccp_alpha) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocess", pre),
        ("clf", clf),
    ])


def frame_metrics(y_true, y_pred) -> Dict[str, float]:
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


def episode_metrics(df_eval: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    g = df_eval.groupby("episode_id")
    true_ep = g["y"].max()
    pred_ep = g[pred_col].max()

    pos_mask = true_ep == 1
    neg_mask = true_ep == 0

    episode_warning_recall = float(pred_ep[pos_mask].mean()) if pos_mask.any() else np.nan
    episode_false_alarm_rate = float(pred_ep[neg_mask].mean()) if neg_mask.any() else np.nan

    return {
        "episode_warning_recall": episode_warning_recall,
        "episode_false_alarm_rate": episode_false_alarm_rate,
    }


def summarize(df_eval: pd.DataFrame, pred_col: str, model_name: str) -> Dict[str, float]:
    out = {"model": model_name}
    out.update(frame_metrics(df_eval["y"], df_eval[pred_col]))
    out.update(episode_metrics(df_eval, pred_col))
    return out


def fit_and_predict(model: Pipeline, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["y"]
    X_eval = eval_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    model.fit(X_train, y_train)
    return model.predict(X_eval)


def tune_tree(train_df: pd.DataFrame, val_df: pd.DataFrame):
    if USE_TRAIN_UNDERSAMPLING:
        fit_df = undersample_train(train_df)
    else:
        fit_df = train_df.copy()

    grid_max_depth = [3, 4, 5, 6, 8, None]
    grid_min_leaf = [1, 5, 10, 20, 50]
    grid_ccp_alpha = [0.0, 0.0005, 0.001, 0.002, 0.005]

    rows = []
    best = None
    best_key = None

    for max_depth in grid_max_depth:
        for min_leaf in grid_min_leaf:
            for ccp_alpha in grid_ccp_alpha:
                model = make_pipeline(max_depth, min_leaf, ccp_alpha)
                pred_val = fit_and_predict(model, fit_df, val_df)

                tmp = val_df.copy()
                tmp["y_pred"] = pred_val
                s = summarize(tmp, "y_pred", "decision_tree")
                s.update({
                    "max_depth": max_depth if max_depth is not None else -1,
                    "min_samples_leaf": min_leaf,
                    "ccp_alpha": ccp_alpha,
                })
                rows.append(s)

                key = (
                    s["f2"],
                    s["recall"],
                    s["specificity"],
                    s["precision"],
                )
                if best is None or key > best_key:
                    best = {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_leaf,
                        "ccp_alpha": ccp_alpha,
                        "val_summary": s,
                    }
                    best_key = key

    return best, pd.DataFrame(rows)


def get_feature_names_from_pipeline(model: Pipeline) -> List[str]:
    pre = model.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out()
    return list(feature_names)


def save_tree_artifacts(model: Pipeline, outdir: Path):
    clf = model.named_steps["clf"]
    feature_names = get_feature_names_from_pipeline(model)

    rules_text = export_text(clf, feature_names=feature_names, decimals=4)
    (outdir / "dt_tree_rules.txt").write_text(rules_text, encoding="utf-8")

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(outdir / "dt_feature_importances.csv", index=False)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = prepare_dataframe(df)

    train, val, test = grouped_split(df)

    print(f"Total frames: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique():,}")
    print(f"Train frames: {len(train):,} | Val frames: {len(val):,} | Test frames: {len(test):,}")
    print(
        "Positive rate train: {:.4f} | val: {:.4f} | test: {:.4f}".format(
            train["y"].mean(), val["y"].mean(), test["y"].mean()
        )
    )

    best, tuning_df = tune_tree(train, val)
    tuning_df.to_csv(OUTDIR / "dt_validation_grid.csv", index=False)

    print("\n=== Best validation configuration ===")
    print({
        "max_depth": best["max_depth"],
        "min_samples_leaf": best["min_samples_leaf"],
        "ccp_alpha": best["ccp_alpha"],
        "val_precision": best["val_summary"]["precision"],
        "val_recall": best["val_summary"]["recall"],
        "val_f2": best["val_summary"]["f2"],
        "val_specificity": best["val_summary"]["specificity"],
        "val_episode_warning_recall": best["val_summary"]["episode_warning_recall"],
        "val_episode_false_alarm_rate": best["val_summary"]["episode_false_alarm_rate"],
    })

    # Final chosen model (fit on TRAIN only, like other split-based experiments)
    fit_df = undersample_train(train) if USE_TRAIN_UNDERSAMPLING else train.copy()
    final_model = make_pipeline(
        best["max_depth"],
        best["min_samples_leaf"],
        best["ccp_alpha"],
    )

    X_fit = fit_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_fit = fit_df["y"]
    final_model.fit(X_fit, y_fit)

    rows = []
    test_predictions = None

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        X = split_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        pred = final_model.predict(X)
        score = final_model.predict_proba(X)[:, 1]

        tmp = split_df.copy()
        tmp["y_pred"] = pred
        tmp["y_score"] = score
        s = summarize(tmp, "y_pred", "decision_tree_ml_baseline")
        s["split"] = split_name
        rows.append(s)

        print(f"\n=== {split_name.upper()} summary ===")
        print(
            f"tn={s['tn']} fp={s['fp']} fn={s['fn']} tp={s['tp']} "
            f"precision={s['precision']:.6f} recall={s['recall']:.6f} "
            f"f2={s['f2']:.6f} specificity={s['specificity']:.6f} "
            f"episode_warning_recall={s['episode_warning_recall']:.6f} "
            f"episode_false_alarm_rate={s['episode_false_alarm_rate']:.6f}"
        )

        if split_name == "test":
            test_predictions = tmp.copy()

    summary_df = pd.DataFrame(rows)[[
        "split", "model",
        "tn", "fp", "fn", "tp",
        "precision", "recall", "f2", "specificity",
        "episode_warning_recall", "episode_false_alarm_rate"
    ]]
    summary_df.to_csv(OUTDIR / "dt_split_summary.csv", index=False)

    with open(OUTDIR / "dt_best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": INPUT_CSV,
                "random_state": RANDOM_STATE,
                "use_train_undersampling": USE_TRAIN_UNDERSAMPLING,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "best_max_depth": best["max_depth"],
                "best_min_samples_leaf": best["min_samples_leaf"],
                "best_ccp_alpha": best["ccp_alpha"],
                "best_validation_summary": best["val_summary"],
            },
            f,
            indent=2,
        )

    if test_predictions is not None:
        keep_cols = [
            "frame", "time", "episode_id", "scenario_id",
            "weather", "ego_type", "other_type",
            "TTC_conflict", "DRAC", "ego_tti_s", "other_tti_s",
            "gt_binary", "y", "y_pred", "y_score"
        ]
        keep_cols = [c for c in keep_cols if c in test_predictions.columns]
        test_predictions[keep_cols].to_csv(OUTDIR / "dt_test_predictions.csv", index=False)

    save_tree_artifacts(final_model, OUTDIR)

    print(f"\nSaved files in: {OUTDIR.resolve()}")
    print("- dt_best_config.json")
    print("- dt_validation_grid.csv")
    print("- dt_split_summary.csv")
    print("- dt_test_predictions.csv")
    print("- dt_feature_importances.csv")
    print("- dt_tree_rules.txt")


if __name__ == "__main__":
    main()
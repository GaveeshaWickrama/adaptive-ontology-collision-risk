#!/usr/bin/env python3
"""
Create a NEW derived dataset (does NOT modify the original CSV) and run OLS regressions
for TTC, PET, DRAC using scikit-learn.

Usage (Windows):
  py analyze_type22_regression.py --in type22_dataset.csv --out type22_derived_surrogate_metrics.csv

It will save:
  1) Derived dataset CSV (new file)
  2) Regression plots (PNG files)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


def make_derived_dataset(
    in_csv: str,
    out_csv: str,
    near_conflict_tti: float = 5.0,
    min_dist_conflict: float = 0.1,
    keep_only_collision_scenarios: bool = False,
) -> pd.DataFrame:
    """
    Reads raw CSV -> creates a copy -> derives TTC_conflict, PET, DRAC -> saves to out_csv.
    NEVER overwrites/modifies the input file.
    """

    raw_df = pd.read_csv(in_csv)
    df = raw_df.copy()

    # Replace inf/-inf with NaN so we can filter safely
    df = df.replace([np.inf, -np.inf], np.nan)

    required = ["ego_tti_s", "other_tti_s", "ego_speed_mps", "ego_dist_to_conflict_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Drop rows where we cannot compute derived metrics
    df = df.dropna(subset=required)

    # Optional: focus only on near-conflict frames for better correlation meaning
    df = df[(df["ego_tti_s"] < near_conflict_tti) & (df["other_tti_s"] < near_conflict_tti)]

    # Derived metrics (conflict-point based)
    df["TTC_conflict"] = df[["ego_tti_s", "other_tti_s"]].min(axis=1)
    df["PET"] = (df["ego_tti_s"] - df["other_tti_s"]).abs()

    # DRAC = v^2 / (2d) using ego vehicle (you can also compute for "other" similarly)
    df["ego_dist_to_conflict_m"] = df["ego_dist_to_conflict_m"].clip(lower=min_dist_conflict)
    df["DRAC"] = (df["ego_speed_mps"] ** 2) / (2.0 * df["ego_dist_to_conflict_m"])

    # Optional: keep only rows from scenarios that eventually collided
    # If your dataset does NOT include scenario_id, we approximate scenario blocks by frame order:
    # - assuming you recorded exactly N frames per scenario sequentially.
    # If you DO have scenario_id, this block is not needed.
    if keep_only_collision_scenarios:
        if "scenario_id" in df.columns:
            collided_scen = df.groupby("scenario_id")["collision_flag"].max()
            collided_ids = set(collided_scen[collided_scen > 0].index)
            df = df[df["scenario_id"].isin(collided_ids)]
        else:
            # If no scenario_id, we can only filter by collision_flag==1 frames (not whole scenarios)
            df = df[df.get("collision_flag", 0) == 1]

    # Save NEW file (never overwrite input unless user explicitly sets same path)
    in_abs = os.path.abspath(in_csv)
    out_abs = os.path.abspath(out_csv)
    if in_abs == out_abs:
        raise ValueError("Output CSV path is the same as input CSV. Choose a NEW output file name.")

    df.to_csv(out_csv, index=False)
    return df


def fit_and_plot_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_png: str,
    degree: int = 1,
    max_points: int = 30000,
):
    """
    Fits OLS (LinearRegression). If degree>1, uses PolynomialFeatures + LinearRegression.
    Saves scatter + fitted curve/line to PNG.
    """

    # Optionally subsample for plotting speed (regression is still fit on full data by default)
    plot_df = df[[x_col, y_col]].dropna()
    if len(plot_df) == 0:
        raise ValueError(f"No valid rows for plotting {y_col} vs {x_col}")

    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42)

    X = df[[x_col]].values
    y = df[y_col].values

    # Build model
    if degree <= 1:
        model = LinearRegression()
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr", LinearRegression())
        ])

    # Fit
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Prepare smooth curve for plot
    x_min = np.nanmin(df[x_col].values)
    x_max = np.nanmax(df[x_col].values)
    x_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    y_grid = model.predict(x_grid)

    # Plot
    plt.figure()
    plt.scatter(plot_df[x_col].values, plot_df[y_col].values, s=2, alpha=0.25)
    plt.plot(x_grid, y_grid)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    title = f"{y_col} vs {x_col} (OLS degree={degree}, R²={r2:.3f})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return r2


def main():
    ap = argparse.ArgumentParser("Derive TTC/PET/DRAC dataset (new file) + regression plots (scikit-learn OLS)")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input RAW CSV (will NOT be modified)")
    ap.add_argument("--out", dest="out_csv", default="type22_derived_surrogate_metrics.csv",
                    help="Output DERIVED CSV (new file)")
    ap.add_argument("--near-tti", type=float, default=5.0, help="Keep frames where ego_tti and other_tti < this (sec)")
    ap.add_argument("--poly-degree", type=int, default=1, help="1=linear, 2+=polynomial curve fit")
    ap.add_argument("--plots-dir", type=str, default="plots", help="Directory to save regression plots")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    df = make_derived_dataset(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        near_conflict_tti=args.near_tti,
    )

    # Regression relationships you asked for
    # 1) DRAC vs TTC_conflict
    r2_1 = fit_and_plot_regression(
        df, "TTC_conflict", "DRAC",
        out_png=os.path.join(args.plots_dir, "DRAC_vs_TTC_conflict.png"),
        degree=args.poly_degree
    )

    # 2) DRAC vs PET
    r2_2 = fit_and_plot_regression(
        df, "PET", "DRAC",
        out_png=os.path.join(args.plots_dir, "DRAC_vs_PET.png"),
        degree=args.poly_degree
    )

    # 3) PET vs TTC_conflict
    r2_3 = fit_and_plot_regression(
        df, "TTC_conflict", "PET",
        out_png=os.path.join(args.plots_dir, "PET_vs_TTC_conflict.png"),
        degree=args.poly_degree
    )

    print("✅ Derived dataset saved to:", args.out_csv)
    print("✅ Plots saved in folder:", args.plots_dir)
    print(f"R² (DRAC vs TTC_conflict): {r2_1:.3f}")
    print(f"R² (DRAC vs PET):         {r2_2:.3f}")
    print(f"R² (PET  vs TTC_conflict): {r2_3:.3f}")


if __name__ == "__main__":
    main()

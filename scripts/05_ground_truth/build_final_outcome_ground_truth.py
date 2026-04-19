import pandas as pd
import numpy as np

INPUT_CSV = "data/interim/type22_valid_precollision_rows.csv"
OUTPUT_CSV = "data/processed/type22_final_labeled_dataset.csv"

# Replace these with your final thesis values
T_REACTION = 0.8
A_NOMINAL = 6.0
T_BUFFER = 0.5

df = pd.read_csv(INPUT_CSV).copy()

# Build scenario id if missing
if "scenario_id" not in df.columns:
    df["scenario_id"] = df["ego_id"].astype(str) + "_" + df["other_id"].astype(str)

# Read original file again to find actual collision times from all rows
raw = pd.read_csv("data/interim/type22_derived_surrogate_metrics.csv").copy()
if "scenario_id" not in raw.columns:
    raw["scenario_id"] = raw["ego_id"].astype(str) + "_" + raw["other_id"].astype(str)

collision_times = (
    raw.loc[raw["collision_flag"] == 1]
       .groupby("scenario_id")["time"]
       .min()
       .rename("t_collision")
)

df = df.merge(collision_times, on="scenario_id", how="left")

# Whether this scenario ever collides
df["has_collision"] = df["t_collision"].notna()

# Time remaining to actual collision
df["dt_to_collision"] = df["t_collision"] - df["time"]

# Post-collision rows, if any, should not remain in training data
df["post_collision"] = df["has_collision"] & (df["dt_to_collision"] < 0)

# Avoidability window
df["T_avoid"] = (
    T_REACTION +
    (df["ego_speed_mps"] / A_NOMINAL) +
    T_BUFFER
)

# Final binary ground truth
df["gt_binary"] = "Low"

mask_high = (
    df["has_collision"] &
    (~df["post_collision"]) &
    (df["dt_to_collision"] >= 0) &
    (df["dt_to_collision"] <= df["T_avoid"])
)

df.loc[mask_high, "gt_binary"] = "High"

# Optional continuous outcome-grounded score
df["R_outcome"] = 0.0
df.loc[mask_high, "R_outcome"] = 1 - (
    df.loc[mask_high, "dt_to_collision"] / df.loc[mask_high, "T_avoid"]
)
df["R_outcome"] = df["R_outcome"].clip(0, 1)

# Keep only non-post-collision rows for modeling
df_final = df.loc[~df["post_collision"]].copy()

df_final.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("\nBinary label distribution:")
print(df_final["gt_binary"].value_counts(dropna=False))
print("\nR_outcome summary:")
print(df_final["R_outcome"].describe())
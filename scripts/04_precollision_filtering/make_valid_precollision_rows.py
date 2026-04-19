import pandas as pd
import numpy as np

INPUT_CSV = "data/interim/type22_derived_surrogate_metrics.csv"
OUTPUT_CSV = "data/interim/type22_valid_precollision_rows.csv"

# Calibrated exploratory-filtering weights
A_TTC = 0.5451
B_DRAC = 0.4549

# Normalization caps
TTC_MAX = 5.0
DRAC_MAX = 8.0

df = pd.read_csv(INPUT_CSV).copy()

# Build scenario id if missing
if "scenario_id" not in df.columns:
    df["scenario_id"] = df["ego_id"].astype(str) + "_" + df["other_id"].astype(str)

# Keep only valid pre-collision near-conflict rows
valid_mask = (
    df["TTC_conflict"].notna() &
    df["DRAC"].notna() &
    (df["TTC_conflict"] > 0) &
    (df["collision_flag"] == 0)
)

valid = df.loc[valid_mask].copy()

# Normalize TTC into risk: smaller TTC -> higher risk
valid["T_risk"] = 1 - (np.minimum(valid["TTC_conflict"], TTC_MAX) / TTC_MAX)

# Normalize DRAC into risk: larger DRAC -> higher risk
valid["D_risk"] = np.minimum(valid["DRAC"], DRAC_MAX) / DRAC_MAX

# Clip to [0, 1]
valid["T_risk"] = valid["T_risk"].clip(0, 1)
valid["D_risk"] = valid["D_risk"].clip(0, 1)

# Exploratory filtering score (NOT final ground truth)
valid["R_filter"] = (
    A_TTC * valid["T_risk"] +
    B_DRAC * valid["D_risk"]
).clip(0, 1)

# Optional coarse exploratory class
valid["risk_class"] = "Low"
valid.loc[(valid["R_filter"] >= 1/3) & (valid["R_filter"] < 2/3), "risk_class"] = "Medium"
valid.loc[valid["R_filter"] >= 2/3, "risk_class"] = "High"

valid.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows kept:", len(valid))
print("\nRisk class distribution:")
print(valid["risk_class"].value_counts(dropna=False))
print("\nR_filter summary:")
print(valid["R_filter"].describe())
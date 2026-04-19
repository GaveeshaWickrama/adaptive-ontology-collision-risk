import pandas as pd
import numpy as np

# =========================
# 1. Load derived dataset
# =========================
df = pd.read_csv("data/interim/type22_derived_surrogate_metrics.csv")

# =========================
# 2. Build scenario id
# =========================
df["scenario_id"] = df["ego_id"].astype(str) + "_" + df["other_id"].astype(str)

# =========================
# 3. Keep only valid rows
# =========================
# Pre-collision, valid TTC_conflict, valid DRAC
valid = df.copy()
valid = valid[
    valid["TTC_conflict"].notna() &
    valid["DRAC"].notna() &
    (valid["TTC_conflict"] > 0) &
    (valid["collision_flag"] == 0)
].copy()

# =========================
# 4. Keep only scenarios that eventually collide
# =========================
collision_scenarios = (
    df.groupby("scenario_id")["collision_flag"]
      .max()
      .reset_index()
)
collision_scenarios = collision_scenarios[collision_scenarios["collision_flag"] == 1]["scenario_id"]

valid = valid[valid["scenario_id"].isin(collision_scenarios)].copy()

# =========================
# 5. Normalize TTC-risk and DRAC-risk
# =========================
# You can adjust these caps if needed
TTC_MAX = 5.0
DRAC_MAX = 8.0

# TTC smaller => higher risk
valid["T_risk"] = 1 - np.minimum(valid["TTC_conflict"], TTC_MAX) / TTC_MAX

# DRAC larger => higher risk
valid["D_risk"] = np.minimum(valid["DRAC"], DRAC_MAX) / DRAC_MAX

# Clip to [0, 1]
valid["T_risk"] = valid["T_risk"].clip(0, 1)
valid["D_risk"] = valid["D_risk"].clip(0, 1)

# =========================
# 6. Sort within scenario by time
# =========================
valid = valid.sort_values(["scenario_id", "time"]).copy()

# =========================
# 7. Split each scenario into early and late parts
# =========================
# early = first 30% of valid conflict frames
# late  = last 30% of valid conflict frames
def label_phase(group):
    n = len(group)
    if n < 10:
        group["phase"] = "ignore"
        return group

    early_end = int(np.floor(0.3 * n))
    late_start = int(np.floor(0.7 * n))

    phase = np.array(["middle"] * n, dtype=object)
    phase[:early_end] = "early"
    phase[late_start:] = "late"

    group = group.copy()
    group["phase"] = phase
    return group

valid = valid.groupby("scenario_id", group_keys=False).apply(label_phase)
valid = valid[valid["phase"] != "ignore"].copy()

# =========================
# 8. Discriminative power
# =========================
# Mean difference between late and early frames
disc_t = (
    valid.groupby("phase")["T_risk"].mean()["late"]
    - valid.groupby("phase")["T_risk"].mean()["early"]
)

disc_d = (
    valid.groupby("phase")["D_risk"].mean()["late"]
    - valid.groupby("phase")["D_risk"].mean()["early"]
)

disc_t = max(disc_t, 0)
disc_d = max(disc_d, 0)

# =========================
# 9. Progression toward collision
# =========================
# For each scenario, correlate metric with frame order
prog_t_list = []
prog_d_list = []

for sid, g in valid.groupby("scenario_id"):
    g = g.sort_values("time").copy()
    if len(g) < 10:
        continue

    g["progress"] = np.linspace(0, 1, len(g))

    # correlation with progress
    t_corr = g["T_risk"].corr(g["progress"])
    d_corr = g["D_risk"].corr(g["progress"])

    if pd.notna(t_corr):
        prog_t_list.append(max(t_corr, 0))
    if pd.notna(d_corr):
        prog_d_list.append(max(d_corr, 0))

prog_t = np.mean(prog_t_list) if len(prog_t_list) > 0 else 0
prog_d = np.mean(prog_d_list) if len(prog_d_list) > 0 else 0

# =========================
# 10. Combine the two ideas
# =========================
# 50% discriminative power + 50% progression
score_t = 0.5 * disc_t + 0.5 * prog_t
score_d = 0.5 * disc_d + 0.5 * prog_d

# =========================
# 11. Convert to weights
# =========================
total = score_t + score_d
if total == 0:
    a = 0.5
    b = 0.5
else:
    a = score_t / total
    b = score_d / total

print("Discriminative TTC score :", disc_t)
print("Discriminative DRAC score:", disc_d)
print("Progression TTC score    :", prog_t)
print("Progression DRAC score   :", prog_d)
print("Final TTC weight (a)     :", round(a, 4))
print("Final DRAC weight (b)    :", round(b, 4))
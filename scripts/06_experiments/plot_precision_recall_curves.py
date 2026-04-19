import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path

# -----------------------------
# Load Experiment 4
# -----------------------------
exp4 = pd.read_csv("outputs/metrics/exp4_fuzzy_ontology_inspired_reasoning/exp4_test_inferred_labels.csv")
y_true_exp4 = exp4["gt_binary_num"]
y_score_exp4 = exp4["fuzzy_risk_score"]

# -----------------------------
# Load Decision Tree
# -----------------------------
dt = pd.read_csv("outputs/metrics/ml_decision_tree_baseline/dt_test_predictions.csv")
y_true_dt = dt["y"]
y_score_dt = dt["y_score"]

# -----------------------------
# Load Experiment 5
# -----------------------------
exp5 = pd.read_csv("outputs/metrics/exp5_binary_results/binary_test_predictions.csv")
y_true_exp5 = exp5["y"]
y_score_exp5 = exp5["adaptive_score"]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

for name, y_true, y_score in [
    ("Experiment 4", y_true_exp4, y_score_exp4),
    ("Experiment 5", y_true_exp5, y_score_exp5),
    ("Decision Tree", y_true_dt, y_score_dt),
]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves on the Test Set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = Path("outputs/figures/precision_recall_curves.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
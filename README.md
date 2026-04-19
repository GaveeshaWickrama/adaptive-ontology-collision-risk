# Adaptive Ontology-Based Collision Risk Assessment for Autonomous Driving

This repository contains the research code for generating CARLA-based driving scenarios, constructing derived surrogate safety metrics, building outcome-based ground-truth labels, and evaluating ontology-inspired and machine-learning baselines for collision risk assessment.

## Repository overview

The project follows this pipeline:

1. **CARLA scenario generation**
   - Script: `scripts/01_generate_dataset/carla_dataset_type22.py`
   - Requirement: CARLA simulator must already be running in the background
   - Requirement: the CARLA Python API / matching `.egg` file must be available in the Python environment
   - Output: `data/raw/type22_dataset.csv`
   - If the dataset has already been generated, copy it into `data/raw/` and continue with the remaining steps

2. **Derived surrogate-metric construction + regression/correlation analysis**
   - Script: `scripts/02_surrogate_metrics/correlation_checking.py`
   - Input: `data/raw/type22_dataset.csv`
   - Output: `data/interim/type22_derived_surrogate_metrics.csv`
   - Output: regression/correlation plots in `outputs/figures/`
   - Example command from inside the script folder:
     `py correlation_checking.py --in type22_dataset.csv --out type22_derived_surrogate_metrics.csv`

3. **Exploratory TTC/DRAC weight estimation**
   - Script: `scripts/03_weight_estimation/getting_a_b_values_for_ground_truth.py`
   - Input: `data/interim/type22_derived_surrogate_metrics.csv`
   - Output: estimated TTC weight `a` and DRAC weight `b`

4. **Exploratory valid pre-collision filtering**
   - Script: `scripts/04_precollision_filtering/make_valid_precollision_rows.py`
   - Input: `data/interim/type22_derived_surrogate_metrics.csv`
   - Uses: `a`, `b`, `TTC_MAX`, `DRAC_MAX`
   - Output: `data/interim/type22_valid_precollision_rows.csv`

5. **Final outcome-based ground-truth construction**
   - Script: `scripts/05_ground_truth/build_final_outcome_ground_truth.py`
   - Input: `data/interim/type22_valid_precollision_rows.csv`
   - Input: `data/interim/type22_derived_surrogate_metrics.csv`
   - Output: `data/processed/type22_final_labeled_dataset.csv`

6. **Main experiments + ML baselines**
   - Folder: `scripts/06_experiments/`
   - `exp1_static_fixed_gate.py` - static fixed-gate baseline
   - `exp2_deterministic_context.py` - deterministic contextual thresholding
   - `exp3_instances.py` - ontology-equivalent / instance-based experiment
   - `exp3_python_reasoning_split.py` - Python reasoning split variant
   - `exp4_fuzzy_ontology_inspired_reasoning.py` - fuzzy ontology-inspired reasoning
   - `binary_threshold_evaluation.py` - adaptive threshold model evaluation (Experiment 5)
   - `dt_ml_baseline.py` - decision tree baseline
   - `rf_paper_style_binary.py` - random forest baseline
   - `plot_precision_recall_curves.py` - precision-recall curve comparison across selected models

## Folder structure

- `data/raw/` - original generated datasets
- `data/interim/` - intermediate processed CSV files
- `data/processed/` - final labeled datasets
- `outputs/figures/` - plots and figures
- `outputs/logs/` - logs from runs
- `outputs/metrics/` - evaluation summaries
- `scripts/` - pipeline and experiment scripts
- `src/` - reusable helper functions
- `docs/` - project documentation


## Setup

### 1. Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
## Experiments

The main experiment and baseline scripts are stored in `scripts/06_experiments/`:

- `exp1_static_fixed_gate.py`
- `exp2_deterministic_context.py`
- `exp3_instances.py`
- `exp3_python_reasoning_split.py`
- `exp4_fuzzy_ontology_inspired_reasoning.py`
- `binary_threshold_evaluation.py` - Experiment 5
- `dt_ml_baseline.py`
- `rf_paper_style_binary.py`
- `plot_precision_recall_curves.py`

## Evaluation outputs

Recommended output locations:

- prediction/evaluation CSV files -> `outputs/metrics/`
- plots and figures -> `outputs/figures/`

For example, the precision-recall comparison plot can be saved as:

- `outputs/figures/precision_recall_curves.png`
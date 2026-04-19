# Research pipeline

## Workflow

CARLA scenario generation  
→ derived surrogate-metric construction + regression/correlation analysis  
→ exploratory TTC/DRAC weight estimation  
→ exploratory valid pre-collision filtering  
→ final outcome-based ground-truth construction  
→ main experiments + ML baselines

## Detailed mapping

### 1. CARLA scenario generation
- Script: `carla_dataset_type22.py`
- Output: `type22_dataset.csv`

### 2. Derived surrogate-metric construction + regression/correlation analysis
- Script: `correlation_checking.py`
- Input: `type22_dataset.csv`
- Output: `type22_derived_surrogate_metrics.csv`
- Output: regression plots for TTC_conflict / PET / DRAC relationships

### 3. Exploratory TTC/DRAC weight estimation
- Script: `getting_a_b_values_for_ground_truth.py`
- Input: `type22_derived_surrogate_metrics.csv`
- Output: TTC weight `a`, DRAC weight `b`

### 4. Exploratory valid pre-collision filtering
- Script: `make_valid_precollision_rows.py`
- Input: `type22_derived_surrogate_metrics.csv`
- Uses: `a`, `b`, `TTC_MAX`, `DRAC_MAX`
- Output: `type22_valid_precollision_rows.csv`

### 5. Final outcome-based ground-truth construction
- Script: `build_final_outcome_ground_truth.py`
- Input: `type22_valid_precollision_rows.csv`
- Input: `type22_derived_surrogate_metrics.csv` for first collision time
- Output: `type22_final_labeled_dataset.csv`

### 6. Main experiments + ML baselines
- `exp1_static_fixed_gate.py`
- `exp2_deterministic_context.py`
- `exp3_instances.py`
- `exp3_python_reasoning_split.py`
- `exp4_fuzzy_ontology_inspired_reasoning.py`
- `binary_threshold_evaluation.py`
- `dt_ml_baseline.py`
- `rf_paper_style_binary.py`
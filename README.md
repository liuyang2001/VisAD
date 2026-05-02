# VisTASD Project

## 0. Project Directory

```text
├── config/                 # Configuration files
├── dataset/                # Dataset directory (SKAB & ATSADBench)
├── src/                    # Core source code
├── tools/                  # Initialization & Sensitivity scripts
├── main.py                 # Main entry point for inference
├── eval_metrics.py         # Evaluation script
└── requirements.txt
```

The ATSADBench and SKAB datasets have been placed in the dataset/ATSADBench and dataset/SKAB directories, respectively. 
Note that ATSADBench is only partially public due to confidentiality agreements; the release includes 800 data points, comprising normal references and test data for the three anomaly detection tasks (FVA, CDA, TVDA).
All supplementary materials are included in the appendix.pdf

## 1. Setup and Preparation

**Step 1: Install Dependencies**

Install the required dependencies based on `requirement.txt`:
```bash
pip install -r requirement.txt
```

**Step 2: Configuration**

Open config/main_config_skab.yaml and config/main_config_atsad.yaml, and fill in your API Key:
- `model_name`
- `api_key`
- `base_url`

**Step 3: Build Reference Libraries**

Run the initialization scripts to build the reference normal window set:
```bash
# For SKAB
python tools/build_skab_lib.py

# For ATSADBench
python tools/build_atsad_lib.py
```

---

## 2. Running VisTASD

**Step 1: Run Main Script**

Run the inference using the standard mode (all views):
```bash
# For SKAB
python main.py --config config/main_config_skab.yaml --mode all

# For ATSADBench
python main.py --config config/main_config_atsad.yaml --mode all
```

**Step 2: Get Metrics**

Run the evaluation script to obtain statistical indicators:
```bash
# For SKAB
python eval_metrics.py --config config/main_config_skab.yaml --mode all

# For ATSADBench
python eval_metrics.py --config config/main_config_atsad.yaml --mode all
```

---

## 3. Ablation Studies

**Step 1: Run Main Script with Different Modes**

Select one of the following modes: eu_only(NoVarCorr), mi_only(NoGeoDis), value_only(NoTimefeat), or no_residual(NoResGraph).

```bash
# Example: Running No-Residual mode on SKAB
python main.py --config config/main_config_skab.yaml --mode no_residual

# Example: Running EU-Only mode on ATSADBench
python main.py --config config/main_config_atsad.yaml --mode eu_only
```

**Step 2: Get Metrics**

Calculate metrics for the specific mode:
```bash
# Example
python eval_metrics.py --config config/main_config_skab.yaml --mode no_residual
```

## 4. Different Scaling Factors

**Step 1: Run Automation Script**

Run the sensitivity analysis tools. These scripts automatically scale parameters from 0.25x to 2.00x, perform inference, and calculate metrics.

```bash
# For SKAB
python tools/run_sensitivity_skab.py

# For ATSADBench
python tools/run_sensitivity_atsad.py
```

**Step 2: Check Results**

The summary reports will be saved directly to:
- results_skab/sensitivity/FINAL_SUMMARY_SKAB.xlsx
- results_atsad/sensitivity/FINAL_SUMMARY_ATSADBench.xlsx

import sys
import json
import pandas as pd
import numpy as np
import argparse
import yaml
import time
import re
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from src.feature_engine import FeatureEngine
from src.utils_io import read_data_file
from main import get_window_generator
from src.vlm_agent import VLMAgent

PROMPT_TEXT_ONLY = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only numerical matrices.

You will be given 6 matrices describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.

=====================================================
MATRICES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This matrix is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The numerical scale is:
        - 0.000  ≈ No difference from NORMAL,
        - -1.000 ≈ Maximum negative deviation,
        - 1.000  ≈ Maximum positive deviation.
      The further from 0, the larger the difference from NORMAL.

(2) TEST Euclidean-diff (diff)
    - Same as (1), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(3) TEST Euclidean-diff (var)
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 16).

(4) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise mutual information matrix is computed for the NORMAL window.
    - This matrix is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].

(5) TEST MI-diff (diff)
    - Same as (4), but the mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST MI-diff (var)
    - Same as (4), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 16).

Variable names appear as row and column headers. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff matrices and MI-diff matrices (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - **IMPORTANT: Be highly sensitive. If even a faint or subtle anomaly is visible in the numerical values of any single one of the six residual matrices, you MUST prioritize it and label those time steps as anomalous (1). Do not ignore minor deviations from 0.**
   - The output should be a binary array of length {window_size}.

2. Root Cause Analysis
   - If any time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the matrices to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the matrices to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

def apply_soft_threshold(matrix, t1, t2):
    A = np.abs(matrix)
    sign = np.sign(matrix)
    out = np.zeros_like(matrix, dtype=float)
    denom = (t2 - t1) if (t2 - t1) > 1e-12 else 1e-12
    mid_mask = (A >= t1) & (A < t2)
    out[mid_mask] = (A[mid_mask] - t1) / denom
    high_mask = A >= t2
    out[high_mask] = 1.0
    return out * sign

def format_matrix_to_text(matrix, var_names):
    short_names = [n[:8] for n in var_names]
    header = " " * 10 + " ".join([f"{n:>10}" for n in short_names])
    lines = [header]
    for i, row in enumerate(matrix):
        row_str = f"{short_names[i]:>8}: " + " ".join([f"{val:10.8f}" for val in row])
        lines.append(row_str)
    return "\n".join(lines)

def custom_parser(raw_text, window_size):
    json_match = re.search(r"(.*?)((?:```json\s*)?\{.*?\}(?:\s*```)?)", raw_text, re.DOTALL)
    analysis_text = ""
    parsed_data = {"Label": [0] * window_size, "root_cause_variables": []}
    if json_match:
        pre_text = json_match.group(1).strip()
        analysis_text = pre_text.split("Analysis Process:")[-1].strip() if "Analysis Process:" in pre_text else pre_text
        json_content = json_match.group(2)
        json_str = re.search(r"\{.*\}", json_content, re.DOTALL)
        if json_str:
            try:
                data = json.loads(json_str.group(0))
                inner_analysis = next((v for k, v in data.items() if "analysis" in k.lower()), None)
                if inner_analysis: analysis_text = inner_analysis
                source = data
                if "Final Answer" in data: source = data["Final Answer"]
                elif "final_answer" in data: source = data["final_answer"]
                label_key = next((k for k in source.keys() if "label" in k.lower()), None)
                if label_key: parsed_data["Label"] = source[label_key]
                rc_key = next((k for k in source.keys() if "root" in k.lower()), None)
                if rc_key: parsed_data["root_cause_variables"] = source[rc_key]
            except: pass
    labels = parsed_data["Label"]
    if len(labels) < window_size: labels += [0] * (window_size - len(labels))
    parsed_data["Label"] = labels[:window_size]
    parsed_data["Analysis Process"] = analysis_text
    return parsed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT_DIR / args.config, encoding="utf-8"))
    is_atsad = "atsad" in args.config.lower()
    if is_atsad:
        data_dir = ROOT_DIR / "data" / "dataset" / "ATSADBench"
        num_vars = 27
    else:
        data_dir = ROOT_DIR / "data" / "dataset" / "SKAB"
        num_vars = 8
    window_size = cfg['experiment'].get('window_size', 32)
    stride = cfg['experiment'].get('stride', 32)
    params_map = json.load(open(ROOT_DIR / cfg['paths']['dataset_params'], 'r', encoding="utf-8"))
    engine = FeatureEngine(ROOT_DIR / cfg['paths']['resources'])
    agent = VLMAgent(cfg['model'], window_size=window_size)
    base_out = ROOT_DIR / cfg['paths']['output_dir'] / "text_ablation"
    base_out.mkdir(parents=True, exist_ok=True)
    first_prompt_logged = False
    for fname in cfg['experiment']['target_files']:
        clean_name = Path(fname).stem
        print(f"\nProcessing File: {fname}")
        json_save_dir = base_out / "json_details" / clean_name
        matrix_save_dir = base_out / "debug_matrices" / clean_name
        json_save_dir.mkdir(parents=True, exist_ok=True)
        matrix_save_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / fname
        if not file_path.exists(): continue
        df = read_data_file(file_path)
        var_names = df.columns[:num_vars].tolist()
        file_params = params_map.get(fname, {})
        file_results = []
        window_gen = get_window_generator(df, window_size, stride, num_vars)
        for win_info in tqdm(window_gen, desc="Analyzing"):
            win_id = win_info["id"]
            raw_residuals = engine.process_single_window(win_info["data"])
            matrix_sections = []
            m_keys = ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]
            for idx, k in enumerate(m_keys):
                p = file_params.get(k, {"t1": 0.45, "t2": 0.50})
                thresh_mat = apply_soft_threshold(raw_residuals[k], p["t1"], p["t2"])
                mat_str = format_matrix_to_text(thresh_mat, var_names)
                matrix_sections.append(f"({idx+1}) {k.upper()} RESIDUAL MATRIX:\n{mat_str}")
            payload = "\n\n".join(matrix_sections)
            with open(matrix_save_dir / f"matrices_w_{win_id:05d}.txt", "w", encoding="utf-8") as f:
                f.write(payload)
            full_prompt = PROMPT_TEXT_ONLY.format(window_size=window_size) + "\n\n### INPUT DATA (Residuals):\n" + payload
            if not first_prompt_logged:
                with open(base_out / f"first_prompt_sample.txt", "w", encoding="utf-8") as f:
                    f.write(full_prompt)
                first_prompt_logged = True
            start_time = time.perf_counter()
            try:
                response = agent.client.chat.completions.create(
                    model=agent.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0
                )
                raw_response_text = response.choices[0].message.content
                parsed_res = custom_parser(raw_response_text, window_size)
            except Exception as e:
                print(f"API Error: {e}")
                raw_response_text = f"Error: {str(e)}"
                parsed_res = {"Label": [0]*window_size, "root_cause_variables": [], "Analysis Process": "Failed"}
            elapsed = round(time.perf_counter() - start_time, 4)

            json_content = {
                "window_id": win_id,
                "filename": fname,
                "mode": "text_ablation",
                "latency_seconds": elapsed,
                "parsed_result": parsed_res,        
                "raw_model_response": raw_response_text, 
                "raw_text_input": payload  
            }
            with open(json_save_dir / f"vlm_reply_w_{win_id:05d}.json", "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)

            file_results.append({
                "Window_ID": win_id, "Start_Time": win_info["start"], "End_Time": win_info["end"],
                "Ground_Truth_Window": 1 if sum(win_info["labels"]) > 0 else 0,
                "Pred_Label_Window": 1 if sum(parsed_res["Label"]) > 0 else 0,
                "Raw_Pred_Array": str(parsed_res["Label"]), "Latency": elapsed
            })
        pd.DataFrame(file_results).to_excel(base_out / f"Result_{clean_name}_text_ablation.xlsx", index=False)

if __name__ == "__main__":
    main()
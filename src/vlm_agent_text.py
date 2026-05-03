import json
import re
import time
import numpy as np
from openai import OpenAI

PROMPT_TEXT_ALL = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series.
You will be provided with 6 numerical matrices describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.
The matrices are in normalized difference format [-1, 1].

=====================================================
Numerical Matrices (in order)
=====================================================
(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This final matrix is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The scale is: [-1, 1].
      The closer to 0, the smaller the difference; the closer to -1 or 1, the larger the difference from NORMAL.

(2) TEST Euclidean-diff (diff)
    - Same as (1), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(3) TEST Euclidean-diff (var)
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 16).

(4) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise mutual information matrix is computed for the NORMAL window.
    - This final matrix is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The scale is: [-1,1].
      The closer to 0, the smaller the difference; the closer to -1 or 1, the larger the difference from NORMAL.

(5) TEST MI-diff (diff)
    - Same as (4), but the mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST MI-diff (var)
    - Same as (4), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 16).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff matrices and MI-diff matrices (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.
2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The Euclidean-diff matrices (1)-(3)
       - The MI-diff matrices (4)-(6)

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

class VLMAgentText:
    def __init__(self, config, window_size=64):
        self.client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        self.model_name = config.get('model_name', "gemini-3-pro-preview")
        self.window_size = window_size

    def _matrix_to_text(self, matrix, var_names):
        
        col_width = 30
        
        header = " " * col_width + "".join([f"{name:>{col_width}}" for name in var_names])
        lines = [header]
        
        for i, row in enumerate(matrix):
            row_str = f"{var_names[i]:<{col_width}}"
            

            row_str += "".join([f"{val:>{col_width}.8f}" for val in row])
            lines.append(row_str)
            
        return "\n".join(lines)

    def _parse_json(self, text):
        is_padded = False
        
        analysis_process = "No process found"
        if "Analysis Process" in text:
            try:
                parts = re.split(r'Analysis Process["\']?\s*[:\-]?\s*', text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    analysis_process = parts[1].split("Final Answer:")[0].strip()
            except: pass

        json_candidates = re.findall(r'\{.*?\s*"Label":\s*\[.*?\]\s*\}', text, re.DOTALL)
        
        target_label = None
        target_rc = []
        
        if json_candidates:
            try:
                data = json.loads(json_candidates[0].replace('\\"', '"'))
                target_label = data.get("Label")
                target_rc = data.get("root_cause_variables", [])
            except:
                pass
        
        if target_label is None or not isinstance(target_label, list):
            target_label = [0] * self.window_size
            is_padded = True
        elif len(target_label) != self.window_size:
            target_label = target_label[:self.window_size] + [0] * max(0, self.window_size - len(target_label))
            
        return {
            "Label": target_label,
            "root_cause_variables": target_rc,
            "analysis_process": analysis_process,
            "is_padded": is_padded
        }
    def analyze(self, mode, matrices_map,var_names):
        prompt_text = PROMPT_TEXT_ALL.format(window_size=self.window_size)
        
        content = prompt_text + "\n\n================ MATRIX DATA ================\n"
        for k in ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]:
            if k in matrices_map:
                content += f"\nMatrix: {k}\n" + self._matrix_to_text(matrices_map[k],var_names) + "\n"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"attempt:{attempt}")
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0
                )
                end_time = time.perf_counter()
                
                raw_text = response.choices[0].message.content
                result = self._parse_json(raw_text)
                
                result["latency"] = round(end_time - start_time, 4)
                result["raw_model_response"] = raw_text
                result["full_prompt"] = content
                print(f"attempt:{attempt} success")
                return result
            
            except Exception as e:
                print(f"attempt:{attempt} error:{e}")
                print(f"attempt:{attempt} fail")
                print("sleep 2s...")
                time.sleep(2)
        
        return {
            "Label": [0]*self.window_size, 
            "root_cause_variables": [], 
            "latency": 0.0, 
            "is_padded": True, 
            "error": "api_failed"
        }
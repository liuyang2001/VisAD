import yaml
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from src.feature_engine import FeatureEngine
from src.visualizer import ParametricVisualizer
from src.vlm_agent import VLMAgent
from src.utils_io import read_data_file 

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_segments(df, window_size):
    segments = []
    start_idx = 0
    df = df.reset_index(drop=True)
    
    if 'Segment_boundary' not in df.columns:
        if len(df) >= window_size:
            segments.append(df)
        return segments

    boundary_indices = df.index[df['Segment_boundary'] == 1].tolist()
    
    for b_idx in boundary_indices:
        seg = df.loc[start_idx : b_idx]
        if len(seg) >= window_size:
            segments.append(seg)
        start_idx = b_idx + 1
        
    if start_idx < len(df):
        seg = df.loc[start_idx:]
        if len(seg) >= window_size:
            segments.append(seg)
            
    return segments

def get_window_generator(df, window_size, stride, num_vars):

    segments = get_segments(df, window_size)
    global_win_id = 0
    
    for seg_df in segments:
        base_idx = seg_df.index[0]
        
        seg_vals = seg_df.iloc[:, :num_vars].values
        seg_labels = seg_df['Label'].values
        seg_len = len(seg_df)
        
        num_wins_in_seg = (seg_len - window_size) // stride + 1
        
        for i in range(num_wins_in_seg):
            rel_start = i * stride
            rel_end = rel_start + window_size
            
            abs_start = base_idx + rel_start
            abs_end = base_idx + rel_end
            
            yield {
                "id": global_win_id,
                "start": abs_start,
                "end": abs_end,
                "data": seg_vals[rel_start:rel_end],
                "labels": seg_labels[rel_start:rel_end]
            }
            global_win_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config_atsad.yaml", help="Path to main config")
    parser.add_argument("--params_file", type=str, default=None, help="Override dataset_params.json path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--mode", type=str, default="all", help="Override experiment mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    if args.mode: mode = args.mode
    else: mode = cfg['experiment']['mode']
    
    is_no_residual = (mode == "no_residual")
    print(f">>> [Init] Mode: {mode} (No Residual: {is_no_residual})")

    if args.output_dir: base_output_dir = Path(args.output_dir)
    else: base_output_dir = Path(cfg['paths']['output_dir']) / mode
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    img_save_dir = base_output_dir / "debug_images"

    if is_no_residual:
        print(">>> [Init] No-Residual Mode: Skipping params loading.")
        params_path = None
    else:
        if args.params_file:
            params_path = Path(args.params_file)
        else:
            base_param_dir = Path(cfg['paths']['dataset_params']).parent
            params_path = base_param_dir / f"dataset_params_{mode}.json"
        
        print(f">>> [Init] Loading Params: {params_path}")
        if not params_path.exists():
             # Fallback
             params_path = Path(cfg['paths']['dataset_params'])
             if not params_path.exists():
                 raise FileNotFoundError(f"Params file {params_path} not found.")

    engine = FeatureEngine(cfg['paths']['resources'])
    
    window_size = cfg['experiment'].get('window_size', 64)
    stride = cfg['experiment'].get('stride', 64)
    num_vars = cfg['experiment'].get('num_vars', 8)
    
    print(f">>> [Config] Window: {window_size} | Stride: {stride} | Vars: {num_vars}")
    
    visualizer = ParametricVisualizer(config_path=params_path, save_dir=img_save_dir)
    # agent = VLMAgent(cfg['model'])
    # agent = VLMAgent(cfg['model'], window_size=window_size)
    if mode == "text_matrix_all":
        from src.vlm_agent_text import VLMAgentText
        agent = VLMAgentText(cfg['model'], window_size=window_size)
    else:
        from src.vlm_agent import VLMAgent
        agent = VLMAgent(cfg['model'], window_size=window_size)
    data_dir = Path(cfg['paths']['dataset'])
    if not data_dir.exists():
        if "atsad" in args.config.lower():
            data_dir = Path("data/dataset/ATSADBench")
        else:
            data_dir = Path("data/dataset/SKAB")
    # data_dir = Path(cfg['paths']['dataset'])
    
    for fname in cfg['experiment']['target_files']:
        print(f"\nProcessing File: {fname}")
        file_path = data_dir / fname
        if not file_path.exists(): 
            print("File not found.")
            continue

        clean_name = Path(fname).stem 
        json_save_dir = base_output_dir / "json_details" / clean_name
        json_save_dir.mkdir(parents=True, exist_ok=True)

        df = read_data_file(file_path)
        var_names = df.columns[:num_vars].tolist()

        file_results = []
        
        window_gen = get_window_generator(df, window_size, stride, num_vars)
        
        for win_info in tqdm(window_gen, desc="Analyzing", unit="win"):
            win_id = win_info["id"]
            print(win_id)
            start_idx = win_info["start"]
            end_idx = win_info["end"]
            window_data = win_info["data"]
            true_label_seg = win_info["labels"]
            
            json_filename = json_save_dir / f"vlm_reply_w_{win_id:05d}.json"
            
            if json_filename.exists():
                with open(json_filename, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                analysis_result = cached_data.get("full_response", {})
                latency = cached_data.get("latency_seconds", 0.0)
                print(f"  [Skip] Window {win_id} already processed.")
            else:
                if is_no_residual:
                    matrices_data = engine.get_test_matrices_only(window_data)
                else:
                    matrices_data = engine.process_single_window(window_data)
                if mode == "text_matrix_all":
                    fname_key = Path(fname).name
                    current_params = visualizer.params_map.get(fname_key, {k: {"t1": 1.0, "t2": 1.0} for k in visualizer.all_matrix_types})
                    
                    processed_matrices = {}
                    for m_key, m_val in matrices_data.items():
                        p = current_params.get(m_key, {"t1": 1.0, "t2": 1.0})
                        processed_matrices[m_key] = visualizer._soft_threshold(m_val, p["t1"], p["t2"])
                    analysis_result = agent.analyze(mode, processed_matrices, var_names)
                    prompt_filename = json_save_dir / f"prompt_w_{win_id:05d}.txt"
                    with open(prompt_filename, "w", encoding="utf-8") as f:
                        f.write(analysis_result.get("full_prompt", "No prompt found"))
                    prompt_text = analysis_result.pop("full_prompt", "") 
                else:
                    should_skip = (mode == "no_threshold_all") 
                    images_b64 = visualizer.generate_views(
                        fname, 
                        matrices_data, 
                        window_idx=win_id, 
                        var_names=var_names,
                        use_grayscale=is_no_residual,
                        skip_threshold=should_skip 
                    )
                    
                    analysis_result = agent.analyze(mode, images_b64)
                latency = analysis_result.get("latency", 0.0)
                # latency=0.0
                
                json_content = {
                    "window_id": win_id,
                    "filename": fname,
                    "mode": mode,
                    "latency_seconds": latency,
                    "full_response": analysis_result 
                }
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(json_content, f, ensure_ascii=False, indent=2)
            pred_labels = analysis_result.get("Label", [0]*64)
            is_padded = analysis_result.get("is_padded", False) 
            if is_padded:
                print(f"  [!] WARNING: Window {win_id} parsing failed or length mismatch. Results are padded/defaulted.")
            if len(pred_labels) > window_size:
                pred_labels = pred_labels[:window_size]
            elif len(pred_labels) < window_size:
                pred_labels = pred_labels + [0] * (window_size - len(pred_labels))
            
            is_anomaly = 1 if sum(pred_labels) > 0 else 0
            
            row_record = {
                "Window_ID": win_id,
                "Start_Time": start_idx,
                "End_Time": end_idx,
                "Ground_Truth_Window": 1 if sum(true_label_seg) > 0 else 0,
                "Pred_Label_Window": is_anomaly,
                "Root_Causes": ",".join(analysis_result.get("root_cause_variables", [])),
                "Raw_Pred_Array": str(pred_labels),
                "Latency": latency,
                "Is_Padded": is_padded  
            }
            file_results.append(row_record)

        result_df = pd.DataFrame(file_results)
        save_name = f"Result_{clean_name}_{mode}.xlsx"
        save_path = base_output_dir / save_name
        
        result_df.to_excel(save_path, index=False)
        print(f"  -> Saved summary to: {save_path}")

    print("\n>>> Task completed.")

if __name__ == "__main__":
    main()
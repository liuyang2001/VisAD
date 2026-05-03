import argparse
import yaml
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return precision, recall, f1, acc, tp, fp, fn, tn

def point_adjustment(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(int)
    pred_pa = pred.copy().astype(int)
    anomaly_state = False
    n = len(gt)
    for i in range(n):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0: break
                pred_pa[j] = 1
            for j in range(i, n):
                if gt[j] == 0: break
                pred_pa[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_pa[i] = 1
    precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(
        gt, pred_pa, average="binary", zero_division=0
    )
    accuracy_pa = accuracy_score(gt, pred_pa)
    return precision_pa, recall_pa, f1_pa, accuracy_pa

def compute_latency_contiguity_on_windows(win_gt: np.ndarray, win_pred: np.ndarray):
    win_gt = win_gt.astype(int)
    win_pred = win_pred.astype(int)
    idx_pos = np.where(win_gt == 1)[0]
    if len(idx_pos) == 0:
        return 0.0, 1.0 
    segments = []
    start = idx_pos[0]
    for i in range(1, len(idx_pos)):
        if idx_pos[i] != idx_pos[i-1] + 1:
            segments.append((start, idx_pos[i-1]))
            start = idx_pos[i]
    segments.append((start, idx_pos[-1]))

    latencies = []
    contigs = []
    for (s, e) in segments:
        seg_len = e - s + 1
        first_p = -1
        for j in range(s, len(win_pred)):
            if win_pred[j] == 1:
                first_p = j
                break
        latency = (first_p - s) if first_p != -1 else seg_len
        latencies.append(max(0, latency))
        hits = np.sum(win_pred[s:e+1])
        contigs.append(hits / seg_len)
    return np.mean(latencies), np.mean(contigs)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_original_data(file_path):
    suffix = file_path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(file_path, sep=None, engine='python')
    else:
        return pd.read_excel(file_path)

def reconstruct_series(result_df, original_df):
    gt_full = original_df['Label'].values.astype(int)
    pred_full = np.zeros_like(gt_full, dtype=int)
    for _, row in result_df.iterrows():
        s = int(row['Start_Time'])
        e = int(row['End_Time'])
        try:
            raw_array_str = row['Raw_Pred_Array']
            local_preds = ast.literal_eval(raw_array_str)
            length_to_fill = min(e, len(pred_full)) - s
            if length_to_fill > 0:
                pred_full[s : s+length_to_fill] = local_preds[:length_to_fill]
        except Exception as ex:
            print(f"Error parsing window at {s}: {ex}")
    return gt_full, pred_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--mode", type=str, default=None, help="Override config mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode if args.mode else cfg['experiment']['mode']
    
    data_dir = Path(cfg['paths']['dataset'])
    if not data_dir.exists():
        if "atsad" in args.config.lower():
            data_dir = Path("data/dataset/ATSADBench")
        else:
            data_dir = Path("data/dataset/SKAB")

    base_output_dir = Path(cfg['paths']['output_dir'])
    target_results_dir = base_output_dir / mode 
    
    if not target_results_dir.exists():
        print(f"Error: Results directory {target_results_dir} not found.")
        return

    print(f"\n" + "="*60)
    print(f"=== Evaluating Metrics for Mode: [{mode}] ===")
    print("="*60)
    
    result_files = sorted(list(target_results_dir.glob(f"Result_*_{mode}.xlsx")))
    if not result_files:
        print(f"No result files found in {target_results_dir}")
        return

    summary_list = []
    
    for res_file in result_files:
        fname_base = res_file.name.replace("Result_", "").replace(f"_{mode}.xlsx", "")
        potential_files = list(data_dir.glob(f"{fname_base}*"))
        if not potential_files:
            continue
        
        csv_path = potential_files[0]
        print(f"  -> Processing: {fname_base}")
        
        try:
            res_df = pd.read_excel(res_file)
            orig_df = read_original_data(csv_path)
            
            inf_time = res_df['Latency'].mean() if 'Latency' in res_df.columns else 0.0

            gt, pred = reconstruct_series(res_df, orig_df)
            
            # Point, PA, Window Metrics
            p, r, f1, acc, tp, fp, fn, tn = compute_binary_metrics(gt, pred)
            p_pa, r_pa, f1_pa, acc_pa = point_adjustment(gt, pred)
            
            WINDOW_SIZE = cfg['experiment']['window_size']
            win_gt, win_pred = [], []
            for i in range(0, len(gt), WINDOW_SIZE):
                chunk_gt = gt[i : i + WINDOW_SIZE]
                chunk_pred = pred[i : i + WINDOW_SIZE]
                if len(chunk_gt) == 0: continue
                win_gt.append(1 if np.any(chunk_gt == 1) else 0)
                win_pred.append(1 if np.any(chunk_pred == 1) else 0)
            
            pw, rw, f1w, accw, _, _, _, _ = compute_binary_metrics(np.array(win_gt), np.array(win_pred))
            avg_lat, avg_cont = compute_latency_contiguity_on_windows(np.array(win_gt), np.array(win_pred))
            
            row = {
                "Dataset": fname_base,
                "F1_point": f1, "P_point": p, "R_point": r, "Acc_point": acc,
                "F1_PA": f1_pa, "P_PA": p_pa, "R_PA": r_pa,
                "F1_win": f1w, "P_win": pw, "R_win": rw, "Acc_win": accw,
                "Avg_Latency_Win": avg_lat,
                "Avg_Contiguity_Win": avg_cont,
                "Avg_Inference_Latency": inf_time
            }
            summary_list.append(row)
            pd.DataFrame([row]).to_excel(res_file.parent / f"Metrics_{fname_base}.xlsx", index=False)
        except Exception as e:
            print(f"    [Error] {fname_base}: {e}")

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        
        extra_rows = []
        
        if "atsad" in args.config.lower():
            tasks = ["CDA", "FVA", "TVDA"]
            print("\n>>> Grouping ATSAD Tasks (Averaging IL & OL)...")
            for t_name in tasks:
                task_df = summary_df[summary_df['Dataset'].str.contains(t_name)]
                if not task_df.empty:
                    t_mean = task_df.mean(numeric_only=True).to_dict()
                    t_mean["Dataset"] = f"AVG_{t_name}"
                    extra_rows.append(t_mean)
        
        mean_all = summary_df.mean(numeric_only=True).to_dict()
        mean_all["Dataset"] = "TOTAL_AVERAGE"
        
        final_summary_df = pd.concat([
            summary_df, 
            pd.DataFrame(extra_rows), 
            pd.DataFrame([mean_all])
        ], ignore_index=True)
        
        numeric_cols = final_summary_df.select_dtypes(include=[np.number]).columns
        final_summary_df[numeric_cols] = final_summary_df[numeric_cols].round(4)
        
        summary_path = target_results_dir / f"SUMMARY_METRICS_{mode}.xlsx"
        final_summary_df.to_excel(summary_path, index=False)
        
        print(f"\nFinal summary for {mode.upper()} saved to: {summary_path}")
        for row in extra_rows:
            print(f"  -> {row['Dataset']}: F1_point={row['F1_point']:.4f}, F1_win={row['F1_win']:.4f}")

if __name__ == "__main__":
    main()
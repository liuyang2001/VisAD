import sys
import torch
import pandas as pd
import numpy as np
import argparse
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from src.cnn_baseline_net import ConvAutoEncoder
from src.feature_engine import FeatureEngine
from src.utils_io import read_data_file
from main import get_window_generator

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_percentile_threshold(model, train_lib, m_keys, device, p=95):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for item in tqdm(train_lib, desc="Generating Threshold"):
            stack = np.stack([item["mats"][k] for k in m_keys])
            inp = torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(device)
            
            out = model(inp)
            mse = torch.mean((inp - out)**2).item()
            errors.append(mse)
    
    threshold = np.percentile(errors, p)
    print(f">>> Established Normal Threshold ({p}th): {threshold:.8f}")
    return threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config_atsad.yaml")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    cfg = load_config(cfg_path)
    mode = "cnn_ablation"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device}")
    
    is_atsad = "atsad" in args.config.lower()
    data_dir = ROOT_DIR / ("data/dataset/ATSADBench" if is_atsad else "data/dataset/SKAB")
    model_file = f"cnn_{'atsad' if is_atsad else 'skab'}_v_final.pth"

    resource_path = ROOT_DIR / cfg['paths']['resources']
    with open(resource_path, "rb") as f:
        train_lib = pickle.load(f)["train_lib"]
    
    engine = FeatureEngine(resource_path)
    num_vars = cfg['experiment']['num_vars']
    
    model = ConvAutoEncoder(num_channels=6, matrix_size=num_vars).to(device)
    model_path = ROOT_DIR / "data" / model_file
    if not model_path.exists():
        print(f"[Error] CNN model not found. Run train_cnn_baseline.py first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))

    m_keys = ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]
    threshold = get_percentile_threshold(model, train_lib, m_keys, device, p=95)

    output_dir = ROOT_DIR / cfg['paths']['output_dir'] / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in cfg['experiment']['target_files']:
        print(f"\nCNN Processing: {fname}")
        file_path = data_dir / fname
        if not file_path.exists(): continue

        df = read_data_file(file_path)
        file_results = []
        window_gen = get_window_generator(df, cfg['experiment']['window_size'], cfg['experiment']['stride'], num_vars)
        
        model.eval()
        for win_info in tqdm(window_gen):
            matrices_data = engine.get_test_matrices_only(win_info["data"])
            stack = np.stack([matrices_data[k] for k in m_keys])
            
            test_input = torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                reconstructed = model(test_input)
                score = torch.mean((test_input - reconstructed)**2).item()
            
            is_anomaly = 1 if score > threshold else 0
            
            pred_labels = [is_anomaly] * cfg['experiment']['window_size']
            file_results.append({
                "Window_ID": win_info["id"],
                "Start_Time": win_info["start"],
                "End_Time": win_info["end"],
                "Ground_Truth_Window": 1 if sum(win_info["labels"]) > 0 else 0,
                "Pred_Label_Window": is_anomaly,
                "Raw_Pred_Array": str(pred_labels),
                "CNN_Score": score
            })

        result_df = pd.DataFrame(file_results)
        save_path = output_dir / f"Result_{Path(fname).stem}_{mode}.xlsx"
        result_df.to_excel(save_path, index=False)
        print(f"  -> Saved results to: {save_path}")

if __name__ == "__main__":
    main()
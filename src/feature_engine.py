import numpy as np
import pandas as pd
import pickle
from src.math_utils import get_6_matrices

class FeatureEngine:
    def __init__(self, resource_path):
        with open(resource_path, "rb") as f:
            data = pickle.load(f)
        
        self.train_lib = data["train_lib"]
        self.global_stats = data["global_stats"]
        self.mat_stats = data["mat_stats"]
        self.window_size = 64
        
    def _compute_and_norm(self, raw_window_data):

        w_norm = ((raw_window_data - self.global_stats["min"]) / self.global_stats["rng"]).T
        
        curr_mats_raw = get_6_matrices(w_norm)

        curr_mats = {}
        for k, v in curr_mats_raw.items():

            curr_mats[k] = (v - self.mat_stats[k]["min"]) / self.mat_stats[k]["rng"]
            
        return w_norm, curr_mats

    def process_single_window(self, raw_window_data):
        w_norm = ((raw_window_data - self.global_stats["min"]) / self.global_stats["rng"]).T
        
        curr_mats_raw = get_6_matrices(w_norm)
        curr_mats = {}
        for k, v in curr_mats_raw.items():
            curr_mats[k] = (v - self.mat_stats[k]["min"]) / self.mat_stats[k]["rng"]

        min_mse = float('inf')
        best_ref = None
        best_ref_idx = -1 

        for idx, lib_win in enumerate(self.train_lib): 
            mse = np.mean((w_norm - lib_win["raw_seq"])**2)
            if mse < min_mse:
                min_mse = mse
                best_ref = lib_win
                best_ref_idx = idx 

        if best_ref_idx != -1:
            print(f"  [Retrieval] Match Found: Reference Window ID = {best_ref_idx}")

        residuals = {}
        if best_ref is not None:
            for k in curr_mats:
                residuals[k] = curr_mats[k] - best_ref["mats"][k]
        else:
            residuals = curr_mats 
            
        return residuals
    
    def get_test_matrices_only(self, raw_window_data):

        _, curr_mats = self._compute_and_norm(raw_window_data)
        return curr_mats
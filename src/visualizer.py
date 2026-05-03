import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False     
class ParametricVisualizer:
    def __init__(self, config_path=None, save_dir=None):

        self.params_map = {}
        
        if config_path:
            self.config_path = Path(config_path)
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding="utf-8") as f:
                    self.params_map = json.load(f)
            else:
                print(f"[Visualizer] Warning: Config path {config_path} not found. Defaulting to empty params.")
        
        self.all_matrix_types = ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]
        
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _soft_threshold(self, matrix, t1, t2):
        if t1 >= 1.0 and t2 >= 1.0:
            return np.zeros_like(matrix)
        A = np.abs(matrix)
        sign = np.sign(matrix)
        out = np.zeros_like(matrix, dtype=float)
        denom = (t2 - t1) if (t2 - t1) > 1e-12 else 1e-12
        mid_mask = (A >= t1) & (A < t2)
        out[mid_mask] = (A[mid_mask] - t1) / denom
        high_mask = A >= t2
        out[high_mask] = 1.0
        return out * sign

    def _mat_to_b64_and_save(self, matrix, title, var_names=None, save_name=None, specific_dir=None, grayscale=False):
        n = matrix.shape[0]
        scale = 32
        H, W = n * scale, n * scale
        dpi, label_w, right_w, title_h, bottom_h = 100, 120, 60, 40, 120
        fig_w_px = W + label_w + right_w
        fig_h_px = H + title_h + bottom_h

        fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

        ax_title = fig.add_axes([0, (bottom_h + H) / fig_h_px, 1, title_h / fig_h_px])
        ax_title.axis("off")
        if title: ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=14, fontweight="bold")

        ax_left = fig.add_axes([0, bottom_h / fig_h_px, label_w / fig_w_px, H / fig_h_px])
        ax_left.set_xlim(0, 1); ax_left.set_ylim(n, 0); ax_left.axis("off")
        if var_names:
            for i, name in enumerate(var_names):
                ax_left.text(0.95, i + 0.5, name, ha="right", va="center", fontsize=6, fontweight="bold")

        ax_img = fig.add_axes([label_w / fig_w_px, bottom_h / fig_h_px, W / fig_w_px, H / fig_h_px])
        
        if grayscale:
            ax_img.imshow(matrix, vmin=0, vmax=1, cmap="gray", interpolation="nearest", aspect="auto", extent=[0, n, n, 0])
        else:
            ax_img.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest", aspect="auto", extent=[0, n, n, 0])
            
        ax_img.set_xticks([]); ax_img.set_yticks([]); ax_img.set_ylim(n, 0)

        ax_bottom = fig.add_axes([label_w / fig_w_px, 0, W / fig_w_px, bottom_h / fig_h_px])
        ax_bottom.set_xlim(0, n); ax_bottom.set_ylim(0, 1); ax_bottom.axis("off")
        if var_names:
            for j, name in enumerate(var_names):
                ax_bottom.text(j + 0.5, 0.5, name, ha="center", va="center", rotation=90, fontsize=6, fontweight="bold")

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi) 
        b64_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        
        target_dir = specific_dir if specific_dir else self.save_dir
        if target_dir and save_name:
            target_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_dir / save_name, format='png', dpi=dpi)
            
        plt.close(fig)
        return b64_str

    def generate_views(self, filename, raw_matrices, window_idx=None, var_names=None, use_grayscale=False,skip_threshold=False):
        fname_key = Path(filename).name 
        
        if use_grayscale:
            target_keys = self.all_matrix_types
            current_params = {}
        else:
            if fname_key in self.params_map:
                current_params = self.params_map[fname_key]
                target_keys = list(current_params.keys())
            else:
                target_keys = self.all_matrix_types
                current_params = {k: {"t1": 1.0, "t2": 1.0} for k in target_keys}

        if self.save_dir:
            clean_dir_name = fname_key.replace(".csv", "")
            current_file_img_dir = self.save_dir / clean_dir_name
        else:
            current_file_img_dir = None

        outputs = {}
        for m_key in target_keys:
            if m_key not in raw_matrices: continue

            if use_grayscale:
                processed = raw_matrices[m_key]
            elif skip_threshold: 
                processed = raw_matrices[m_key]
            else:
                p = current_params.get(m_key, {"t1": 1.0, "t2": 1.0})
                processed = self._soft_threshold(raw_matrices[m_key], p["t1"], p["t2"])
            
            plot_title = m_key 
            if window_idx is not None:
                save_name = f"w_{window_idx:03d}_{m_key}_test.png"
                plot_title = f"w_{window_idx:03d}_{m_key}"
            else:
                save_name = None

            b64_str = self._mat_to_b64_and_save(
                processed, 
                title=plot_title, 
                var_names=var_names, 
                save_name=save_name,
                specific_dir=current_file_img_dir,
                grayscale=use_grayscale 
            )
            outputs[m_key] = b64_str
            
        return outputs
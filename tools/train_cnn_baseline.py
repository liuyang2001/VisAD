import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
from src.cnn_baseline_net import ConvAutoEncoder

def train_cnn(dataset_name="atsad"):
    lib_path = ROOT_DIR / f"data/resources_{dataset_name}/train_ref_lib.pkl"
    matrix_size = 27 if dataset_name == "atsad" else 8
    model_save_path = ROOT_DIR / f"data/cnn_{dataset_name}_v_final.pth"

    with open(lib_path, "rb") as f:
        train_lib = pickle.load(f)["train_lib"]

    m_keys = ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]
    inputs = [np.stack([item["mats"][k] for k in m_keys]) for item in train_lib]
    train_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = train_tensor.to(device)

    model = ConvAutoEncoder(num_channels=6, matrix_size=matrix_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5)
    
    criterion = nn.MSELoss()

    model.train()
    print(f"\n>>> Deep Training CNN for {dataset_name} on {device}...")
    
    best_loss = float('inf')
    last_lr = optimizer.param_groups[0]['lr']

    for epoch in range(20000):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"  [LR Update] Epoch {epoch+1}: LR decreased to {current_lr:.6f}")
            last_lr = current_lr
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/20000], Loss: {loss.item():.8f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_save_path)
            
        if loss.item() < 1e-9:
            print(">>> Reached convergence threshold. Stopping early.")
            break

    print(f">>> Training Complete. Best Loss: {best_loss:.8f}")

if __name__ == "__main__":
    train_cnn("atsad")
    train_cnn("skab")
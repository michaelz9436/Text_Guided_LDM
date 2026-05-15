import os
import sys
import yaml
import glob
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 确保能正常 import 模块
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)

from lldm.regressor import NumAtomsRegressor

# ----------------------------
# 1. 自定义 Dataset 与内存加载逻辑
# ----------------------------
class Zinc9MRegDataset(Dataset):
    def __init__(self, Zx, Zh, n):
        self.Zx = Zx
        self.Zh = Zh
        self.n = n

    def __len__(self):
        return len(self.n)

    def __getitem__(self, idx):
        return {
            'zx': self.Zx[idx],
            'zh': self.Zh[idx],
            'n': self.n[idx]
        }

def load_chunked_dataset_to_ram(latent_dir, atoms_dir, max_chunks=30):
    """
    搜索并成对加载指定数量的 Chunk，转换为内存中连续的 Tensor。
    通过 max_chunks 参数限制只加载前 30 个 chunk。
    """
    latent_files = sorted(glob.glob(os.path.join(latent_dir, "chunk_*.pt")))
    atoms_files = sorted(glob.glob(os.path.join(atoms_dir, "chunk_*_atoms.pt")))
    
    if len(latent_files) != len(atoms_files):
        print(f"[Warning] Latent chunks ({len(latent_files)}) 和 Atom chunks ({len(atoms_files)}) 数量不一致！")
    
    # 匹配最小数量，确保成对
    num_files = min(len(latent_files), len(atoms_files))
    
    # 限制只读取前 max_chunks 个 chunk
    if num_files > max_chunks:
        print(f"[INFO] 找到了 {num_files} 对 chunks，但根据设置只加载前 {max_chunks} 对。")
        num_files = max_chunks
        
    latent_files = latent_files[:num_files]
    atoms_files = atoms_files[:num_files]
    
    all_Zx, all_Zh, all_n = [], [], []
    
    print(f"[INFO] Loading {num_files} chunks to RAM...")
    for lf, af in tqdm(zip(latent_files, atoms_files), total=num_files, desc="Parsing chunks"):
        ld = torch.load(lf, map_location='cpu')
        ad = torch.load(af, map_location='cpu')
        
        # 提取并堆叠
        Zx_chunk = torch.stack([item['Zx'] for item in ld])
        Zh_chunk = torch.stack([item['Zh'] for item in ld])
        n_chunk = torch.tensor([item['n'] for item in ad], dtype=torch.float32)
        
        all_Zx.append(Zx_chunk)
        all_Zh.append(Zh_chunk)
        all_n.append(n_chunk)
        
    print("[INFO] Concatenating tensors...")
    Zx_tensor = torch.cat(all_Zx, dim=0)
    Zh_tensor = torch.cat(all_Zh, dim=0)
    n_tensor = torch.cat(all_n, dim=0)
    
    print(f"[INFO] Dataset loaded! Total samples: {len(n_tensor)}")
    return Zinc9MRegDataset(Zx_tensor, Zh_tensor, n_tensor)


def get_dataloaders(config):
    # 这里传入 max_chunks=30
    dataset = load_chunked_dataset_to_ram(config['latent_dir'], config['atoms_dir'], max_chunks=30)
    
    val_size = int(len(dataset) * config.get('val_split', 0.05))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 纯内存张量，num_workers=0 速度最快
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    return train_loader, val_loader

# ----------------------------
# 2. 训练与评估流程
# ----------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct_exact, correct_tolerance_1, correct_tolerance_2 = 0, 0, 0
    total_samples = 0
    criterion = nn.L1Loss(reduction='sum')
    
    with torch.no_grad():
        for batch in val_loader:
            zx = batch['zx'].to(device)
            zh = batch['zh'].to(device)
            target_n = batch['n'].to(device)
            
            pred_n = model(zx, zh).squeeze()
            
            total_loss += criterion(pred_n, target_n).item()
            pred_rounded = torch.round(pred_n)
            diff = torch.abs(pred_rounded - target_n)
            
            correct_exact += (diff == 0).sum().item()
            correct_tolerance_1 += (diff <= 1).sum().item()
            correct_tolerance_2 += (diff <= 2).sum().item()
            total_samples += target_n.size(0)
            
    avg_mae = total_loss / total_samples
    acc_exact = correct_exact / total_samples
    acc_tol_1 = correct_tolerance_1 / total_samples
    acc_tol_2 = correct_tolerance_2 / total_samples
    
    return avg_mae, acc_exact, acc_tol_1, acc_tol_2

def main():
    config_path = os.path.join(BASE_DIR, 'train/train_scripts/configs/zinc9m_regressor_config.yaml')
    config = load_config(config_path)
    device = torch.device(config.get('device', 'cpu'))
    
    exp_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 60)
    print(f" Experiment: {config['exp_name']}")
    print(f" Saving to: {exp_dir}")
    print("=" * 60)
    
    train_loader, val_loader = get_dataloaders(config)
    
    print("[INFO] Initializing Regressor Model...")
    model = NumAtomsRegressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.L1Loss() 
    
    best_mae = float('inf')
    last_save_path = os.path.join(exp_dir, 'last.pt')
    best_save_path = os.path.join(exp_dir, 'best.pt')
    
    print("[INFO] Starting Training...")
    for epoch in range(config['n_epochs']):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            zx = batch['zx'].to(device)
            zh = batch['zh'].to(device)
            target_n = batch['n'].to(device)
            
            optimizer.zero_grad()
            pred_n = model(zx, zh).squeeze()
            
            loss = criterion(pred_n, target_n)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % config['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['n_epochs']}] Batch[{batch_idx+1}/{len(train_loader)}] Loss (MAE): {loss.item():.4f}")
                
        # 跑完一个 Epoch 进行验证
        avg_train_loss = total_train_loss / len(train_loader)
        print("\n[INFO] Evaluating on validation set...")
        val_mae, acc_exact, acc_tol_1, acc_tol_2 = evaluate(model, val_loader, device)
        
        print(f"==> Epoch {epoch+1} completed.")
        print(f"    Train MAE: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"    Val Acc (Exact): {acc_exact*100:.2f}% | Val Acc (+-1): {acc_tol_1*100:.2f}% | Val Acc (+-2): {acc_tol_2*100:.2f}%")
        
        # 保存断点
        state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config': config,
            'val_mae': val_mae
        }
        
        torch.save(state, last_save_path)
        print(f"[Saved] Last Checkpoint updated at {last_save_path}")
        
        # 额外保存最佳模型
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(state, best_save_path)
            print(f"    [Saved] *** New Best Checkpoint at {best_save_path} ***")
        print("-" * 60)

    print("[INFO] Training Finished!")

if __name__ == '__main__':
    main()
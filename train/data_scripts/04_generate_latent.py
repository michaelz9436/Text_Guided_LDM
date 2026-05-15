import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)

from model.train_loop import TrainLoop, center_pos
from config.config import load_config
from utils.dataset import ZincLMDBDataset

MAP_ATOM_TYPE_ONLY_TO_INDEX = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
CHUNK_SIZE = 81920  
BATCH_SIZE = 5120   
OUTPUT_DIR = os.path.join(BASE_DIR, "train/data/zinc9m_latent/") 
lmdb_path = os.path.join(BASE_DIR, "train/data/zinc9m_subset/ZINC_9M.lmdb")
indices_path = os.path.join(BASE_DIR, "train/data/zinc9m_subset/dataset_indices.npy")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed_idx": -1, "chunk_count": 0}

def save_progress(idx, chunk_count):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"last_processed_idx": idx, "chunk_count": chunk_count}, f)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Model...")
    config_path = 'config/config.yaml'
    ckpt_path = 'checkpoints/vae.ckpt'
    
    config = load_config(config_path)
    model = TrainLoop(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.to(device)
    model.eval()

    print("Loading Training Dataset...")
    dataset = ZincLMDBDataset(lmdb_path, indices_path, split='train')
    
    progress = load_progress()
    start_idx = progress["last_processed_idx"] + 1
    chunk_count = progress["chunk_count"]
    
    if start_idx >= len(dataset):
        print("All molecules already processed.")
        return

    print(f"Resuming from index: {start_idx}, Current Chunk Count: {chunk_count}")

    subset_indices = list(range(start_idx, len(dataset)))
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    loader = DataLoader(
        subset_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    latent_buffer = []
    current_idx_in_total = start_idx
    error_batches = []

    pbar = tqdm(total=len(dataset), initial=start_idx, desc="Encoding Latent")
    
    K = model.cfg['encoder_config']['ligand_v_dim']


    for batch in loader:
        num_mols_in_batch = batch.batch.max().item() + 1
        batch_mol_ids = dataset.ids[current_idx_in_total : current_idx_in_total + num_mols_in_batch]
        
        try:
            batch = batch.to(device)
            h_idx = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX.get(a.item(), 0) for a in batch.h], device=device)
            one_hot_h = F.one_hot(h_idx, K).float()
            x_centered, _ = center_pos(batch.x, batch.batch, mode=True)

            with torch.no_grad():
                Zh, Zx, global_batch, _, _ = model.encode(one_hot_h, x_centered, batch.batch, deterministic=True)

            Zh = Zh.view(num_mols_in_batch, 10, -1)
            Zx = Zx.view(num_mols_in_batch, 10, -1)

            # 正常收集
            for i in range(num_mols_in_batch):
                latent_buffer.append({
                    'id': batch_mol_ids[i],
                    'Zh': Zh[i].cpu(),  
                    'Zx': Zx[i].cpu()  
                })
            # ===============================================

        except Exception as e:
            print(f"\n[Error] Batch starting at {current_idx_in_total} failed: {e}")
            for i in range(num_mols_in_batch):
                latent_buffer.append({
                    'id': batch_mol_ids[i],
                    'Zh': torch.zeros(10, config['optimal_layer_config']['latent_dim']), 
                    'Zx': torch.zeros(10, 3) 
                })
            error_batches.append((current_idx_in_total, str(e)))

        current_idx_in_total += num_mols_in_batch
        pbar.update(num_mols_in_batch)

        if len(latent_buffer) >= CHUNK_SIZE:
            chunk_count += 1
            chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
            
            torch.save(latent_buffer[:CHUNK_SIZE], chunk_filename)
            
            save_progress(current_idx_in_total - (len(latent_buffer) - CHUNK_SIZE) - 1, chunk_count)
            
            latent_buffer = latent_buffer[CHUNK_SIZE:]


    if len(latent_buffer) > 0:
        chunk_count += 1
        chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
        torch.save(latent_buffer, chunk_filename)
        save_progress(len(dataset) - 1, chunk_count)

    pbar.close()
    print(f"Encoding finished. Total chunks: {chunk_count}")

if __name__ == "__main__":
    main()
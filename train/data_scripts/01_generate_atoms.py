import os
import sys
import json
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)

from utils.dataset import ZincLMDBDataset

CHUNK_SIZE = 81920 
BATCH_SIZE = 5120    
OUTPUT_DIR = os.path.join(BASE_DIR, "train/data/zinc9m_atoms/") 
lmdb_path = os.path.join(BASE_DIR, "train/data/zinc9m_subset/ZINC_9M.lmdb")
indices_path = os.path.join(BASE_DIR, "train/data/zinc9m_subset/dataset_indices.npy")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "metadata_atoms.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed_idx": -1, "chunk_count": 0}

def save_progress(idx, chunk_count):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"last_processed_idx": idx, "chunk_count": chunk_count}, f)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Training Dataset for Atom Counts Extraction...")

    # initialize the full dataset 
    dataset = ZincLMDBDataset(lmdb_path, indices_path, split='train')
    
    progress = load_progress()
    start_idx = progress["last_processed_idx"] + 1
    chunk_count = progress["chunk_count"]
    
    if start_idx >= len(dataset):
        print("All molecules' atom counts already extracted.")
        return

    print(f"Resuming from index: {start_idx}, Current Chunk Count: {chunk_count}")

    subset_indices = list(range(start_idx, len(dataset)))
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    loader = DataLoader(
        subset_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=False
    )

    atom_buffer =[]
    current_idx_in_total = start_idx

    pbar = tqdm(total=len(dataset), initial=start_idx, desc="Extracting Atom Counts")
    
    for batch in loader:
        num_mols_in_batch = batch.batch.max().item() + 1
        batch_mol_ids = dataset.ids[current_idx_in_total : current_idx_in_total + num_mols_in_batch]
        
        try:

            if hasattr(batch, 'ptr') and batch.ptr is not None:
                atom_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
            else:
                atom_counts = batch.batch.bincount().tolist()

            for i in range(num_mols_in_batch):
                atom_buffer.append({
                    'id': batch_mol_ids[i],
                    'n': float(atom_counts[i])  # to ensure JSON serializability
                })

        except Exception as e:
            print(f"\n[Warning] CPU Processing batch failed at {current_idx_in_total}: {e}")

            for i in range(num_mols_in_batch):
                atom_buffer.append({
                    'id': batch_mol_ids[i],
                    'n': 0.0
                })

        current_idx_in_total += num_mols_in_batch
        pbar.update(num_mols_in_batch)

        # full chunk ready, save to disk
        if len(atom_buffer) >= CHUNK_SIZE:
            chunk_count += 1
            chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}_atoms.pt")
            
            torch.save(atom_buffer[:CHUNK_SIZE], chunk_filename)
            
            save_progress(current_idx_in_total - (len(atom_buffer) - CHUNK_SIZE) - 1, chunk_count)
            
            # 移除已保存部分
            atom_buffer = atom_buffer[CHUNK_SIZE:]

    if len(atom_buffer) > 0:
        chunk_count += 1
        chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}_atoms.pt")
        torch.save(atom_buffer, chunk_filename)
        save_progress(len(dataset) - 1, chunk_count)

    pbar.close()
    print(f"Extraction finished. Total atom chunks: {chunk_count}")

if __name__ == "__main__":
    main()
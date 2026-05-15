import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

from utils.dataset import ZincLMDBDataset
from utils.build_mol import MoleculeBuilder
import utils.mol_descriptor_rich as desc_engine


class Config:
    LMDB_PATH = os.path.join(BASE_DIR, "train/data/zinc9m_subset/ZINC_9M.lmdb")
    INDICES_PATH = os.path.join(BASE_DIR, "train/data/zinc9m_subset/dataset_indices.npy")
    OUTPUT_DIR = os.path.join(BASE_DIR, "train/data/zinc9m_desc/")
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "metadata.json")


    CHUNK_SIZE = 81920
    N_VARIANTS = 10            
    SEED = 42       

    # language richness controls how detailed and varied the descriptions are. 
    LANGUAGE_RICHNESS = 0.8
    GRANULARITY = 0.6
    PROB_EXACT_STRUCTURAL_COUNT = 0.25


def load_progress():
    if os.path.exists(Config.PROGRESS_FILE):
        with open(Config.PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed_idx": -1, "chunk_count": 0}

def save_progress(idx, chunk_count):
    with open(Config.PROGRESS_FILE, 'w') as f:
        json.dump({"last_processed_idx": idx, "chunk_count": chunk_count}, f)

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("Initializing Dataset...")
    dataset = ZincLMDBDataset(Config.LMDB_PATH, Config.INDICES_PATH, split='train')
    
    progress = load_progress()
    start_idx = progress["last_processed_idx"] + 1
    chunk_count = progress["chunk_count"]
    
    if start_idx >= len(dataset):
        print("All molecules already processed.")
        return

    print(f"Resuming from index: {start_idx}, Chunk Count: {chunk_count}")

    builder = MoleculeBuilder()
    
    desc_engine.Config.N_VARIANTS = Config.N_VARIANTS
    desc_engine.Config.LANGUAGE_RICHNESS = Config.LANGUAGE_RICHNESS
    desc_engine.Config.GRANULARITY = Config.GRANULARITY
    desc_engine.Config.PROB_EXACT_STRUCTURAL_COUNT = Config.PROB_EXACT_STRUCTURAL_COUNT

    desc_buffer = []
    
    pbar = tqdm(total=len(dataset), initial=start_idx, desc="Generating Descriptions")

    for i in range(start_idx, len(dataset)):
        try:
            data = dataset[i]
            mol_id = dataset.ids[i]
            
            coords = data.x.numpy()
            atoms = data.h.numpy().tolist()
            mol = builder.build_mol(coords, atoms)
            
            if mol is not None:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    Chem.FastFindRings(mol)
                    Chem.SanitizeMol(mol, Chem.SANITIZE_SYMMRINGS | Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_SETCONJUGATION | Chem.SANITIZE_SETHYBRIDIZATION)
                except Exception as e:
                    print(f"\n[Warning] Sanitization failed for mol at index {i}: {e}")
                    Chem.FastFindRings(mol)
                # ══════════════════════════════════════════
                prompts = desc_engine.generate_prompt_variants(
                    mol, 
                    n_variants=Config.N_VARIANTS, 
                    seed=Config.SEED + i
                )
            else:
                prompts = ["A molecular structure."] * Config.N_VARIANTS
                print(f"\n[Warning] Failed to build mol at index {i}")

            desc_buffer.append({
                'id': mol_id,
                'prompts': prompts
            })

            if len(desc_buffer) >= Config.CHUNK_SIZE:
                chunk_count += 1
                chunk_filename = os.path.join(Config.OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
                
                torch.save(desc_buffer, chunk_filename)
                
                save_progress(i, chunk_count)
                desc_buffer = []

            pbar.update(1)

        except Exception as e:
            print(f"\n[Error] Critical error at index {i}: {e}")
            desc_buffer.append({
                'id': dataset.ids[i],
                'prompts': ["Error generating description."] * Config.N_VARIANTS
            })
            pbar.update(1)
            continue

    if desc_buffer:
        chunk_count += 1
        chunk_filename = os.path.join(Config.OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
        torch.save(desc_buffer, chunk_filename)
        save_progress(len(dataset) - 1, chunk_count)

    pbar.close()
    print(f"Description generation finished. Total chunks: {chunk_count}")

if __name__ == "__main__":
    main()
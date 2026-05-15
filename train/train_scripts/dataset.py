import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

_NUM_ZINC_NODES = 10  

class ZincLatentDataset(Dataset):
    def __init__(self, data_dir, text_emb_dir, max_chunks=30):
        super().__init__()
        
        all_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.pt")))
        self.chunk_files =[f for f in all_files if not f.endswith("_atoms.pt")]
        
        if len(self.chunk_files) == 0:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
            
        if max_chunks is not None:
            self.chunk_files = self.chunk_files[:max_chunks]

        self.all_zx = []
        self.all_zh = []
        self.all_n =[]  
        self.all_text_emb = []

        print(f"Loading data from {len(self.chunk_files)} chunks...")

        for file_path in tqdm(self.chunk_files, desc="Loading Chunks"):
            mof_list = torch.load(file_path, map_location='cpu', weights_only=False)
            
            chunk_basename = os.path.basename(file_path)
            emb_file_path = os.path.join(text_emb_dir, f"emb_{chunk_basename}")
            if not os.path.exists(emb_file_path):
                raise FileNotFoundError(f"Missing corresponding text emb file: {emb_file_path}")
            
            emb_list = torch.load(emb_file_path, map_location='cpu', weights_only=False)

            Zx_chunk = torch.stack([item['Zx'].float() for item in mof_list])
            Zh_chunk = torch.stack([item['Zh'].float() for item in mof_list])
            
            Emb_chunk = torch.stack([item['embeddings'].float() for item in emb_list])

            centroid = Zx_chunk.mean(dim=1, keepdim=True)
            Zx_centered = Zx_chunk - centroid

            self.all_zx.append(Zx_centered)
            self.all_zh.append(Zh_chunk)
            self.all_n.append(torch.zeros(len(mof_list)))
            self.all_text_emb.append(Emb_chunk) # 


        self.all_zx = torch.cat(self.all_zx, dim=0)
        self.all_zh = torch.cat(self.all_zh, dim=0)
        self.all_n = torch.cat(self.all_n, dim=0)
        self.all_text_emb = torch.cat(self.all_text_emb, dim=0)
        print(f"Text Emb Shape: {self.all_text_emb.shape}")
        print(f"Dataset Loaded! Total Molecules: {self.all_zx.shape[0]}")
        print(f"Shapes: Zx {self.all_zx.shape}, Zh {self.all_zh.shape}")

    def __len__(self):
        return self.all_zx.shape[0]

    def __getitem__(self, idx):
        embs = self.all_text_emb[idx]
        
        rand_emb_idx = torch.randint(0, embs.size(0), (1,)).item()
        selected_emb = embs[rand_emb_idx] #  (384,)

        return {
            'zx': self.all_zx[idx],
            'zh': self.all_zh[idx],
            'n': self.all_n[idx],
            'text_emb': selected_emb
        }


def get_zinc_train_val_dataloaders(data_dir, text_emb_dir, batch_size, val_split=0.1, num_workers=0, max_chunks=None):

    dataset = ZincLatentDataset(data_dir=data_dir, text_emb_dir=text_emb_dir, max_chunks=max_chunks)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    raise RuntimeError("This dataset is meant to be used as part of the training pipeline. Please run the full training script instead.")
    _zinc_dir = "you should not use this"
    _ds = ZincLatentDataset(data_dir=_zinc_dir, max_chunks=2) 
    print("ZincLatentDataset __len__:", len(_ds))
    _sample = _ds[0]
    _zx, _zh, _n = _sample["zx"], _sample["zh"], _sample["n"]
    assert _zx.shape == (10, 3), f"Wrong Zx shape: {_zx.shape}"
    assert _zh.shape == (10, 32), f"Wrong Zh shape: {_zh.shape}"
    print("Data verification passed! Ready for actual EGNN supervision.")
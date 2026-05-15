import os
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# 元素符号到原子序数的映射表
SYMBOL_TO_ATOMIC_NUM = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'As': 33, 'Se': 34, 'Br': 35, 'I': 53
}

class ZincLMDBDataset(Dataset):
    def __init__(self, lmdb_path, indices_path, split='train'):
        super().__init__()
        self.lmdb_path = lmdb_path
        if not os.path.exists(indices_path):
            raise FileNotFoundError(f"Indices file not found: {indices_path}")
            
        indices_dict = np.load(indices_path, allow_pickle=True).item()
        self.ids = indices_dict[f'{split}_indices']
        self.env = None 

    def _connect_db(self):
        self.env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.env is None:
            self._connect_db()
        
        mol_id = str(self.ids[idx]).encode()
        
        with self.env.begin() as txn:
            raw_data = txn.get(mol_id)
        
        if raw_data is None:
            return self.__getitem__((idx + 1) % len(self))
            
        data_dict = pickle.loads(raw_data)
        
        # --- 处理原子 (atoms) ---
        atoms_raw = data_dict['atoms']
        
        # 如果是字符串列表 ['C', 'N', ...], 转换为原子序数 [6, 7, ...]
        if isinstance(atoms_raw[0], str):
            atomic_numbers = [SYMBOL_TO_ATOMIC_NUM.get(s, 6) for s in atoms_raw]
        else:
            atomic_numbers = atoms_raw
            
        h = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # --- 处理坐标 (coordinates) ---
        # 确保是 float32 并转为 (N, 3)
        coords_raw = data_dict['coordinates']
        pos = torch.tensor(coords_raw, dtype=torch.float32).view(-1, 3)
        
        # 简单校验：原子数和坐标数是否匹配
        if h.shape[0] != pos.shape[0]:
            # 如果不匹配，可能是数据损坏，尝试取下一个
            return self.__getitem__((idx + 1) % len(self))

        return Data(h=h, x=pos)
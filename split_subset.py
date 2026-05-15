import lmdb
import numpy as np
import os
from tqdm import tqdm


orig_lmdb_file = '/home/course_project/MolFLAE/Latent_Experiments/data/zinc9m/ZINC_9M.lmdb'
orig_idx_path = '/home/course_project/MolFLAE/Latent_Experiments/data/zinc9m/dataset_indices.npy'

new_dir = os.path.expanduser('~/text_guided_LDM/train/data/zinc9m_subset/')
os.makedirs(new_dir, exist_ok=True)

new_lmdb_file = os.path.join(new_dir, 'ZINC_9M.lmdb')
new_idx_path = os.path.join(new_dir, 'dataset_indices.npy')

orig_data = np.load(orig_idx_path, allow_pickle=True).item()
orig_train_indices = orig_data['train_indices'][:3000]


env_in = lmdb.open(orig_lmdb_file, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
env_out = lmdb.open(new_lmdb_file, map_size=1099511627776, subdir=False) 

print(f'Starting physical copy of {len(orig_train_indices)} molecules to {new_lmdb_file}...')

with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
    for new_idx, old_idx in enumerate(tqdm(orig_train_indices)):
        key_in = str(old_idx).encode('ascii')
        key_out = str(new_idx).encode('ascii')
        
        value = txn_in.get(key_in)
        if value is not None:
            txn_out.put(key_out, value)

env_in.close()
env_out.close()


new_indices_dict = {
    'train_indices': np.arange(0, 900).tolist(),
    'val_indices': np.arange(900, 1000).tolist(),
    'test_indices': []
}
np.save(new_idx_path, new_indices_dict)

print('\nSuccess! Physical copy complete.')
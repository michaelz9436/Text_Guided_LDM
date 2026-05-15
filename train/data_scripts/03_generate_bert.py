import os
import glob
import json
import torch
from sentence_transformers import SentenceTransformer

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)


class Config:
    INPUT_DIR = os.path.join(BASE_DIR, "train/data/zinc9m_desc/")
    OUTPUT_DIR = os.path.join(BASE_DIR, "train/data/zinc9m_BERT_emb/")
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "emb_metadata.json")

    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    
    MAX_CHUNKS_TO_PROCESS = 30  
    N_VARIANTS = 10      
    
    BATCH_SIZE = 4096           


def load_progress():
    if os.path.exists(Config.PROGRESS_FILE):
        with open(Config.PROGRESS_FILE, 'r') as f:
            return json.load(f).get("processed_chunks", [])
    return []

def save_progress(processed_chunks):
    with open(Config.PROGRESS_FILE, 'w') as f:
        json.dump({"processed_chunks": processed_chunks}, f)

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    input_chunks = sorted(glob.glob(os.path.join(Config.INPUT_DIR, "chunk_*.pt")))
    
    if not input_chunks:
        print(f"❌ No chunk files found in {Config.INPUT_DIR}")
        return

    target_chunks = input_chunks[:Config.MAX_CHUNKS_TO_PROCESS]
    print(f"🔍 Found {len(input_chunks)} chunks, targeting first {len(target_chunks)} chunks.")

    processed_chunks = load_progress()
    print(f"✅ Already processed {len(processed_chunks)} chunks.")

    print(f"\n🚀 Loading SentenceTransformer model: {Config.MODEL_NAME} ...")
    model = SentenceTransformer(Config.MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Model loaded on device: {device}")

    for chunk_path in target_chunks:
        chunk_filename = os.path.basename(chunk_path)
        
        if chunk_filename in processed_chunks:
            print(f"⏭️  Skipping {chunk_filename} (Already processed)")
            continue

        print(f"\n🔄 Processing {chunk_filename} ...")
        
        output_filename = f"emb_{chunk_filename}"
        output_path = os.path.join(Config.OUTPUT_DIR, output_filename)

        data = torch.load(chunk_path)
        num_mols = len(data)
        print(f"   -> Loaded {num_mols} molecules.")

        all_prompts = []
        for item in data:
            if len(item['prompts']) != Config.N_VARIANTS:
                raise ValueError(f"Molecule {item['id']} has {len(item['prompts'])} prompts, expected {Config.N_VARIANTS}!")
            all_prompts.extend(item['prompts'])
        
        total_sentences = len(all_prompts)
        print(f"   -> Encoding {total_sentences} sentences in batches of {Config.BATCH_SIZE}...")

        with torch.no_grad():
            embeddings = model.encode(
                all_prompts,
                batch_size=Config.BATCH_SIZE,
                convert_to_tensor=True,    
                show_progress_bar=True      
            )
        
        embeddings = embeddings.cpu()
        emb_dim = embeddings.shape[-1]
        
        emb_reshaped = embeddings.view(num_mols, Config.N_VARIANTS, emb_dim)

        out_data = []
        for i, item in enumerate(data):
            out_data.append({
                'id': item['id'],
                'embeddings': emb_reshaped[i].clone() 
            })

        torch.save(out_data, output_path)
        print(f"💾 Saved embeddings to {output_filename} (Shape per mol: {Config.N_VARIANTS}x{emb_dim})")

        processed_chunks.append(chunk_filename)
        save_progress(processed_chunks)

    print("\n🎉 All target chunks processed successfully!")

if __name__ == "__main__":
    main()
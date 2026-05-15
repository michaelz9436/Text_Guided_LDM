import os
import sys
import argparse
import yaml
import torch
import copy
from torch.nn.utils import clip_grad_norm_

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_DIR)

from dataset import get_zinc_train_val_dataloaders
from lldm.diffusion import LinkerLatentDiffusion
from equivariant_diffusion.utils import EMA

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, model_ema, optimizer, epoch, config, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'ema_state': model_ema.state_dict() if model_ema is not None else None,
        'optimizer_state': optimizer.state_dict(),
        'config': config
    }
    torch.save(checkpoint, save_path)
    print(f"[INFO] Checkpoint saved to {save_path}")

def load_checkpoint(resume_path, model, model_ema, optimizer, device):
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if model_ema is not None and checkpoint['ema_state'] is not None:
        model_ema.load_state_dict(checkpoint['ema_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser(description='Train LLDM on ZINC latents')
    parser.add_argument('--config', type=str, default=os.path.join(BASE_DIR, 'train/train_scripts/configs/lldm_config.yaml'))
    args = parser.parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(os.getcwd(), config_path))
    config = load_config(config_path)
    device = torch.device(config.get('device', 'cpu'))
    
    exp_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[INFO] Experiment directory: {exp_dir}")

    # load data
    print("[INFO] Loading Dataset...")
    dataloader, _val_loader = get_zinc_train_val_dataloaders(
        data_dir=config['data_dir'],
        text_emb_dir=config['text_emb_dir'],   # text_emb_dir 
        batch_size=config['batch_size'],
        val_split=config.get('val_split', 0.1),
        num_workers=config.get('num_workers', 0),
        max_chunks=config.get('max_chunks', None) 
    )
    

    print("[INFO] Initializing Model...")
    model = LinkerLatentDiffusion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-12)
    
    # initialize EMA model
    if config['ema_decay'] > 0:
        model_ema = copy.deepcopy(model)
        model_ema.eval()
        for param in model_ema.parameters():
            param.requires_grad = False
        ema_helper = EMA(config['ema_decay'])
    else:
        model_ema = None
        ema_helper = None

    # resume from checkpoint if specified
    start_epoch = 0
    if config.get('resume_checkpoint') is not None:
        start_epoch = load_checkpoint(
            config['resume_checkpoint'], model, model_ema, optimizer, device
        ) + 1

    # train loos
    print("[INFO] Starting Training...")
    for epoch in range(start_epoch, config['n_epochs']):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            zx = batch['zx'].to(device)
            zh = batch['zh'].to(device)
            
            # get text embedding and move to device
            text_emb = batch['text_emb'].to(device)
            
            optimizer.zero_grad()
            loss = model(zx, zh, condition=text_emb)
            loss.backward()
            
            if config['clip_grad']:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            total_loss += loss.item()
            
            # update EMA model
            if ema_helper is not None:
                ema_helper.update_model_average(model_ema, model)
            
            if (batch_idx + 1) % config['log_interval'] == 0:
                print(f"Epoch [{epoch}/{config['n_epochs']}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # save checkpoint every epoch
        save_checkpoint(model, model_ema, optimizer, epoch, config, os.path.join(exp_dir, 'last.pt'))
        
        # if epoch % config['save_interval'] == 0, also save a separate checkpoint
        if epoch > 0 and epoch % config['save_interval'] == 0:
            save_checkpoint(model, model_ema, optimizer, epoch, config, os.path.join(exp_dir, f'epoch_{epoch}.pt'))

    print("[INFO] Training Finished!")

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # info aggregation network
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # node update network
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, dists):
        """
        h: [B, N, hidden_dim]
        dists: [B, N, N, 1]
        """
        B, N, _ = h.size()
        
        # [B, N, N, hidden_dim]
        h_i = h.unsqueeze(2).expand(B, N, N, -1)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)
        
        # [B, N, N, hidden_dim]
        msg_inputs = torch.cat([h_i, h_j, dists], dim=-1)
        messages = self.message_mlp(msg_inputs)
        

        agg_messages = messages.mean(dim=2)  # [B, N, hidden_dim]
        
        # residual update
        update_inputs = torch.cat([h, agg_messages], dim=-1)
        h_new = h + self.update_mlp(update_inputs)
        
        return h_new

class NumAtomsRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        latent_dim = config['latent_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        

        self.node_emb = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN 
        self.layers = nn.ModuleList([DistanceGNNLayer(hidden_dim) for _ in range(num_layers)])
        
        # readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) # scalar output for regression
        )

    def forward(self, zx, zh):
        # zx: [B, 10, 3], zh: [B, 10, 32]
        
        # euclidean distances
        diff = zx.unsqueeze(2) - zx.unsqueeze(1)  # [B, 10, 10, 3]
        dists = torch.norm(diff, dim=-1, keepdim=True) # [B, 10, 10, 1]
        

        h = self.node_emb(zh)
        
        for layer in self.layers:
            h = layer(h, dists)
            
        # global pooling
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1)[0]
        h_global = torch.cat([h_mean, h_max], dim=-1)
        
        pred_n = self.readout(h_global).squeeze(-1)
        return pred_n
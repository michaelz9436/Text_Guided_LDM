import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from egnn.models import EGNN_dynamics_QM9
from equivariant_diffusion.en_diffusion import PredefinedNoiseSchedule
from equivariant_diffusion.utils import remove_mean, sample_center_gravity_zero_gaussian

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

class LinkerLatentDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_nodes = config['num_nodes']    
        self.n_dims = config['spatial_dim']   
        self.in_node_nf = config['latent_dim']
        
        self.norm_x = 4.0
        self.norm_h = 1.0
        
        self.T = config['diffusion_steps']
        self.gamma = PredefinedNoiseSchedule(
            noise_schedule=config['diffusion_noise_schedule'], 
            timesteps=self.T, 
            precision=config['diffusion_noise_precision']
        )
        
        self.dynamics = EGNN_dynamics_QM9(
            in_node_nf=self.in_node_nf + 1,
            context_node_nf=0,
            n_dims=self.n_dims, 
            hidden_nf=config['nf'], 
            device=config.get('device', 'cpu'),
            n_layers=config['n_layers'], 
            attention=config['attention'], 
            condition_time=True, 
            tanh=config['tanh'], 
            mode=config['model'], 
            norm_constant=config['norm_constant'],
            sin_embedding=config['sin_embedding'], 
            aggregation_method=config['aggregation_method'],
            condition_dim=config.get('text_emb_dim', 384)  
        )

    def forward(self, zx, zh, condition=None):
        B, N, _ = zx.size()
        device = zx.device
        
        # set condition to zero with a certain probability during training for classifier-free guidance
        if condition is not None and self.training:
            uncond_prob = self.config.get('uncond_prob', 0.15)
            if np.random.rand() < uncond_prob:
                condition = torch.zeros_like(condition)
        
        zx = zx / self.norm_x
        zh = zh / self.norm_h
        
        t_int = torch.randint(1, self.T + 1, size=(B, 1), device=device).float()
        t = t_int / self.T
        
        gamma_t = self.gamma(t).view(B, 1, 1)
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        
        eps_x = sample_center_gravity_zero_gaussian(size=(B, N, self.n_dims), device=device)
        eps_h = torch.randn(size=(B, N, self.in_node_nf), device=device)
        
        zx_t = alpha_t * zx + sigma_t * eps_x
        zh_t = alpha_t * zh + sigma_t * eps_h
        zx_t = remove_mean(zx_t)
        
        xh_t = torch.cat([zx_t, zh_t], dim=-1)
        node_mask = torch.ones(B, N, 1, device=device)
        edge_mask = (1 - torch.eye(N, device=device)).unsqueeze(0).repeat(B, 1, 1).view(-1, 1)
        
        # 传入 condition
        pred_eps = self.dynamics._forward(t, xh_t, node_mask, edge_mask, condition=condition)
        
        pred_eps_x = pred_eps[:, :, :self.n_dims]
        pred_eps_h = pred_eps[:, :, self.n_dims:]
        
        loss_x = sum_except_batch((eps_x - pred_eps_x) ** 2) / (N * self.n_dims)
        loss_h = sum_except_batch((eps_h - pred_eps_h) ** 2) / (N * self.in_node_nf)
        
        loss = (loss_x + loss_h).mean()
        return loss
    
    @torch.no_grad()
    def sample(self, n_samples, condition=None):
        device = next(self.parameters()).device
        
        # make sure condition has the right batch size if given
        if condition is not None:
            n_samples = condition.size(0)

        # if condition is None, create a zero condition 
        # this makes sure that the model can still be used for unconditional generation
        if condition is None and self.config.get('text_emb_dim', 0) > 0:
            condition = torch.zeros(n_samples, self.config['text_emb_dim'], device=device)
            
            
        zx = sample_center_gravity_zero_gaussian(size=(n_samples, self.n_nodes, self.n_dims), device=device)
        zh = torch.randn(size=(n_samples, self.n_nodes, self.in_node_nf), device=device)
        
        node_mask = torch.ones(n_samples, self.n_nodes, 1, device=device)
        edge_mask = (1 - torch.eye(self.n_nodes, device=device)).unsqueeze(0).repeat(n_samples, 1, 1).view(-1, 1)

        w = self.config.get('guidance_scale', 3.0)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device).float() / self.T
            t_array = torch.full((n_samples, 1), fill_value=s + 1, device=device).float() / self.T
            
            gamma_s = self.gamma(s_array).view(n_samples, 1, 1)
            gamma_t = self.gamma(t_array).view(n_samples, 1, 1)
            
            log_alpha2_t = F.logsigmoid(-gamma_t)
            log_alpha2_s = F.logsigmoid(-gamma_s)
            alpha_t_given_s = torch.exp(0.5 * (log_alpha2_t - log_alpha2_s))
            sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma_s = torch.sqrt(torch.sigmoid(gamma_s))
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            
            xh_t = torch.cat([zx, zh], dim=-1)
            
            # cfg sampling
            if condition is not None:
                # conditioned and unconditioned predictions
                xh_t_double = torch.cat([xh_t, xh_t], dim=0)
                t_double = torch.cat([t_array, t_array], dim=0)
                mask_double = torch.cat([node_mask, node_mask], dim=0)
                edge_double = torch.cat([edge_mask, edge_mask], dim=0)
                
                cond_double = torch.cat([condition, torch.zeros_like(condition)], dim=0)
                
                pred_eps_double = self.dynamics._forward(t_double, xh_t_double, mask_double, edge_double, condition=cond_double)
                
                pred_eps_cond, pred_eps_uncond = pred_eps_double.chunk(2, dim=0)
                
                # CFG: e_uncond + w * (e_cond - e_uncond)
                pred_eps = pred_eps_uncond + w * (pred_eps_cond - pred_eps_uncond)
            else:
                pred_eps = self.dynamics._forward(t_array, xh_t, node_mask, edge_mask, condition=None)
            
            pred_eps_x = pred_eps[:, :, :self.n_dims]
            pred_eps_h = pred_eps[:, :, self.n_dims:]
            
            mu_x = zx / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * pred_eps_x
            mu_h = zh / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * pred_eps_h
            
            sigma = (sigma_t_given_s * sigma_s) / sigma_t
            
            if s > 0:
                noise_x = sample_center_gravity_zero_gaussian(size=(n_samples, self.n_nodes, self.n_dims), device=device)
                noise_h = torch.randn(size=(n_samples, self.n_nodes, self.in_node_nf), device=device)
            else:
                noise_x = torch.zeros_like(zx)
                noise_h = torch.zeros_like(zh)
            
            zx = mu_x + sigma * noise_x
            zh = mu_h + sigma * noise_h
            zx = remove_mean(zx)

        zx = zx * self.norm_x
        zh = zh * self.norm_h
        
        return zx, zh
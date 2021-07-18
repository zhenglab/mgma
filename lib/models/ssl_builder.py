# Implementation for SSL Models.
# 
# Code partially referenced from:
# https://github.com/PatrickHua/SimSiam
# https://github.com/facebookresearch/simsiam

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from math import pi, cos

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def D(p, z, version='original'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()
    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class SimSiam(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=cfg.SSL.DIM_IN, hidden_dim=cfg.SSL.DIM_IN, out_dim=cfg.SSL.DIM_IN)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(in_dim=cfg.SSL.DIM_IN, hidden_dim=int(cfg.SSL.DIM_IN / 4), out_dim=cfg.SSL.DIM_IN)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, progress=None):
        x1, x2 = x[0], x[1]
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L

class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()

        self.backbone = backbone
        self.projector = MLP(in_dim=cfg.SSL.DIM_IN, hidden_dim=cfg.SSL.DIM_IN * 2, out_dim=cfg.SSL.DIM_IN // 2)
        self.online_encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(in_dim=cfg.SSL.DIM_IN // 2, hidden_dim=cfg.SSL.DIM_IN * 2, out_dim=cfg.SSL.DIM_IN // 2)
#         raise NotImplementedError('Please put update_moving_average to training')
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def target_ema(self, k, K, base_ema=4e-3):
        # tau_base = 0.996 
        return 1 - base_ema * (cos(pi*k/K)+1)/2 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x, progress=None):
        x1, x2 = x[0], x[1]
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t      = self.target_encoder

        z1_o = f_o(x1)
        z2_o = f_o(x2)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            if progress is not None and len(progress) == 2:
                cur_global_iter, max_iters = progress[0], progress[1]
            else:
                cur_global_iter, max_iters = 0, 1
            self.update_moving_average(cur_global_iter, max_iters)
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2 
        return L

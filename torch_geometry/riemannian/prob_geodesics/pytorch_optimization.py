#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch import vmap
from torch.func import grad

from torch import Tensor
from typing import Callable, Dict, Tuple
from abc import ABC

from torch_geometry.riemannian.manifolds import RiemannianManifold
from torch_geometry.line_search import Backtracking

#%% Objective Function

class GeoCurve(torch.nn.Module):
    def __init__(self, 
                 zt:Tensor,
                 )->None:
        super(GeoCurve, self).__init__()
        
        self.zt = torch.nn.Parameter(zt, requires_grad=True)
        
        return
    
    def forward(self, 
                )->Tensor:
        
        return self.zt
    
#%% Minimization Problem

class TorchOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 lr_rate:float=0.01,
                 optimizer:Callable=torch.optim.Adam,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        self.optimizer = optimizer
        
        self.lr_rate = lr_rate
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     T+1,
                                                                     dtype=z0.dtype)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    @torch.no_grad()
    def score_norm2(self, 
                    zt:Tensor,
                    )->Tensor:

        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return score_norm2
    
    @torch.no_grad()
    def energy(self, 
               zt:Tensor,
               )->Tensor:
        
        zt = torch.vstack((self.z0, zt, self.zT))
        Gt = vmap(self.M.G)(zt[:-1])
        dzt = zt[1:]-zt[:-1]
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', dzt, Gt, dzt))

    def reg_energy(self, 
                   zt:Tensor,
                   )->Tensor:
        
        zt = torch.vstack((self.z0, zt, self.zT))
        Gt = vmap(self.M.G)(zt[:-1])
        dzt = zt[1:]-zt[:-1]
        
        energy = torch.sum(torch.einsum('...i,...ij,...j->...', dzt, Gt, dzt))
        
        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return energy + self.lam_norm*score_norm2
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 )->Tensor:
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        
        self.G0 = self.M.G(z0).detach()
        
        zt = self.init_fun(z0,zT,self.T)

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        model = GeoCurve(zt=zt)
        optim = self.optimizer(model.parameters(), lr=self.lr_rate)
        loss_fn = self.reg_energy
        
        model.train()
        for i in range(0, self.max_iter):
            zt = model.forward()
            loss = loss_fn(zt)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
        zt = model.forward()
        reg_energy = self.reg_energy(zt).item()
        zt = torch.vstack((z0, zt, zT)).detach()
            
        return zt, reg_energy, None, self.max_iter

#%% Probabilistic GEORCE for Euclidean Background Metric

class TorchEuclideanOptimization(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 lr_rate:float=0.01,
                 optimizer:Callable=torch.optim.Adam,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:

        self.score_fun = score_fun
        self.optimizer = optimizer
        
        self.lr_rate = lr_rate
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     T+1,
                                                                     dtype=z0.dtype)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def score_norm2(self, 
                    zt:Tensor,
                    )->Tensor:
        
        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)

        return score_norm2

    def energy(self, 
               zt:Tensor,
               )->Tensor:

        zt = torch.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return torch.sum(ut*ut)
    
    def reg_energy(self, 
                   zt:Tensor,
                   )->Tensor:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 )->Tensor:
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        zt = self.init_fun(z0,zT,self.T)

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        model = GeoCurve(zt=zt)
        optim = self.optimizer(model.parameters(), lr=self.lr_rate)
        loss_fn = self.reg_energy
        
        model.train()
        for idx in range(0, self.max_iter):
            zt = model.forward()
            loss = loss_fn(zt)
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(idx)
            print(loss.item())
            
        zt = model.forward()
        reg_energy = self.reg_energy(zt).item()
        zt = torch.vstack((z0, zt, zT)).detach()
            
        return zt, reg_energy, None, self.max_iter
        
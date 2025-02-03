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

#%% Gradient Descent Estimation of Geodesics

class ProbSGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable[[Tensor, Tensor, int], Tensor]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
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
                   *args,
                   )->Tensor:
        
        zt = torch.vstack((self.z0, zt, self.zT))
        Gt = vmap(self.M.G)(zt[:-1])
        dzt = zt[1:]-zt[:-1]
        
        energy = torch.sum(torch.einsum('...i,...ij,...j->...', dzt, Gt, dzt))
        
        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return energy + self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Tensor,
                   *args,
                   )->Tensor:
        
        return (grad(self.reg_energy)(zt,*args)).detach()
    
    @torch.no_grad()
    def Dregenergy_fast(self,
                        zt:Tensor,
                        ut:Tensor,
                        Gt:Tensor,
                        gt:Tensor,
                        )->Tensor:
        
        denergy = gt+2.*(torch.einsum('tij,tj->ti', Gt[:-1], ut[:-1])-torch.einsum('tij,tj->ti', Gt[1:], ut[1:]))

        return denergy
    
    def inner_product(self,
                      zt:Tensor,
                      ut:Tensor,
                      )->Tensor:
        
        Gt = vmap(self.M.G)(zt)
        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ut, Gt, ut))+self.lam_norm*score_norm2, Gt.detach()
    
    def gt(self,
           zt:Tensor,
           ut:Tensor,
           )->Tuple[Tensor]:
        
        gt, Gt = grad(self.inner_product, has_aux=True)(zt, ut[1:])
        Gt = torch.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                           Gt))
        
        return gt.detach(), Gt.detach()
    
    @torch.no_grad()
    def update_scheme(self, 
                      gt:Tensor, 
                      gt_inv:Tensor,
                      )->Tensor:

        g_cumsum = torch.flip(torch.cumsum(torch.flip(gt, dims=[0]), dim=0), dims=[0])
        ginv_sum = torch.sum(gt_inv, dim=0)
        
        rhs = torch.sum(torch.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), dim=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        
        muT = -torch.linalg.solve(ginv_sum, rhs)
        mut = torch.vstack((muT+g_cumsum, muT))
        
        return mut
    
    @torch.no_grad()
    def update_xt(self,
                  zt:Tensor,
                  alpha:Tensor,
                  ut_hat:Tensor,
                  ut:Tensor,
                  )->Tensor:
        
        return self.z0+torch.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], dim=0)
    
    def cond_fun(self, 
                 grad_norm:Tensor,
                 idx:int,
                 )->Tensor:

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    zt:Tensor,
                    ut:Tensor,
                    Gt:Tensor,
                    gt:Tensor,
                    gt_inv:Tensor,
                    grad_norm:Tensor,
                    idx:int,
                    )->Tensor:

        mut = self.update_scheme(gt, gt_inv)

        ut_hat = -0.5*torch.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+torch.cumsum(ut[:-1], dim=0)
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gt[1:])))
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(zt,ut, Gt, gt)).item()
        
        idx += 1
            
        return zt, ut, Gt, gt, gt_inv, grad_norm, idx
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 step:str="while",
                 )->Tensor:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Dregenergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        self.diff = zT-z0
        self.dim = len(z0)
        
        self.G0 = self.M.G(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        
        zt = self.init_fun(z0,zT,self.T)
        ut = torch.ones((self.T, self.dim), dtype=z0.dtype)*self.diff/self.T

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gt[1:])))
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(zt,ut, Gt, gt)).item()
        if step == "while":
            idx = 0
            while self.cond_fun(grad_norm, idx):
                zt, ut, Gt, gt, gt_inv, grad_norm, idx = self.georce_step(zt, 
                                                                          ut, 
                                                                          Gt, 
                                                                          gt, 
                                                                          gt_inv, 
                                                                          grad_norm, 
                                                                          idx,
                                                                          )
        elif step == "for":
            for idx in range(self.max_iter):
                zt, ut, Gt, gt, gt_inv, grad_norm, idx = self.georce_step(zt, 
                                                                          ut, 
                                                                          Gt, 
                                                                          gt, 
                                                                          gt_inv, 
                                                                          grad_norm, 
                                                                          idx,
                                                                          )
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        reg_energy = self.reg_energy(zt).item()
        zt = torch.vstack((z0, zt, zT)).detach()
            
        return zt, reg_energy, grad_norm, idx

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbEuclideanSGEORCE(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Tensor, Tensor, int], Tensor]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 n_samples:int=10,
                 sigma:float=1.0,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:

        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        self.n_samples = n_samples
        self.sigma = sigma
        
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
                   *args,
                   )->Tensor:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Tensor,
                   *args,
                   )->Tensor:
        
        return grad(self.reg_energy)(zt,*args).detach()
    
    def Dregenergy_fast(self,
                        ut:Tensor,
                        gt:Tensor,
                        )->Tensor:
        
        return gt+2.*(ut[:-1]-ut[1:])

    def inner_product(self,
                      zt:Tensor,
                      eps:Tensor
                      )->Tensor:

        ztilde = zt + self.sigma*eps
        score = self.score_fun(ztilde)
        score_norm2 = torch.sum(score**2)
        
        return self.lam_norm*score_norm2
    
    def gt(self,
           zt:Tensor,
           )->Tensor:
        
        eps = torch.randn_like(zt)
        gt = grad(self.inner_product, argnums=0)(zt, eps)

        return gt.detach()
        
    @torch.no_grad()
    def update_xt(self,
                  zt:Tensor,
                  alpha:Tensor,
                  ut_hat:Tensor,
                  ut:Tensor,
                  )->Tensor:
        
        return self.z0+torch.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], dim=0)
    
    @torch.no_grad()
    def update_ut(self,
                  gt:Tensor,
                  )->Tensor:
        
        g_cumsum = torch.vstack((torch.flip(torch.cumsum(torch.flip(gt, dims=[0]), dim=0), dims=[0]), torch.zeros(self.dim)))
        g_sum = torch.sum(g_cumsum, dim=0)/self.T
        
        return self.diff/self.T+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 grad_norm:Tensor,
                 idx:int,
                 )->Tensor:

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                   zt:Tensor,
                   ut:Tensor,
                   gt:Tensor,
                   grad_norm:Tensor,
                   idx:int,
                   )->Tensor:

        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+torch.cumsum(ut[:-1], dim=0)
        
        print(self.reg_energy(zt).item())
        
        gt = torch.mean(torch.stack([self.gt(zt) for _ in range(self.n_samples)]), dim=0)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(ut, gt)).item()
        
        return zt, ut, gt, grad_norm, idx+1
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 step:str="while",
                 )->Tensor:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Dregenergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        self.diff = zT-z0
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        ut = torch.ones((self.T, self.dim), dtype=z0.dtype, requires_grad=False)*self.diff/self.T

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*energy_init/score_norm2_init
        print(self.reg_energy(zt).item())
        gt = torch.mean(torch.stack([self.gt(zt) for _ in range(self.n_samples)]), dim=0)#self.gt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(ut, gt)).item()
        if step == "while":
            idx = 0
            print(idx)
            while self.cond_fun(grad_norm, idx):
                zt, ut, gt, grad_norm, idx = self.georce_step(zt, ut, gt, grad_norm, idx)
                print(idx)
        elif step == "for":
            for idx in range(self.max_iter):
                zt, ut, gt, grad_norm, idx = self.georce_step(zt, ut, gt, grad_norm, idx)
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        reg_energy = self.reg_energy(zt).item()
        
        zt = torch.vstack((z0, zt, zT)).detach()
            
        return zt, reg_energy, grad_norm, idx
        
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

class ProbAdaGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable[[Tensor, Tensor, int], Tensor]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
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

class ProbEuclideanAdaGEORCE(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Tensor, Tensor, int], Tensor]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 )->None:

        self.score_fun = score_fun
        
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
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
                   *args,
                   )->Tensor:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy_fast(self,
                        ut:Tensor,
                        gt:Tensor,
                        )->Tensor:
        
        return gt+2.*(ut[:-1]-ut[1:])

    def inner_product(self,
                      zt:Tensor,
                      )->Tensor:

        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return self.lam_norm*score_norm2
    
    def sgt(self,
           zt:Tensor,
           )->Tensor:
        
        with torch.no_grad():
            self.model.zt = torch.nn.Parameter(zt)
        
        zt = self.model.forward()
        loss = self.inner_product(zt)
        loss.backward()
        gt = self.model.zt.grad
        
        #grad_fun = grad(self.inner_product)
        #gt = torch.stack([grad_fun(z).detach() for z in zt])

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
    
    @torch.no_grad()
    def adaptive_default(self,
                         sgt_k1:Tensor,
                         sgt_k2:Tensor,
                         rg_k1:Tensor,
                         rg_k2:Tensor,
                         idx:int,
                         )->Tuple:

        sgt_k2 = (1.-self.beta1)*sgt_k2+self.beta1*sgt_k1
        rg_k2 = (1.-self.beta2)*rg_k2+self.beta2*rg_k1
        
        beta1 = self.beta1**(idx+2)
        beta2 = self.beta2**(idx+2)

        sgt_hat = sgt_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(torch.sqrt(1+vt)+self.eps)
        
        kappa = torch.min(torch.Tensor([lr, 1.0])).item()
        
        return sgt_k2, sgt_hat, rg_k2, beta1, beta2, kappa, idx

    @torch.no_grad()
    def cond_fun(self, 
                 grad_norm:Tensor,
                 idx:int,
                 )->Tensor:

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    @torch.no_grad()
    def georce_step(self,
                    zt:Tensor,
                    ut:Tensor,
                    sgt:Tensor,
                    kappa:float,
                    )->Tuple:
        
        ut_hat = self.update_ut(sgt)
        zt_hat = self.z0+torch.cumsum(ut_hat[:-1], dim=0)
        
        return zt+kappa*(zt_hat-zt), ut+kappa*(ut_hat-ut)
    
    def adaptive_step(self,
                      zt:Tensor,
                      ut:Tensor,
                      sgt_k1:Tensor, 
                      sgt_hat:Tensor,
                      rg_k1:Tensor,
                      grad_norm:Tensor,
                      kappa:float,
                      idx:int,
                      )->Tensor:
        
        zt, ut = self.georce_step(zt,
                                  ut,
                                  sgt_hat,
                                  kappa,
                                  )
        sgt_k2 = self.sgt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        rg_k2 = torch.sum(sgt_k2**2)
        
        sgt_k2, sgt_hat, rg_k2, beta1, beta2, kappa, idx = self.adaptive_default(sgt_k1, 
                                                                                 sgt_k2, 
                                                                                 rg_k1, 
                                                                                 rg_k2, 
                                                                                 idx,
                                                                                 )
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(ut, sgt_hat)).item()
        
        return zt, ut, sgt_k2, sgt_hat, rg_k2, grad_norm, kappa, idx+1
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 step:str="while",
                 )->Tensor:
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        self.diff = zT-z0
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        ut = torch.ones((self.T, self.dim), dtype=z0.dtype, requires_grad=False)*self.diff/self.T
        self.model = GeoCurve(zt)

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*energy_init/score_norm2_init
        print(self.reg_energy(zt).item())
        sgt = self.sgt(zt)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        rg = torch.sum(sgt**2)
        grad_norm = torch.linalg.norm(self.Dregenergy_fast(ut, sgt)).item()
        if step == "while":
            idx = 0
            print(idx)
            while self.cond_fun(grad_norm, idx):
                zt, ut, sgt, sgt_hat, rg, grad_norm, kappa, idx = self.adaptive_step(zt, 
                                                                                     ut, 
                                                                                     sgt, 
                                                                                     sgt, 
                                                                                     rg, 
                                                                                     grad_norm, 
                                                                                     self.lr_rate, 
                                                                                     idx,
                                                                                     )
                print(self.reg_energy(zt).item())
                print(idx)
        elif step == "for":
            for _ in range(self.max_iter):
                zt, ut, sgt, sgt_hat, rg, grad_norm, kappa, idx = self.adaptive_step(zt, 
                                                                                     ut, 
                                                                                     sgt,
                                                                                     sgt,
                                                                                     rg,
                                                                                     grad_norm, 
                                                                                     self.lr_rate, 
                                                                                     idx,
                                                                                     )
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        reg_energy = self.reg_energy(zt).item()
        zt = torch.vstack((z0, zt, zT)).detach()
            
        return zt, reg_energy, grad_norm, idx
        
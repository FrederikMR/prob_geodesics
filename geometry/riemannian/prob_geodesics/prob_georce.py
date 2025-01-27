#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.riemannian.manifolds import RiemannianManifold
from geometry.line_search import Backtracking

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
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
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:
        
        def energy_path(energy:Array,
                        y:Tuple,
                        )->Tuple[Array]:

            z, dz = y
            
            G = self.M.G(z)

            energy += jnp.einsum('i,ij,j->', dz, G, dz)
            
            return (energy,)*2

        term1 = zt[0]-self.z0
        energy_init = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        zt = jnp.vstack((zt, self.zT))
        
        energy, _ = lax.scan(energy_path,
                             init=energy_init,
                             xs=(zt[:-1], zt[1:]-zt[:-1]),
                             )
        
        return energy
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:
        
        def reg_energy_path(reg_energy:Array,
                            y:Tuple,
                            )->Tuple[Array]:

            z, dz = y
            
            G = self.M.G(z)
            score = self.score_fun(z)

            reg_energy += jnp.einsum('i,ij,j->', dz, G, dz) + self.lam_norm*jnp.sum(score**2)
            
            return (reg_energy,)*2

        term1 = zt[0]-self.z0
        reg_energy_init = jnp.einsum('i,ij,j->', term1, self.G0, term1) + self.score_norm20
        zt = jnp.vstack((zt, self.zT))
        
        reg_energy, _ = lax.scan(reg_energy_path,
                                 init=reg_energy_init,
                                 xs=(zt[:-1], zt[1:]-zt[:-1]),
                                 )
        
        return reg_energy
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def Dregenergy_fast(self,
                        zt:Array,
                        ut:Array,
                        Gt:Array,
                        gt:Array,
                        )->Array:
        
        denergy = gt+2.*(jnp.einsum('tij,tj->ti', Gt[:-1], ut[:-1])-jnp.einsum('tij,tj->ti', Gt[1:], ut[1:]))

        return denergy
    
    def inner_product(self,
                      z:Array,
                      u:Array,
                      )->Array:
        
        G = self.M.G(z)
        score = self.score_fun(z)
        score_norm2 = jnp.sum(score**2)
        
        return jnp.einsum('i,ij,j->', u, G, u)+self.lam_norm*score_norm2, lax.stop_gradient(G)
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Tuple[Array]:
        
        def inner_product(carry:Tuple[Array],
                          y:Tuple[Array],
                          )->Tuple[Array]:
            
            g, G = carry
            z, u = y
            
            g, G = lax.stop_gradient(grad(self.inner_product, has_aux=True)(z, u))
            
            return ((g, G),)*2
        
        _, (gt, Gt) = lax.scan(inner_product,
                               xs=(zt,ut[1:]),
                               init=(jnp.zeros(self.M.dim, dtype=zt.dtype), 
                                     jnp.zeros((self.M.dim,self.M.dim), dtype=zt.dtype)),
                               )

        Gt = jnp.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                         Gt))
        gt = lax.stop_gradient(gt)
        Gt = lax.stop_gradient(Gt)
        
        return gt, Gt
    
    def update_scheme(self, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], axis=0)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, Gt, gt, gt_inv, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, Gt, gt, gt_inv, grad_norm, idx = carry
        
        mut = self.update_scheme(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gt[1:])))
        grad_norm = jnp.linalg.norm(self.Dregenergy_fast(zt,ut, Gt, gt).reshape(-1))
        
        return (zt, ut, Gt, gt, gt_inv, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut = carry
        
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, vmap(lambda z: self.M.Ginv(z))(zt)))
        
        mut = self.update_scheme(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        return ((zt, ut),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Dregenergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0
        self.zT = zT
        self.diff = zT-z0
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.M.G(z0))
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        ut = jnp.ones((self.T, self.dim), dtype=z0.dtype)*self.diff/self.T

        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        if step == "while":
            gt, Gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gt_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gt[1:])))
            grad_norm = jnp.linalg.norm(self.Dregenergy_fast(zt,ut, Gt, gt).reshape(-1))
            
            zt, ut, Gt, gt, gt_inv, grad_norm, idx = lax.while_loop(self.cond_fun, 
                                                                    self.while_step, 
                                                                    init_val=(zt, 
                                                                              ut, 
                                                                              Gt, 
                                                                              gt, 
                                                                              gt_inv, 
                                                                              grad_norm, 
                                                                              0),
                                                                    )
            reg_energy = self.reg_energy(zt)
            
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut),
                              xs=jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            reg_energy = vmap(self.reg_energy)(zt)
            grad_norm = jnp.linalg.norm(vmap(self.Dregenergy)(zt).reshape(len(zt),-1), axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, reg_energy, grad_norm, idx

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbEuclideanGEORCE(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:

        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:

        zt = jnp.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return jnp.sum(ut*ut)
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def Dregenergy_fast(self,
                        ut:Array,
                        gt:Array,
                        )->Array:
        
        return gt+2.*(ut[:-1]-ut[1:])

    def inner_product(self,
                      z:Array,
                      )->Array:

        score = self.score_fun(z)
        score_norm2 = jnp.sum(score**2)
        
        return self.lam_norm*score_norm2
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Tuple[Array]:
        
        def inner_product(g:Array,
                          z:Array,
                          )->Tuple[Array]:
            
            g = lax.stop_gradient(grad(self.inner_product)(z))
            
            return (g,)*2
        
        _, gt = lax.scan(inner_product,
                         xs=zt,
                         init=jnp.zeros(self.dim, dtype=zt.dtype),
                         )

        gt = lax.stop_gradient(gt)
        
        return gt
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], axis=0)
    
    def update_ut(self,
                  gt:Array,
                  )->Array:
        
        g_cumsum = jnp.vstack((jnp.cumsum(gt[::-1], axis=0)[::-1], jnp.zeros(self.dim)))
        g_sum = jnp.sum(g_cumsum, axis=0)/self.T
        
        return self.diff/self.T+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, gt, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, gt, grad_norm, idx = carry
        
        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = jnp.linalg.norm(self.Dregenergy_fast(ut, gt).reshape(-1))
        
        return (zt, ut, gt, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut = carry
        
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        return ((zt, ut),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Dregenergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0
        self.zT = zT
        self.diff = zT-z0
        self.dim = len(z0)
        
        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        ut = jnp.ones((self.T, self.dim), dtype=z0.dtype)*self.diff/self.T

        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        if step == "while":
            gt = self.gt(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            grad_norm = jnp.linalg.norm(self.Dregenergy_fast(ut, gt).reshape(-1))
            
            zt, ut, gt, grad_norm, idx = lax.while_loop(self.cond_fun, 
                                                        self.while_step, 
                                                        init_val=(zt, 
                                                                  ut, 
                                                                  gt, 
                                                                  grad_norm, 
                                                                  0),
                                                        )
            reg_energy = self.reg_energy(zt)
            
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut),
                              xs=jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            reg_energy = vmap(self.reg_energy)(zt)
            grad_norm = jnp.linalg.norm(vmap(self.Dregenergy)(zt).reshape(len(zt),-1), axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, reg_energy, grad_norm, idx
        
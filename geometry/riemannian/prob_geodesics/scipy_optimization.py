#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.riemannian.manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class ScipyOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.M = M
        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        zt = zt.reshape(-1, self.dim)
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:
        
        zt = zt.reshape(-1, self.dim)

        zt = jnp.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return jnp.sum(jnp.einsum('ti,ti->t', ut, ut))
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:
        
        zt = zt.reshape(-1,self.dim)

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def HessRegEnergy(self,
                      zt:Array,
                      )->Array:
        
        return hessian(self.reg_energy)(zt)
    
    def HessPRegEnergy(self,
                       zt:Array,
                       p:Array,
                       )->Array:
        
        return jnp.dot(hessian(self.reg_energy)(zt), p)
    
    def callback(self,
                 zt:Array
                 )->Array:
        
        self.save_zt.append(zt.reshape(-1, self.dim))
        
        return
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)

        self.G0 = lax.stop_gradient(self.M.G(z0))
        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        
        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        if step == "while":
            res = minimize(fun = self.reg_energy, 
                           x0=zt.reshape(-1), 
                           method=self.method, 
                           jac=self.Dregenergy,
                           hess=self.HessRegEnergy,
                           hessp=self.HessPRegEnergy,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
        
            zt = res.x.reshape(-1,self.dim)
            reg_energy = self.reg_energy(zt)
            zt = jnp.vstack((z0, zt, zT))
            grad_norm =  jnp.linalg.norm(res.jac.reshape(-1,self.dim))
            idx = res.nit
        elif step == "for":
            res = minimize(fun = self.reg_energy,
                           x0=zt.reshape(-1),
                           method=self.method,
                           jac=self.Dregenergy,
                           hess=self.HessRegEnergy,
                           hessp=self.HessPRegEnergy,
                           callback=self.callback,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
            
            zt = jnp.stack([zt.reshape(-1,self.dim) for zt in self.save_zt])
            
            reg_energy = self.reg_energy(zt)
            grad_norm = jnp.linalg.norm(vmap(self.Dregenergy)(zt), axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, reg_energy, grad_norm, idx
    
#%% Gradient Descent Estimation of Geodesics

class ScipyEuclideanOptimization(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun

        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def score_norm2(self, 
                    zt:Array,
                    )->Array:
        
        def score_norm2_path(score_norm2:Array,
                             z:Array,
                             )->Tuple[Array]:
            
            score = self.score_fun(z)

            score_norm2 += jnp.sum(score**2)
            
            return (score_norm2,)*2
        
        zt = zt.reshape(-1, self.dim)
        
        score_norm2, _ = lax.scan(score_norm2_path,
                                  init=self.score_norm20,
                                  xs=zt,
                                  )
        
        return score_norm2
    
    def energy(self, 
               zt:Array,
               )->Array:
        
        zt = zt.reshape(-1, self.dim)

        zt = jnp.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return jnp.sum(jnp.einsum('ti,ti->t', ut, ut))
    
    def reg_energy(self, 
                   zt:Array,
                   *args,
                   )->Array:
        
        zt = zt.reshape(-1, self.dim)

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return energy+self.lam_norm*score_norm2
    
    def Dregenergy(self,
                   zt:Array,
                   *args,
                   )->Array:
        
        return lax.stop_gradient(grad(self.reg_energy)(zt,*args))
    
    def HessRegEnergy(self,
                      zt:Array,
                      )->Array:
        
        return hessian(self.reg_energy)(zt)
    
    def HessPRegEnergy(self,
                       zt:Array,
                       p:Array,
                       )->Array:
        
        return jnp.dot(hessian(self.reg_energy)(zt), p)
    
    def callback(self,
                 zt:Array
                 )->Array:
        
        self.save_zt.append(zt.reshape(-1, self.dim))
        
        return
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)

        self.score_norm20 = lax.stop_gradient(jnp.sum(self.score_fun(z0)**2))
        
        zt = self.init_fun(z0,zT,self.T)
        
        energy_init = self.energy(zt)
        score_norm2_init = self.score_norm2(zt)
        self.lam_norm = self.lam*energy_init/score_norm2_init
        
        if step == "while":
            res = minimize(fun = self.reg_energy, 
                           x0=zt.reshape(-1), 
                           method=self.method, 
                           jac=self.Dregenergy,
                           hess=self.HessRegEnergy,
                           hessp=self.HessPRegEnergy,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
        
            zt = res.x.reshape(-1,self.dim)
            reg_energy = self.reg_energy(zt)
            zt = jnp.vstack((z0, zt, zT))
            grad_norm =  jnp.linalg.norm(res.jac.reshape(-1,self.dim))
            idx = res.nit
        elif step == "for":
            res = minimize(fun = self.reg_energy,
                           x0=zt.reshape(-1),
                           method=self.method,
                           jac=self.Dregenergy,
                           hess=self.HessRegEnergy,
                           hessp=self.HessPRegEnergy,
                           callback=self.callback,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
            
            zt = jnp.stack([zt.reshape(-1,self.dim) for zt in self.save_zt])
            
            reg_energy = vmap(self.reg_energy)(zt)
            grad_norm = jnp.linalg.norm(vmap(self.Dregenergy)(zt), axis=1)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, reg_energy, grad_norm, idx
    
    
    
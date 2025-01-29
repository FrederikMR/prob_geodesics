#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor

from typing import Callable
from abc import ABC
    
#%% Backtracking Line Search

class Backtracking(ABC):
    def __init__(self,
                 obj_fun:Callable,
                 update_fun:Callable,
                 grad_fun:Callable,
                 criterion:str="armijo",
                 alpha:float=1.0,
                 rho:float=0.9,
                 c:float=0.25,
                 c1:float=0.90,
                 c2:float=0.1,
                 max_iter:int=100,
                 )->None:
        #https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.grad_fun = grad_fun
        
        self.alpha = alpha
        self.rho = rho
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        
        if criterion == "fixed":
            self.condition = self.fixed_condition
        elif criterion == "naive":
            self.condition = self.naive_condition
        elif criterion == "armijo":
            self.condition = self.armijo_condition
        elif criterion == "curvature":
            self.condition = self.curvature_condition
        elif criterion == "strong_curvature":
            self.condition = self.strong_curvature_condition
        elif criterion == "wolfe":
            self.condition = self.wolfe_condition
        elif criterion == "strong_wolfe":
            self.condition = self.strong_wolfe_condition
        elif criterion == "goldstein":
            self.condition = self.goldstein_condition
        else:
            raise ValueError("Invalid criterion for line search")
        
        self.x = None
        self.obj0 = None
        
        return
    
    @torch.no_grad()
    def fixed_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        return False
    
    @torch.no_grad()
    def naive_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        return obj>self.obj0
    
    @torch.no_grad()
    def armijo_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        val1 = self.obj0+self.c1*alpha*torch.dot(self.pk, self.grad0)
        
        return obj>val1
    
    def curvature_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        grad_val = self.grad_fun(x_new, *args)
        val0 = torch.dot(grad_val, self.pk)
        val1 = self.c2*torch.dot(self.pk, self.grad0)
        
        return val0 < val1
    
    def strong_curvature_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        grad_val = self.grad_fun(x_new, *args)
        val0 = torch.dot(grad_val, self.pk)
        val1 = self.c2*torch.dot(self.pk, self.grad0)
        
        return torch.abs(val0) > torch.abs(val1)
    
    def wolfe_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        armijo = self.armijo_condition(x_new, obj, alpha, *args)
        curvature = self.curvature_condition(x_new, obj, alpha, *args)
        
        return armijo & curvature
    
    def strong_wolfe_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        armijo = self.armijo_condition(x_new, obj, alpha, *args)
        curvature = self.strong_curvature_condition(x_new, obj, alpha, *args)
            
        return armijo & curvature

    @torch.no_grad()    
    def goldstein_condition(self, x_new:Tensor, obj:Tensor, alpha:Tensor, *args)->bool:
        
        val0 = self.obj0 + (1-self.c)*alpha*torch.dot(self.grad0, self.pk)
        val1 = self.obj0 + self.c*alpha*torch.dot(self.grad0, self.pk)
        
        bool1 = val0>obj
        bool2 = val1<obj
        
        return bool1 & bool2
    
    @torch.no_grad()
    def cond_fun(self, 
                 alpha,
                 idx,
                 *args,
                 )->Tensor:

        x_new = self.update_fun(self.x, alpha, *args)
        obj = self.obj_fun(x_new, *args)
        bool_val = self.condition(x_new, obj, alpha, *args)
        
        return (bool_val) & (idx < self.max_iter)
    
    def update_alpha(self,
                     alpha:float,
                     idx:int,
                     )->Tensor:

        return self.rho*alpha, idx+1
    
    def __call__(self, 
                 x:Tensor,
                 *args,
                 )->Tensor:
        
        self.x = x
        self.obj0 = self.obj_fun(x,*args).item()
        grad_val = self.grad_fun(x,*args).detach()
        self.pk = -grad_val
        self.grad0 = grad_val
        
        alpha, idx = self.alpha, 0
        while self.cond_fun(alpha, idx, *args):
            alpha, idx = self.update_alpha(alpha, idx)
            
        print(idx)

        return alpha
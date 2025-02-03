#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:17:01 2025

@author: fmry
"""

#%% Modules

import torch
from torch import Tensor

#%% Geodesic Optimization Module

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
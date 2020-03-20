# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:21:59 2020

@author: Xinyi Mou
"""

import torch
import torch.nn as nn
import numpy as np

def init_embed(input_embed):
    scope = np.sqrt(3.0 / input_embed.size(1))
    nn.init.uniform_(input_embed, -scope, scope)

def init_linear_weight_bias(input_linear):    
    nn.init.xavier_uniform_(input_linear.weight)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + 1))
    if input_linear.bias is not None:
        input_linear.bias.data.uniform_(-scope, scope)
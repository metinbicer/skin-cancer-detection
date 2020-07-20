# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:50:37 2020

functions for processing the model

@author: Metin Bicer
"""

import torch


'''
creates an fc layer. This function is used in another function after checking
for the layer_name. Otherwise, it throws AttributeError

inputs:
    model           --> pretrained model (of any available in pytorch)
    num_classes     --> number of classes for the problem
    layer_name      --> name of the layer containing last linear layer (fc, classifier etc)
    init_fun        --> initializer for new fc layer

outputs:
    model --> pretrained model modified for the problem
'''
def _lastLayer(model, layer_name, num_classes, init_fun):
    # get the model's fc layer
    current_layer = getattr(model, layer_name)
    
    # this layer can be linear only or a sequential
    if isinstance(current_layer, torch.nn.Linear):
        # create new fc layer
        new_layer = torch.nn.Linear(current_layer.in_features, num_classes)
        
        # initalize the weights
        if init_fun != None:
            init_fun(new_layer.weight)
        
        # set this layer
        setattr(model, layer_name, new_layer)
        
    elif isinstance(model.classifier, torch.nn.Sequential):
        # create new fc layer
        new_layer = torch.nn.Linear(current_layer[-1].in_features, num_classes)
        
        # initalize the weights
        if init_fun != None:
            init_fun(new_layer.weight)
            
        # set this layer
        getattr(model, layer_name)[-1] = new_layer
        
    return model
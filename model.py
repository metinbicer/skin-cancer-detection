# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:50:37 2020

functions for processing the model

@author: Metin Bicer
"""

import torch


'''
prepare the model (pretrained) for training. Freeze the conv layers, add 
fc layer for the number of classes in the problem

inputs:
    model           --> pretrained model (of any available in pytorch)
    num_classes     --> number of classes for the problem
    init_fun        --> initializer for new fc layer
    freeze_convnet  --> if True, dont train the convnet, else training starts 
                        from the pretrained weightings
outputs:
    model --> pretrained model modified for the problem
'''
def prepareModel(model, num_classes, init_fun=None, freeze_convnet=True, **kwargs):
    
    # freeze the parameters of pretrained model
    if freeze_convnet:
        for param in model.parameters():
            param.requires_grad = False
    
    # model can have different fully connected layer names (fc, classifier)
    
    # try fc
    if hasattr(model, 'fc'):
        model = _lastLayer(model, 'fc', num_classes, init_fun)
    # try classifier
    elif hasattr(model, 'classifier'):
        model = _lastLayer(model, 'classifier', num_classes, init_fun)
    elif hasattr(model, kwargs['layer_name']):
        model = _lastLayer(model, kwargs['layer_name'], num_classes, init_fun)
    # raise exception
    else:
        raise AttributeError('Check the name of the last layer in the model. \n' +
                             'Then call the function with the argument "layer_name=your_layer_name"')
        
    return model


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
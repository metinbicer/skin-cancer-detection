# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:33:56 2020

Utility functions for loading data, visualization etc

@author: Metin Bicer

"""

import os
import torch
from torchvision import datasets, transforms, models


# loads the data 
# inputs:
    # model (if inceptionV3, define input size)
    # batch_size
def dataLoader(model, batch_size=32):

    # if the model is Inception V3, the input size is 299x299    
    size = 299 if isinstance(model, models.inception.Inception3) else 225
    
    # define transformation for each dataset
    transform = {'train': transforms.Compose([
                                              transforms.Resize(size),
                                              transforms.CenterCrop(size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])
                                              ]),
                 'valid': transforms.Compose([
                                              transforms.Resize(size),
                                              transforms.CenterCrop(size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])
                                              ]),
                 'test': transforms.Compose([
                                              transforms.Resize(size),
                                              transforms.CenterCrop(size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])
                                              ])
                 }
    
    # data sets from each folder under data
    dataset = {key: datasets.ImageFolder(os.path.join('data', key), transform=transform[key]) for key in transform.keys()}
    
    loaders = {key: torch.utils.data.DataLoader(data, batch_size, shuffle=True) for key, data in dataset.items()}
    
    return loaders
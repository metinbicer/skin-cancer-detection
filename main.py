# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:17:34 2020
defines hyperparameters, model and process the data

@author: Metin Bicer
"""

import torch.nn as nn
import torch.optim as optim
import torchvision
from utils import dataLoader, train, test
from model import prepareModel

# define number of classes in the problem
num_classes = 3

# hyperparameters
lr = 0.001
n_epochs = 40
batch_size = 64

# define model from pytorch
model = torchvision.models.inception_v3(pretrained=True)

# a model will be returned with num_classes for the classification
# only trainable parameters are the last layer
model = prepareModel(model, num_classes)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# batch the data
loaders = dataLoader(model, batch_size)

model = train(n_epochs, loaders, model, optimizer, criterion, 
              save_path='Inceptionv3.pt')
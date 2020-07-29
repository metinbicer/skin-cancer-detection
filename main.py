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
lr = 0.0001
n_epochs = 20
batch_size = 32

# define model from pytorch
model = torchvision.models.resnet18(pretrained=True)

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
              save_path='best.pth.tar')

test(loaders['test'], model, criterion)

# create the dict whose keys are indexes and values are class names
idx_to_class = {val:key for key, val in loaders['train'].dataset.class_to_idx.items()}

# predict an example image
predict('data/nevus.jpg', model, idx_to_class)
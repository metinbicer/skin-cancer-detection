# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:33:56 2020

Utility functions for loading data, visualization etc

@author: Metin Bicer

"""

import os
import torch
from torchvision import datasets, transforms, models
import numpy as np

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

'''
train the given model

inputs:
     n_epochs       --> number of epochs
     loaders        --> train and validation loaders (keys are train and valid)
     optimizer
     criterion      --> loss function
     save_path      --> name or path of the model to be saved
     valid_loss_min --> previous validation loss (if trained previously, default Inf)
     use_cuda       --> bool for using GPU computing (default depends on its availability)
outputs:
    model --> trained model
'''
def train(n_epochs, loaders, model, optimizer, criterion, save_path, 
          valid_loss_min=np.Inf, use_cuda=torch.cuda.is_available()):
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # write the training loss 2 times in an epoch
        every_batch = int(len(loaders['train'])/2)
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                # in case model is on cpu, move it to gpu
                model.cuda()
            else:
                # in case model is in gpu, move it to cpu (because use_cuda=False)
                model.cpu()
                
            # clear optimizer
            optimizer.zero_grad()
            
            # forward pass
            output = model(data)
            
            # if inceptionV3 model, then the output is tuple
            # calculate loss
            if isinstance(output, tuple):
                # real part of the loss
                loss = criterion(output[0], target)
                
                # other auxiliary losses (30% of this loss is added to real loss)
                for aux_out in output[1:]:
                    loss += 0.3 * criterion(aux_out, target)
            else:
                loss = criterion(output, target)
            
            # backward pass
            loss.backward()
            
            # optimizer step
            optimizer.step()

            # average training loss 
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if (batch_idx+1) % every_batch == 0:
                print('Epoch: {}/{} \tBatch: {}/{} \tTraining Loss: {:.6f}'.format(
                    epoch,
                    n_epochs,
                    batch_idx+1,
                    len(loaders['train']),
                    train_loss
                    ))
                
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                                                                                            valid_loss_min,
                                                                                            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    
        # print a line starting new epoc
        print('----------------------------------------------------------')
        
    # return trained model
    return model
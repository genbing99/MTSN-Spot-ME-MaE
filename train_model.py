import os
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader

from prepare_data import *
from network import *
from dataloader import *

def train(dataset_name, X, X1, Y, Y1, epochs, lr_classifier, batch_size, ratio):
    start = time.time()
    epochs = epochs
    batch_size = batch_size

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
            
    loss_fn = nn.MSELoss()
    
    X_train, X1_train = X, X1
    Y_train, Y1_train = Y, Y1

    #Downsampling non expression samples to make ratio expression:non-expression 1:ratio
    print('Dataset Labels', Counter(Y_train))
    rem_count = 0
    for key, value in Counter(Y_train).items():
        if key != 0:
            rem_count += value
    rem_count = rem_count * ratio
    
    #Randomly remove non expression samples (With label 0) from dataset
    rem_index = random.sample([index for index, i in enumerate(Y_train) if i==0], rem_count) 
    rem_index += (index for index, i in enumerate(Y_train) if i>0)
    rem_index.sort()
    X_train = [X_train[i] for i in rem_index]
    X1_train = [X1_train[i] for i in rem_index]
    Y_train = [Y_train[i] for i in rem_index]
    Y1_train = [Y1_train[i] for i in rem_index]
    print('After Downsampling Dataset Labels', Counter(Y_train))

    # Initialize training dataloader
    X_train = torch.Tensor(np.array(X_train)).permute(0,3,1,2)
    X1_train = torch.Tensor(np.array(X1_train)).permute(0,3,1,2)
    Y_train = torch.Tensor(np.array(Y_train))
    Y1_train = torch.Tensor(np.array(Y1_train))
    train_dl = DataLoader(
        OFFSpottingDatasetTrain((X_train, X1_train, Y_train, Y1_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    
    print('------Initializing Network-------') #To reset the model at every LOSO testing
    
    model = MTSN().cuda()
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_classifier)

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            x3   = batch[2].to(device)
            x4   = batch[3].to(device)
            x5   = batch[4].to(device)
            x6   = batch[5].to(device)
            y    = batch[6].to(device)
            y1   = batch[7].to(device)
            optimizer.zero_grad()
            yhat = model(x1,x2,x3,x4,x5,x6).view(-1)
            loss1 = loss_fn(yhat, y)
            loss2 = loss_fn(yhat, y1)
            loss = 0.7 * loss1 + 0.3 * loss1
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()

        train_loss  = train_loss / len(train_dl)

        print('Epoch:', epoch, '| loss:', round(loss.item(), 4), '| hard loss:', round(loss1.item(), 4), '| soft_loss:', round(loss2.item(), 4))

    # Save models
    if dataset_name == 'CASME_sq':
        torch.save(model.state_dict(), os.path.join("megc2022-pretrained-weights/cas_weights_new.pkl"))
    if dataset_name == 'SAMMLV':
        torch.save(model.state_dict(), os.path.join("megc2022-pretrained-weights/samm_weights_new.pkl"))

    end = time.time()
    print('Total time taken for training & testing: ' + str(end-start) + 's')

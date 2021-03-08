# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:27:43 2021

@author: jmathew
#https://github.com/wcneill/jn-ml-textbook/blob/master/Deep%20Learning/04%20Recurrent%20Networks/pytorch13b_LSTM.ipynb
#https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
#https://medium.com/@quantumsteinke/whats-the-difference-between-a-matrix-and-a-tensor-4505fbdc576c
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from numpy import array
from numpy.random import uniform
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#load  trajectory simulations saved as mat file 
mat    = sio.loadmat('C:/Users/jmathew/Dropbox (INMACOSY)/James-UCL/LQG/iLQG/LQGxyBase.mat')
dat    = mat['Traj']
In     = dat['In']
Out    = dat['Out']
#convert to numpy arrays
In2    = np.array([i[0] if isinstance(i, np.ndarray) else i for i in In]);
Out2   = np.array([i[0] if isinstance(i, np.ndarray) else i for i in Out]);
In3    = In2[0,:,:];
Out3   = Out2[0,:,:];

#The LSTM model input dimension requires the third dimension that will be the number of the single input row. We'll reshape the x data.
Out4 = Out3.reshape(Out3.shape[0], Out3.shape[1], 1)   

#standardize the data as the values are very large and varied
sc  = MinMaxScaler()
sct = MinMaxScaler()
X_train = sc.fit_transform(Out4.reshape(-1,1))
y_train = sct.fit_transform(Out4.reshape(-1,1))  

# convert training data to tensor
#X_train = torch.tensor(X_train, dtype=torch.float32)
#y_train = torch.tensor(y_train, dtype=torch.float32)

#Convert the numpy arrays to tensors
X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)

#in_dim  = (Out4.shape[1], Out4.shape[2])
#out_dim = Out4.shape[1]

# The function will accept the raw input data and will return a list of tuples. 
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq   = input_data[i:i+tw]
        train_label = input_data[i:i+tw] #[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_window = 12
train_inout_seq = create_inout_sequences(X_train, train_window)

# create lstm model
# input_size = 360    # The number of variables in your sequence data. 
# n_hidden   = 100  # The number of hidden nodes in the LSTM layer.
# n_layers   = 2    # The total number of LSTM layers to stack.
# out_size   = 1    # The size of the output you desire from your RNN.

# lstm   = nn.LSTM(input_size, n_hidden, n_layers, batch_first=True)
# linear = nn.Linear(n_hidden, 1)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

#Define loss and optimizer
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

#training model
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')




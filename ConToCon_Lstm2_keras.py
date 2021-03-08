# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:27:43 2021

@author: jmathew
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

in_dim  = (Out4.shape[1], Out4.shape[2])
out_dim = Out4.shape[1]

# Now, we can split the data into the train and test parts.
xtrain, xtest, ytrain, ytest=train_test_split(Out4, Out4, test_size=0.15)

# defining the sequential model. The sequential model contains LSTM layers with ReLU activations, Dense output layer,  and Adam optimizer with MSE loss function. 
# set the input dimension in the first layer and output dimension in the last layer of the model.

model2 = Sequential()
model2.add(LSTM(64, input_shape=in_dim, activation="relu"))
model2.add(Dense(out_dim))
model2.compile(loss="mse", optimizer="adam") 
model2.summary()

# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir='.\logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]

model2.fit(xtrain, ytrain, epochs=100, batch_size=12, verbose=1, callbacks=keras_callbacks)

# Generate generalization metrics
score = model2.evaluate(xtest, ytest, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
#Predicting and visualizing the results

ypred = model2.predict(xtest)
 
print("y1 MSE:%.4f" % mean_squared_error(ytest[:,0], ypred[:,0]))
print("y2 MSE:%.4f" % mean_squared_error(ytest[:,1], ypred[:,1])) 

#The result can be visualized as below.

x_ax = range(len(xtest))
plt.title("LSTM multi-output prediction")
plt.scatter(x_ax, ytest[:,0],  s=6, label="y1-test")
plt.plot(x_ax, ypred[:,0], label="y1-pred")
plt.scatter(x_ax, ytest[:,1],  s=6, label="y2-test")
plt.plot(x_ax, ypred[:,1], label="y2-pred")
plt.legend()
plt.show()

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
model2_json = model2.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model2_json)
# serialize weights to HDF5
model2.save_weights("model2.h5")
print("Saved model to disk")
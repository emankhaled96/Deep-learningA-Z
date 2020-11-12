# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 02:25:51 2020

@author: admin
"""

# Make a Hybrid Deep Learning Model


# Part 1: Identify the Frauds with SOM
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset 
dataset =pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , -1].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len = 15 ,sigma = 1.0  ,learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X ,num_iteration = 100)


# Visualizing the SOM
from pylab import bone , pcolor , colorbar , plot , show
# to show the window of the figure we use bone
bone()
pcolor(som.distance_map().T)
colorbar()
markers =['o','s']
colors = ['r','g']
for i , x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10 , 
         markeredgewidth = 2)
show()

# Finding the frauds (cheaters)
mappings = som.win_map(X)
# everytime i run this code on a new kernel the values in mapping will change
frauds = np.concatenate((mappings[(8,7)],mappings[(7,8)]) , axis =0)
frauds = sc.inverse_transform(frauds)


# Part 2: Going from Unsupervised to Supervised Deep Learning  

# creating the matrix of Features
customers = dataset.iloc[: , 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1
        
 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

# Building ANN
# 1- Importing ANN Libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 2- Intializing ANN
classifier = Sequential()

# Adding layers
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2 , activation= 'relu' ,input_dim = 15))
classifier.add(Dropout(p=0.1))


# Adding the output layer
classifier.add(Dense(units = 1 , activation = 'sigmoid'))

# Compiling  the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(customers,is_fraud,batch_size =1,epochs=3 )

# Predicting 
Y_pred = classifier.predict(customers)
Y_pred = np.concatenate((dataset.iloc[: , 0:1].values,Y_pred) , axis =1)
Y_pred =Y_pred[Y_pred[:,1].argsort()]
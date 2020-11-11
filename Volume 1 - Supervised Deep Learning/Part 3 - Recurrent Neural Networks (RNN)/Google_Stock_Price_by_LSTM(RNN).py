# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:03:44 2020

@author: admin
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# Part-1 Data Preprocessing 
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the training set 

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: , 1:2].values

# Feature Scaling
# we uses here the normalization method in scaling the features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 Timesteps and 1 output
# it means that at each time t the rnn will look at the last 60 stocks prices ( 60 days before this time t )
# and depend on the trend of these days it will predict the new price
# 1 output is the stock price at t+1
# timesteps can be tunned in the course he tried 20 but it make overfitting problem so he ended up that the best num = 60
x_train = []
y_train = [] 
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

x_train , y_train = np.array(x_train) , np.array(y_train)

# reshaping
x_train = np.reshape(x_train , (x_train.shape[0],x_train.shape[1],1))


# Part-2 : building RNN

# intializing the RNN
regressor = tf.keras.models.Sequential()

# Adding the first LSTM layer and some dropout regularization
# return_sequences = True : to tell this layer that there is another layer after it
regressor.add(tf.keras.layers.LSTM(units = 50 ,return_sequences = True , input_shape = (x_train.shape[1],1)))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the second LSTM layer and dropout regularization
regressor.add(tf.keras.layers.LSTM(units = 50 ,return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the third LSTM layer and dropout regularization
regressor.add(tf.keras.layers.LSTM(units =50 ,return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the fourth LSTM layer and dropout regularization
regressor.add(tf.keras.layers.LSTM(units = 50 ,return_sequences = False))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer
regressor.add(tf.keras.layers.Dense(units=1))

# Compiling RNN
# we can use adam or rmsprop optimizers in this problem
regressor.compile(optimizer= 'adam',loss = 'mean_squared_error')

# fitting the RNN to the training set
regressor.fit(x_train , y_train , epochs = 100 , batch_size = 32)

# Part-3 :Predicting the google stock price and visualizing it
# getting the real stock price in 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[: , 1:2].values

# getting the predicting value for the price of the google stock in january 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']) , axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test) 

# reshaping
x_test = np.reshape(x_test , (x_test.shape[0],x_test.shape[1],1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the results
plt.plot(real_stock_price , color = 'red' , label = 'Real Google Stock Price')
plt.plot(predicted_stock_price , color = 'blue' , label = 'predicted Google Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()












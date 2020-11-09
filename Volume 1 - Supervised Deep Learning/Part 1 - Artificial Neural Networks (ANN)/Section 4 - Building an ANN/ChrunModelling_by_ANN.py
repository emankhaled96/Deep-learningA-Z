# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:08:13 2020

@author: admin
"""
# Importing the main libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Features & Target
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

# Category encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1])
transform = ColumnTransformer([("1",OneHotEncoder(),[1])],remainder="passthrough")

labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X1.fit_transform(X[:,2])

X = transform.fit_transform(X)
X = X[:,1:]

# Data Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.20,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Building ANN
# 1- Importing ANN Libraries
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 2- Intializing ANN
classifier = Sequential()

# Adding layers
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6 , activation= 'relu' ,input_dim = 11))
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6 , activation = 'relu'))
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(units = 1 , activation = 'sigmoid'))

# Compiling  the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(X_train,Y_train,batch_size =10,epochs=100 )

# Predicting 
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Evaluating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print (cm)

# Homework Section 
# Predicting  a new single observation
""" predict if the customer with the following information will leave the bank:
Geography : France
Credit Score : 600
Gender : Male
Age : 40
Tenure : 3
Balance : 60000
Number of products : 2
Has Credit Card : YES
Is Active member :YES
Estimated Salary : 50000"""

# we used sc_X.transform  to make feature scaling to the new observation as the model is trained on scaled features
new_pred = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)               

# Evaluating , Improving , Tunning the ANN

# Evaluating the ANN
# we should connect to a new kernel and execute just the preprocessing part because the ANN part will be included in here
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6 , activation= 'relu' ,input_dim = 11))
    classifier.add(Dense(units = 6 , activation = 'relu'))
    classifier.add(Dense(units = 1 , activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10 , epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train , y = Y_train , cv =10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# we should connect to a new kernel and execute just the preprocessing part because the ANN part will be included in here

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6 , activation= 'relu' ,input_dim = 11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 6 , activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 1 , activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25,32],
              'epochs' : [100,500],
              'optimizer' : ['adam' , 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring= 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train , Y_train)
best_parameters = grid_search.best_params_
best_accuarcy = grid_search.best_score_
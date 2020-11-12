# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:12:59 2020

@author: admin
"""
# importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# setting size
#batch_size = 32

# intializing the CNN
classifier = Sequential()

# Adding first Convoltional Layer
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3),activation='relu'))

# Adding MaxPooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second convolutional layer and maxpooling 
classifier.add(Convolution2D(32,3,3,activation='relu'))

# Adding MaxPooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

"""# adding the third convolutional layer
classifier.add(Convolution2D(32,3,3,activation = 'relu',padding='same'))
classifier.add(MaxPooling2D(pool_size = (2,2),padding='same'))
"""

# Adding Flatten layer
classifier.add(Flatten())

#Adding Full connected layer
# 1/ adding hidden layers
from tensorflow.keras.layers import Dropout
classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dropout(rate = 0.2))

# 2/ adding output layer
classifier.add(Dense(units = 1 , activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting the CNN to the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary',
                                                shuffle=True)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary',
                                             shuffle=True)

# steps_per_epoch = number of images in the training set
# validation_steps = number of images in test set
classifier.fit_generator(training_set,
                    steps_per_epoch =8000,
                    epochs = 25,
                    validation_data = test_set,
                    validation_steps = 2000, 
                    shuffle = True)


# Making a new prediction 
import numpy as np
from keras_preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
    prediction = 'dog'
else:
    prediction = 'cat'
#Accuracy on CPU = 80%
#Accuracy on GPU = 70%

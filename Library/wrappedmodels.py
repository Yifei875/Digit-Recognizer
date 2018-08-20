# Copyright (c) Microsoft Corporation
#
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.base import BaseEstimator, RegressorMixin

class SeqCNN(BaseEstimator, RegressorMixin):
    """
    Description: wrapped a sequential CNN from Keras Sequential API.

    """
    def __init__(self, 
                 epochs = 10,
                 steps_per_epoch = 500,
                 verbose = 2
                 ):
        """
        """
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        

    def fit(self, input_data, validation_data):
        """
        """
        # Initialize a sequential CNN model
        model = Sequential()

        # Add two convolutional (Conv2D) layers, action function is relu
        model.add(Conv2D(filters = 16, 
                         kernel_size = (3, 3), 
                         activation='relu',
                         input_shape = (28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters = 16, 
                         kernel_size = (3, 3), 
                         activation='relu'))
        model.add(BatchNormalization())

        # Add one pooling (MaxPool2D) layer
        model.add(MaxPool2D(strides=(2,2)))
        # Add 25% dropout regularization
        model.add(Dropout(0.25))

        # Add two convolutional (Conv2D) layers, action function is relu
        model.add(Conv2D(filters = 32, 
                         kernel_size = (3, 3), 
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters = 32, 
                         kernel_size = (3, 3), 
                         activation='relu'))
        model.add(BatchNormalization())
        
        # Add one pooling (MaxPool2D) layer
        model.add(MaxPool2D(strides=(2,2)))
        # Add 25% dropout regularization
        model.add(Dropout(0.25))

        # Add one flatten layer
        model.add(Flatten())
        # Add one Dense layer
        model.add(Dense(512, activation='relu'))
        # Add 25% dropout regularization
        model.add(Dropout(0.25))
        # Add one Dense layer
        model.add(Dense(1024, activation='relu'))
        # Add 50% dropout regularization
        model.add(Dropout(0.5))
        # Add one Dense layer 
        model.add(Dense(10, activation='softmax'))
        # Define the optimizer
    
        # Compile the model, using categorial cross entropy and accuracy as metrics
        model.compile(loss='categorical_crossentropy', 
                      optimizer = Adam(lr=1e-4), 
                      metrics=["accuracy"])

        # Learning rate annealer
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)


        model.fit_generator(input_data,
                            validation_data = validation_data,
                            epochs=self.epochs,
                            steps_per_epoch=self.steps_per_epoch,
                            verbose=self.verbose,  #1 for ETA, 0 for silent
                            callbacks=[annealer])
        self = model
        return self

    def predict(self, X):
        """
        """
        y_pred = self.predict(X, batch_size = 64)
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = pd.Series(y_pred,name="Label")
        return y_pred
        
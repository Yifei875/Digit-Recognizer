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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin

class SeqCNN(BaseEstimator, RegressorMixin):
    """
    Description: wrapped a sequential CNN from Keras Sequential API.

    """
    def __init__(self, 
                 epochs, 
                 model_verbose,
                 steps_per_epoch,
                 batch_size= 86,
                 monitor = 'val_acc', 
                 patience = 3, 
                 LR_verbose = 1, 
                 factor = 0.5, 
                 min_lr=0.00001
                 ):
        """
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_verbose = model_verbose
        self.steps_per_epoch = steps_per_epoch
        self.monitor = monitor
        self.patience = patience
        self.LR_verbose =LR_verbose
        self.factor = factor
        self.min_lr = min_lr
        

    def fit(self, data_gen, validation_data):
        """
        """
        # Initialize a sequential CNN model
        model = Sequential()
        # Add two convolutional (Conv2D) layers, action function is relu
        model.add(Conv2D(filters = 32, 
                         kernel_size = (5,5),
                         padding = 'Same', 
                         activation ='relu', 
                         input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, 
                         kernel_size = (5,5),
                         padding = 'Same', 
                         activation ='relu'))
        # Add one pooling (MaxPool2D) layer
        model.add(MaxPool2D(pool_size=(2,2)))
        # Add 25% dropout regularization
        model.add(Dropout(0.25))

        # Add two convolutional (Conv2D) layers, action function is relu
        model.add(Conv2D(filters = 64, 
                         kernel_size = (3,3),
                         padding = 'Same', 
                         activation ='relu'))
        model.add(Conv2D(filters = 64, 
                         kernel_size = (3,3),
                         padding = 'Same', 
                         activation ='relu'))
        # Add one pooling (MaxPool2D) layer
        model.add(MaxPool2D(pool_size=(2,2), 
                            strides=(2,2)))
        # Add 25% dropout regularization
        model.add(Dropout(0.25))

        # Add one flatten layer
        model.add(Flatten())
        # Add one Dense layer
        model.add(Dense(256, activation = "relu"))
        # Add 50% dropout regularization
        model.add(Dropout(0.5))
        # Add one Dense layer
        model.add(Dense(10, activation = "softmax"))

        # Define the optimizer
        optimizer = RMSprop(lr= 0.001, 
                            rho = 0.9, 
                            epsilon = 1e-08, 
                            decay =0.0)
    
        # Compile the model, using categorial cross entropy and accuracy as metrics
        model.compile(optimizer = optimizer ,
                      loss = "categorical_crossentropy",
                      metrics=["accuracy"])
        # Learning rate annealer
        # Reduce the LR by half if the accuracy is not improved after 3 epochs
        learning_rate_reduction = ReduceLROnPlateau(monitor=self.monitor, 
                                                    patience=self.patience, 
                                                    verbose=self.LR_verbose, 
                                                    factor=self.factor, 
                                                    min_lr=self.min_lr)

        self.fit = model.fit_generator(generator = data_gen, 
                                       epochs = self.epochs,
                                       validation_data= validation_data,
                                       verbose = self.model_verbose,
                                       steps_per_epoch=self.steps_per_epoch//self.batch_size,
                                       callbacks=[learning_rate_reduction])
        return self

    def predict(self, X):
        """
        """
        y_pred = self.fit.predict(X)
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = pd.Series(y_pred,name="Label")
        return y_pred
        
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
import datetime
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

import sys
sys.path.append('..')
from keras.preprocessing.image import ImageDataGenerator

def main():
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Loading data...")

    # Read data and merge
    training = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')
    target = training["label"]
    training.drop(labels=["label"], axis = 1, inplace = True)
    print("data prepared successfully.")

    # Grayscale normalization
    training = training/255.0
    test = test/255.0
    
    # Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
    training = training.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # Laebl encoding target
    target = to_categorical(target, num_classes = 10)
    
    # Set random seed
    random_seed = 42

    # Split the trarin and the validation set for the fitting
    print("Splitting training into training and validation sets...")
    X_train, X_val, Y_train, Y_val = train_test_split(training, target, test_size = 0.1, random_state = random_seed)
    
    end_time = datetime.datetime.now().replace(microsecond=0)
    print('Elapsed time of preprocessing: {}'.format(end_time - start_time))

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)
    
    return datagen, X_train, Y_train, X_val, Y_val, test



if __name__=='__main__':
    main()
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
import datetime
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

import sys
sys.path.append('..')
from keras.preprocessing.image import ImageDataGenerator

def main(random_seed = 42):
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Start preprocessing...")

    # prepare training data
    training ='../Data/train.csv'
    input_data = np.loadtxt(training, skiprows=1, dtype='int', delimiter=',')

    # Split the trarin and the validation set for the fitting
    x_train, x_val, y_train, y_val = train_test_split(input_data[:,1:], 
                                                      input_data[:,0], 
                                                      test_size=0.1,
                                                      random_state = random_seed)

    # Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)

    # Grayscale normalization
    x_train = x_train.astype("float32")/255
    x_val = x_val.astype("float32")/255

    # Label encoding target
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # Data augmentation
    datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

    # prepare test data
    test = '../Data/test.csv'
    test_data = np.loadtxt(test, skiprows=1, dtype='int', delimiter=',')
    x_test = test_data.astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1)/255

    end_time = datetime.datetime.now().replace(microsecond=0)
    print('Data preprocessing completed.: {}'.format(end_time - start_time))
    return datagen, x_train, x_val, y_train, y_val, x_test



if __name__=='__main__':
    main()
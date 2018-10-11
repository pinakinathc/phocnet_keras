# This is the classifier code which trains a model to generate
# the PHOC of a word image.
# Reference: https://arxiv.org/abs/1604.00187
#
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# The below code ensures that GPU memory is dynamically allocated
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.utils import multi_gpu_model # If we want to use multiple GPUs

from keras.models import Sequential, model_from_json
from keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
                    LeakyReLU, Activation)
from keras.optimizers import SGD
from keras import losses
from keras.callbacks import TensorBoard

from load_data import load_data
from save_load_weight import *
from evaluate_phoc import *

# Thanks to https://github.com/yhenon/keras-spp for the SPP Layer
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from datetime import datetime

JUMP = 5 # Number of Epochs after we Save Model
GPUS = 3 # Number of GPU we want to use for Train & Test


def create_model():
  """This module creates an Instance of the Sequential Class in Keras.

  Args:
    None.

  Return:
    model: Instance of the Sequential Class
  """
  time_start = datetime.now()

  model = Sequential()
  model.add(Conv2D(64, (3, 3), padding='same',
          activation='relu', input_shape=(50, 100, 1)))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(SpatialPyramidPooling([1, 2, 4]))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(604, activation='sigmoid'))

  model = multi_gpu_model(model, gpus=GPUS)

  loss = losses.binary_crossentropy
  optimizer = SGD(lr=1e-4, momentum=.9, decay=5e-5)
  model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  model.summary()
  print ("Time taken to create model: ", datetime.now()-time_start)
    
  return model


def trainer(model, x_train, y_train, x_valid, y_valid, initial_epoch=0):
  """This trains the model partially and
  returns the partially trained weights

  Args:
    model: Instance of the Sequential Class storing the Neural Network
    x_train: Numpy storing the training Images
    y_train: Numpy storing the PHOC Label of the training Images
    x_valid: Numpy storing the Validation Images
    y_valid: Numpy storing the PHOC Labels of Validation Data
    initial_epoch: Starting Epoch of the partial Train (Default: 0)

  Returns:
    model: Instance of the Sequential Class having partially trained model
  """
  tnsbrd = TensorBoard(log_dir='./logs')
  model.fit(x_train,
            y_train,
            batch_size=10,
            callbacks=[tnsbrd],
            epochs=initial_epoch+JUMP,
            initial_epoch=initial_epoch,
            validation_data=(x_valid, y_valid))
  return model


def train(initial_epoch=0):
  """This is the main driver function which first partialy trains the model.
  Then it passes to the weight Saving & Loading modules.

  Args:
    initial_epoch: Integer. Provides the starting point for Training.

  Returns:
    None.
  """
  time_start = datetime.now()
  model = create_model()
  if initial_epoch: # If you are not starting from begining
    model = load_model_weight(model)
  data = load_data()
  x_train = data[0]
  y_train = data[1]
  x_valid = data[3]
  y_valid = data[4]
  x_test = data[6]
  y_test = data[7]
  test_transcripts = data[8]
  for i in range(initial_epoch, 60000, JUMP):
    model = trainer(model, x_train, y_train, x_valid, y_valid, initial_epoch=i)
    save_model_weight(model) # Saves the model
    map(model, x_test, y_test, test_transcripts) # Calculates the MAP of the model
  print ("Time taken to train the entire model: ", datetime.now()-time_start)
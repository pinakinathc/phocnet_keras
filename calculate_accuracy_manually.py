# This module is provided incase anyone is interested to check the accuracy
# of their trained model. Please note that since we use a naive approach
# of K Nearest Neighbour Algorithm for Prediction, the Test Data should small.
#
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import numpy as np
import tensorflow as tf

from phoc_classifier import create_model
from save_load_weight *
from load_data import load_data

# Uncomment the following if you have a GPU and do not want to use entire GPU.
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)


def calculate():
  start = datetime.now()
  model = create_model()
  model = load_model_weight(model)
  print ("Model loading time: ", datetime.now()-start)

  start = datetime.now()
  data = load_data()
  x_test = data[6]
  y_test = data[7]
  transcripts = data[8]

  print ("Time taken to load data: ", datetime.now()-start)
  start = datetime.now()
  print (x_test.shape)
  y_pred = model.predict(x_test)
  print ("Time taken to predict all data: ", datetime.now()-start)

  print (np.amax(y_pred), np.amin(y_pred))
    
  N = len(transcripts)
  total_acc = 0
  start = datetime.now()
  for k in range(1000):
    check = y_test[k]
    acc = []
    y_pred = np.where(y_pred<0.5, 0, 1)
    acc = np.sum(abs(y_pred - check), axis=1)
    if transcripts[np.argmin(acc)] == transcripts[k]:
      total_acc += 1

  print ("Time taken to calculate l2 dist: ", datetime.now() - start)
  print ("The total accuracy = ", total_acc)
  print ("Total test cases = ", N)

calculate()

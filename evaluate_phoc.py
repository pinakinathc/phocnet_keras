# This file contains modules to evaluate the partially trained model.
# According to the paper the MAP of PHOCNet for IAM dataset In QbE is 72.51.
#
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from load_data import load_data
from save_load_weight import *


def map(model, x_test, y_test, transcripts):
  """This module evaluates the partially trained model using Test Data

  Args:
    model: Instance of Sequential Class storing Neural Network
    x_test: Numpy storing the Test Images
    y_test: Numpy storing the PHOC Labels of Test Data
    transcripts: String storing the characters in the Image.

  Returns:
    map: Floating number storing the Mean Average Precision.
  """
  start = datetime.now()
  y_pred = model.predict(x_test)
  y_pred = np.where(y_pred<0.5, 0, 1)
  print ("Time taken to predict all data: ", datetime.now()-start)
  start = datetime.now() 
  N = len(transcripts)
  precision = {}
  count = {}
  for i in range(N):
    if transcripts[i] not in precision.keys():
      precision[transcripts[i]] = 1
      count[transcripts[i]] = 0
    else:
      precision[transcripts[i]] += 1

  for i in range(N):
    pred = y_pred[i]
    acc = np.sum(abs(y_test-pred), axis=1)
    tmp = np.argmin(acc)
    if transcripts[tmp] == transcripts[i]:
      count[transcripts[tmp]] += 1

  mean_avg_prec = [0, 0]
  for i in range(N):
    if precision[transcripts[i]] <= 1:
      continue
    mean_avg_prec[0] += count[transcripts[i]]*1.0/precision[transcripts[i]] 
    mean_avg_prec[1] += 1

  print ("Time taken to calculate l2 dist: ", datetime.now() - start)
  print ("The Mean Average Precision = ", mean_avg_prec[0]*1./mean_avg_prec[1])
  print ("Total test cases = ", N)

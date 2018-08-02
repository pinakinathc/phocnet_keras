'''This is the classifier code which trains a model to generate
the PHOC of a word image.
Reference: https://arxiv.org/abs/1604.00187

Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''

# The below code ensures that GPU memory is dynamically allocated
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_config.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

import keras
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
													LeakyReLU, Activation)
from keras.optimizers import SGD
from keras import losses
from keras.callbacks import ModelCheckpoint, TensorBoard
from spp.SpatialPyramidPooling import SpatialPyramidPooling
import numpy as np
from load_data_not_resize import load_data
from evaluate_accuracy import accuracy

JUMP = 5

def create_model():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(None, None, 1)))
	#model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	#model.add(LeakyReLU(alpha=0.3))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	# model.add(Flatten())
	model.add(SpatialPyramidPooling([1, 2, 4]))
	model.add(Dense(4096, activation='relu'))
	#model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	#model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(0.5))
	model.add(Dense(604, activation='linear'))
	model.add(Activation('softmax'))

	model.summary()

	return model

def train(x_train, y_train, model=None, epochs=JUMP):
	if model == None:
		model = create_model()

		loss = losses.categorical_crossentropy
		optimizer = SGD(lr=1e-4, momentum=.9, decay=5e-5)
		model.compile(loss=loss, optimizer=optimizer)

		model_ckpt_1 = ModelCheckpoint(
									'saved_models/phoc_weights_last.hdf5',
									period=5,
									save_weights_only=True)
		model_ckpt_2 = ModelCheckpoint(
						'saved_models/phoc_weights_best.hdf5',
						save_best_only=True,
						period=5,
						save_weights_only=True)

		tnsbrd = TensorBoard(log_dir='./phoc_logs')

	N = len(x_train)
	for epoch in range(epochs):	
		for index in range(N):
			model.fit(np.array([x_train[index]]), 
						np.array([y_train[index]]),
						batch_size=1,
						callbacks=[model_ckpt_1, model_ckpt_2, tnsbrd],
						epochs=index+1,
						initial_epoch=index)
	return model

def l2(vec_1, vec_2):
	return ((vec_1-vec_2)**2).sum()**0.5

def accuracy(model, x_test, y_test):
	x = []
	y = []
	y_pred = []
	if len(x_test) > 1000:
		for i in range(1000):
			j = int(1000*np.random.random())
			x.append(x_test[j])
			y_pred.append(model.predict(np.array([x_test[j]])))
			y.append(y_test[j])
		x_test = np.array(x)
		y_test = np.array(y)

	N = len(x_test)
	correct = 0

	for i in range(N):
		min_dist = l2(y_pred[i], y_test[0])
		index = 0
		for j in range(1, N):
			tmp = l2(y_pred[i], y_test[j])
			if tmp < min_dist:
				min_dist = tmp
				index = j

		if index == i:
			correct += 1

	print ("The accuracy of the model is: ", correct*100.0/N)
	return correct*100.0/N

x_train, y_train, x_test, y_test = load_data()

model = train(x_train, y_train)
accuracy(model, x_test, y_test)
for i in range(0, 1000, JUMP):
	model = train(x_train, y_train, model=model, epochs=JUMP)
	accuracy(model, x_test, y_test)
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

def train(x_train, y_train, model=None, epochs=3):
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

def evaluate(model, x_test, y_test):
	y_pred = model.predict(x_test)
	error = ((((y_pred-y_test)**2).sum(axis=1))**1).sum()
	print ("This is the current error: ", error)
	print ("If this model is giving total random value, then it's value\
		should lie around 604000")

x_train, y_train, x_test, y_test = load_data()

model = train(x_train, y_train)
evaluate(model, x_test, y_test)
accuracy(model, x_test, y_test)
for i in range(1000):
	model = train(x_train, y_train, model=model)
	evaluate(model, x_test, y_test)
	accuracy(model, x_test, y_test)
'''This is the classifier code which trains a model to generate
the PHOC of a word image.
Reference: https://arxiv.org/abs/1604.00187

Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''

import keras
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
													LeakyReLU, Activation)
from keras.optimizers import SGD
from keras import losses
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from load_data import load_data

from spp.SpatialPyramidPooling import SpatialPyramidPooling
def create_model():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), padding='same', input_shape=(None, None, 1)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.3))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Conv2D(512, (3, 3), padding='same'))
	model.add(Conv2D(512, (3, 3), padding='same'))
	model.add(Conv2D(512, (3, 3), padding='same'))
	#model.add(Flatten())
	model.add(SpatialPyramidPooling([1, 2, 4]))
	model.add(Dense(4096))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(0.5))
	model.add(Dense(604, activation='linear'))
	model.add(Activation('softmax'))

	model.summary()

	return model

def train():
	model = create_model()

	loss = losses.categorical_crossentropy
	optimizer = SGD(lr=1e-4, momentum=.9, decay=5e-5)
	model.compile(loss=loss, optimizer=optimizer)

	model_ckpt = ModelCheckpoint(
								'saved_models/weights.hdf5',
								period=5)

	tnsbrd = TensorBoard(log_dir='./logs')

	x_train, y_train, _ , _ = load_data()
	x_train = x_train[:10000]
	y_train = y_train[:10000]

	model.fit(x_train, 
						y_train,
						batch_size=10,
						callbacks=[model_ckpt, tnsbrd],
						epochs=10000)

train()
#create_model()
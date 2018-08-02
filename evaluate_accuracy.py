'''This code evaluates the accuracy of the model using k-means algorithm.
We cannot use L2 loss or cross-entropy loss
Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''
import numpy as np

def l2(vec_1, vec_2):
	return ((vec_1-vec_2)**2).sum()**0.5

def accuracy(model, x_test, y_test):
	x = []
	y = []
	if len(x_test) > 1000:
		for i in range(1000):
			j = int(1000*np.random.random())
			x.append(x_test[j])
			y.append(y_test[j])
		x_test = np.array(x)
		y_test = np.array(y)
	y_pred = model.predict(x_test)

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
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras import optimizers


from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

class Neural_Net(object):

	def __init__(self, train_data, val_data, mask):

		self.train_data = train_data
		self.val_data   = val_data
		self.mask       = mask

		self.m          = self.train_data.shape[0]
		self.n 			= self.train_data.shape[1]
		


		self.construct_input()
		# self.train_x = self.train_x[0:1000]
		# self.train_y = self.train_y[0:1000]

		# print(self.train_x.shape)
		# print(self.train_y.shape)


		self.model = self.construct_model()
	#def build_network():


	def construct_input(self):
		m = self.m
		n = self.n

		user_indices, movie_indices = (np.where(self.train_data > 0))
		scores = self.train_data[self.mask]

		num_train_samples = user_indices.shape[0]

		self.train_x = np.zeros((num_train_samples, m+n))
		self.train_y = np.zeros((num_train_samples))

		start = time.time()

		#construct training input and output X, y
		for i in range(num_train_samples):
			u_ind = user_indices[i]
			m_ind = movie_indices[i]

			self.train_x[i, u_ind]   = 1
			self.train_x[i, m+m_ind] = 1

			self.train_y[i] = self.train_data[u_ind, m_ind]



		#construct test inputs for where we need to predict values
		user_indices, movie_indices = np.where(self.mask)
		num_test_samples = user_indices.shape[0]
		self.test_x = np.zeros((num_test_samples, m+n))

		for i in range(num_test_samples):
			u_ind = user_indices[i]
			m_ind = movie_indices[i]

			self.test_x[i, u_ind]   = 1
			self.test_x[i, m+m_ind] = 1

		print(time.time() - start)

	def construct_model(self):
		model = Sequential()

		input_size = self.m + self.n
		#num_neurons = 100000
		one_tenth = int(input_size/10)

		'linearly decrease the number of neurons'
		# print(input_size)


		model.add(Dense(input_size, activation='relu', input_shape=(input_size,)))
		for i in range(10):
			input_size = input_size - one_tenth
			# print(model.output_shape)

			model.add(Dense(input_size, activation='relu') )

		# print (model.output_shape)
		model.add(Dense(1, activation='relu'))


		# model says they optimized the log loss error

		adam = optimizers.Adam(lr=.01, decay=.001)
		model.compile(loss='mean_squared_logarithmic_error', optimizer=adam, metrics=['accuracy'])

		return model

	def train_model(self):

		# lets make checkpoints

		filepath="nn_model_{epoch:02d}.hdf5"
		checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		callbacks_list = [checkpoint]

		self.model.fit(self.train_x, self.train_y, batch_size=100, epochs=3, callbacks=callbacks_list, verbose=1)

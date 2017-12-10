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

class Collaborative_Filtering_Neural_Net(object):

	def __init__(self, train_data, val_data, mask, num_layers=3, learn_rate=.2):

		self.train_data = train_data
		self.val_data   = val_data
		self.mask       = mask
		self.num_layers = num_layers

		self.m          = self.train_data.shape[0]
		self.n 			= self.train_data.shape[1]
		
		self.learn_rate = learn_rate

		self.construct_input()


	def construct_input(self):
		'''
		Construct training input/output from the training data matrix
		and 
		Construct validation input/output from the training/validation 
		'''
		def change_to_one_hot(value, value_range):
			one_hot_vec = np.zeros(len(value_range))
			one_hot_vec[int(value/.5)] = 1
			return one_hot_vec


		m = self.m
		n = self.n

		user_indices, movie_indices = (np.where(self.train_data > 0))
		scores = self.train_data[self.mask]

		num_train_samples = user_indices.shape[0]

		self.train_x = np.zeros((num_train_samples, m+n))
		self.train_y = np.zeros((num_train_samples, 11))

		start = time.time()

		#construct training input and output X, y
		for i in range(num_train_samples):
			u_ind = user_indices[i]
			m_ind = movie_indices[i]

			self.train_x[i, u_ind]   = 1
			self.train_x[i, m+m_ind] = 1

			score 			= self.train_data[u_ind, m_ind]
			self.train_y[i] = change_to_one_hot(score, np.arange(0,5.5,.5))



		#construct test inputs for where we need to predict values
		user_indices, movie_indices = np.where(self.mask)
		num_test_samples = user_indices.shape[0]
		self.test_x = np.zeros((num_test_samples, m+n))
		self.test_y = np.zeros((num_test_samples, 11))

		for i in range(num_test_samples):
			u_ind = user_indices[i]
			m_ind = movie_indices[i]

			self.test_x[i, u_ind]   = 1
			self.test_x[i, m+m_ind] = 1

			score 		   = self.val_data[u_ind, m_ind]
			self.test_y[i] = change_to_one_hot(score, np.arange(0,5.5,.5))

		print(time.time() - start)


	def construct_model(self, hidden_layer_pattern = 'exponential'):
		'''
		Constructs a Neural network with a given pattern.
		The pattern indicates how many neurons should exist at every layer.
		Param:
			hidden_layer_pattern - The input layer and output layer are fixed, but the rate at which the layer sizes
			decreases depends on the parameter, hidden_layer_pattern
		'''
		model = Sequential()
		input_size = self.m + self.n
		
		# add the first layer
		model.add(Dense(input_size, activation='relu', input_shape=(input_size,)))

		#one of the two model architectures tested
		if (hidden_layer_pattern == 'linear'):
			linear_decrease = int(input_size/self.num_layers)
			for i in range(self.num_layers):
				input_size = input_size - linear_decrease
				model.add(Dense(input_size, activation='relu') )

		if (hidden_layer_pattern == 'exponential'):
			exponential_decrease = int((np.exp(np.log(input_size)/(self.num_layers+2))))
			print(exponential_decrease)
			for i in range(self.num_layers):
				input_size = int(input_size/exponential_decrease);
				model.add(Dense(input_size, activation='relu') )

		print (model.output_shape)
		#one hot encoded output
		model.add(Dense(11, activation='relu'))


		# model says they optimized the log loss error

		adam = optimizers.Adam(lr=self.learn_rate, decay=.001)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

		self.model = model

	def train_model(self, model_number = 0):
		'''
		Trains the model. Saves checkpoints of the model at every epoch.
		I personally just stop training when I find that the loss function has barely changed. Since it takes
		so long to perform each epoch on my computer, I just keep running a 20 epoch train, stop it when I
		have to, then train again later.
		Param:
			model_number - Just changes the filename that the model is saved to. 
						   Don't want to overwrite good save files during training, do you?

		Note: these checkpoints are 1GB each.
		'''
		# lets make checkpoints
		filepath = "nn_model_{}_lr_{}".format(model_number,self.learn_rate)
		filepath+= "_{epoch:02d}.hdf5"

		print('learn_rate = {}'.format(self.learn_rate))
		checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		callbacks_list = [checkpoint]

		self.model.fit(self.train_x, self.train_y, batch_size=128, epochs=20, callbacks=callbacks_list, verbose=1)

	def load_model(self, filename):
		'''
		Loads the weights of an identically architectured neural net at the given filepath
		'''
		self.model.load_weights(filename)
		adam = optimizers.Adam(lr=self.learn_rate, decay=.001)
		self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



	def predict_values(self, test_type='validation'):
		'''
		Predicts values based on training or validation data
		Return:
			scores
			predicted values
		'''
		# print(self.model.get_weights())
		if (test_type == 'validation'):
			scores = self.model.predict(self.test_x, verbose=True)
			return scores, self.test_y
		elif (test_type == 'training'):
			scores = self.model.predict(self.train_x, verbose=True)
			return scores, self.train_y

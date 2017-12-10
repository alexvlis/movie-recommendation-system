import numpy as np
import time
import util
from nn import Collaborative_Filtering_Neural_Net

def train():
	'''
	trains a neural net and saves snapshots every epoch. Runs 20 epochs or until you quit the process.
	'''
	train_mat, val_mat, masks = util.k_cross()
	A = util.load_data_matrix()

	start = time.time()
	net = Collaborative_Filtering_Neural_Net(train_mat[0], val_mat[0], masks[0])
	net.learn_rate=.1
	net.construct_model(hidden_layer_pattern = 'exponential')
	# net.load_model('nn_model_exponential_one_hot_learn_rate_.1_lr_0.1_04.hdf5')
	net.train_model(model_number='exponential_one_hot')
	print('time taken to train in seconds:', time.time() - start)

def test(model_name = '', test_type = 'validation'):
	'''
	Gets the accuracy and validation error of a model.
	This function assumes you have been saving your models
	'''
	train_mat, val_mat, masks = util.k_cross()
	A = util.load_data_matrix()

	net = Collaborative_Filtering_Neural_Net(train_mat[0], val_mat[0], masks[0])
	net.learn_rate=.1
	net.construct_model(hidden_layer_pattern = 'exponential')
	net.load_model(model_name)

	pred_scores , true_scores= net.predict_values(test_type = test_type)
	pred_scores = pred_scores.argmax(axis=1)
	true_scores    = true_scores.argmax(axis=1)

	#get Accuracy
	num_correct = np.sum(pred_scores == true_scores)
	accuracy    = num_correct/pred_scores.shape[0]*100

	#get MSE
	error = pred_scores-true_scores
	mse   = np.mean(np.power(error, 2))

	print('The {} accuracy of the model is {}%'.format(test_type, accuracy))
	print('The {} mean squared error of the model is {}'.format(test_type, mse))

if __name__ == '__main__':
	train()
	test('nn_model_exponential_one_hot_round_2_lr_0.1_08.hdf5', test_type='training')
	test('nn_model_exponential_one_hot_round_2_lr_0.1_08.hdf5', test_type='validation')
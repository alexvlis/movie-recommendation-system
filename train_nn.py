import numpy as np
import time
import util
from nn import Neural_Net

if __name__ == '__main__':
	train_mat, val_mat, masks = util.k_cross()
	A = util.load_data_matrix()

	net = Neural_Net(train_mat[0], val_mat[0], masks[0])

	start = time.time()

	net.train_model()
	print('time taken to train in seconds:', time.time() - start)

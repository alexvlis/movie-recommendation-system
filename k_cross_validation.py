import csv
import numpy as np
import pickle
import time

'''
The code herein will separate the A matrix into k separate data matrices, 
with 1/k of the values held as validation error:
A -> A1, A2, A3, A4, A5 ... Ak   training matrices
  -> Y1, Y2, Y3, Y4, Y5 ... Yk   validation matrices
'''


# Transplant every k'th + c nonzero value from a training matrix to validation matrix c
# Therefore, every k'th + c nonzero value in A should now be 0 in training matrix c
#                                          and now be A[i,j] in a validation matrix c


def k_cross(filename='data_matrix.p', path='', k=10):
	'''
	Create the training and validation matrices for k_cross validaton
	param:
		filename - name of pickle file with original (m x n) data matrix, must have file extension
		path     - path to the filename - can be omitted if path is included in filename
		k        - number of training/data sets
	returns:
		training_matrices   - array of (m x n) training matrices
		validation_matrices - array of (m x n) validation matrices where each data point is omitted from the corresponding 
							  training matrix
		index_matrices      - array of array of tuples. The tuples indicate indices of the kth training matrix where an element 
		                      was transplanted to a validation matrix
	'''

	filepath = filename if path == '' else '{}/{}'.format(path,filename)
	A = pickle.load( open('{}'.format(filepath), 'rb'))
	m = A.shape[0]
	n = A.shape[1]

	print('A.shape = {}'.format(A.shape))


	validation_matrices = []
	training_matrices   = []
	index_lists         = []
	for i in range(k):
	    A_copy = A.copy()
	    validation_matrices.append(np.zeros((m,n)))
	    training_matrices.append(A_copy)
	    index_lists.append([])

	it    = 0
	for i in range(A.shape[0]):
	    for j in range(A.shape[1]):
	        if (A[i,j] != 0):
	            training_matrices[it%k][i,j]   = 0
	            validation_matrices[it%k][i,j] = A[i,j]
	            index_lists[it%k].append((i,j))
	            it+=1

	return training_matrices, validation_matrices, index_lists







if __name__ == '__main__':
	'''
	This little test does test for the most important quality:

	This makes sure that the value in the validation matrix is not present in the
	test matrix.

	This is a bit overboard, but helps me sleep at night. 
	You really don't need to run this often.

	last I checked: ~25 second runtime on my toaster for k=10
	'''
	k = 10



	train_mats, val_mats, index_lists = k_cross(k=k)
	m = train_mats[0].shape[0]
	n = train_mats[0].shape[1]

	start = time.time()
	for i in range(m):
	    for j in range(n):
	        for index in range(k):
	            if(train_mats[index][i,j] != 0 and val_mats[index][i,j] != 0):
	                print('we have a problem')
	end = time.time()
	print('you wasted {} seconds of my life'.format(end-start))


import pickle
import numpy as np
import util

class CollaborativeFiltering():
	''' Collaborative Filtering Estimator '''

	@staticmethod
	def pearsonr(r_a, r_i):
		'''
		input: rating vectors of user a (active user) and user i
		output: pearsor correlation coefficient value
		'''

		# Get movies both users rated
		mask = np.equal(r_a>0, r_i>0)
		r_a = r_a[mask]
		r_i = r_i[mask]

		# Get mean rating for each user
		r_a_bar = np.mean(r_a)
		r_i_bar = np.mean(r_i)

		# Calculate pearson correlation coefficient
		return  np.dot(r_a-r_a_bar, r_i-r_i_bar) /         \
				np.sqrt(np.dot(r_a-r_a_bar, r_a-r_a_bar) * \
						np.dot(r_i-r_i_bar, r_i-r_i_bar))  \

	@staticmethod
	def significance(r_a, r_i, thresh):
		S = np.equal(r_a>0, r_i>0).sum()
		if S > thresh:
			return 1
		return S/thresh

	@staticmethod
	def prediction(r, w):
		'''
		input: neighborhood matrix of k rows, weight vector of neighborhood
		output: offset prediction vector for active user
		'''
		m = list(map(lambda x: np.mean(x), r))

		# FIXME:
		return np.dot((r.T - m), w) / np.sum(w)


	'''*************************** Class methods ****************************'''

	def __init__(self, method="neighborhood", k=10, s=50):
		self.method = method
		self.k = k
		self.s = s

	def fit(self, A):
		if self.method == "neighborhood":
			return self.neighborhood_based(A)
		if self.method == "item":
			return self.item_based(A)


	'''*************************** Private methods **************************'''

	def neighborhood_based(self, A):
		A_new = np.array(A) # copy A matrix

		for a, r_a in enumerate(A):
			# weight vector for active user a
			w = np.zeros(A.shape[0])
			w[a] = -1 # ignore active user

			for i, r_i in enumerate(A):
				if i == a:
					# Skip active user
					continue

				w[i] = CollaborativeFiltering.pearsonr(r_a, r_i) * \
						CollaborativeFiltering.significance(r_a, r_i, self.s)

			# Get indices of neighborhood
			K = np.argsort(w)[:self.k]

			mask = r_a==0
			A_new[a, mask] = (np.mean(r_a[r_a>0]) + CollaborativeFiltering.prediction(A[K], w[K]))[mask]

			print("user:", a, end='\r')

		return A_new

	def item_based(self, A):
		pass

if __name__ == "__main__":
	A = util.load_data_matrix()

	cf = CollaborativeFiltering()
	A_new = cf.fit(A)

	print(A, A.shape)
	print(A_new, A_new.shape)

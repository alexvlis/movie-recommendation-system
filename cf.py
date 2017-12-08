import pickle
import numpy as np
import util
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class CollaborativeFiltering():
	''' Collaborative Filtering Estimator '''

	@staticmethod
	def pearsonr(r_a, r_i):
		'''
		input: rating vectors of user a (active user) and user i
		output: pearsor correlation coefficient value
		'''

		# Get movies both users rated
		mask = np.logical_and(r_a>0, r_i>0)
		if mask.sum() == 0:
			return -1 # Does not actually matter what value we return

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
		'''
		input: rating vectors of user a (active user) and user i
		output: significance weight
		'''
		S = np.logical_and(r_a>0, r_i>0).sum()
		if S > thresh:
			return 1
		return S/thresh

	@staticmethod
	def prediction(r, w):
		'''
		input: neighborhood matrix of k rows, weight vector of neighborhood
		output: offset prediction vector for active user
		'''
		return np.dot((r.T - r.mean(axis=1)), w) / np.sum(w)


	'''*************************** Class methods ****************************'''

	def __init__(self, method="neighborhood", k=10, s=50):
		self.method = method
		self.k = k
		self.s = s

	def fit(self, A, verbose=False):
		self.verbose = verbose
		if self.verbose:
			print("Training...")

		if self.method == "neighborhood":
			return self.neighborhood_based(A)
		if self.method == "item":
			return self.neighborhood_based(A.T).T


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

			if self.verbose:
				print("fitting item:", a, end='\r')

		if self.verbose:
			print("\nDone.")

		A_new[A_new>5] = 5.0 # clip all ratings to 5
		return np.around(A_new*2)/2 # round to nearest .5


if __name__ == "__main__":
	# train_mats, val_mats, masks = util.k_cross(k=10)
	# cf = CollaborativeFiltering()

	# k_grid = range(1, 250, 50)
	# s_grid = range(1, 100, 10)

	# try:
	# 	for k in k_grid:
	# 		mse = list()
	# 		for s in s_grid:
	# 			# Set hyperparameters
	# 			cf.k = k
	# 			cf.s = s

	# 			# Stochastically select one batch per iteration
	# 			i = np.random.choice(len(train_mats))
	# 			train = train_mats[i]
	# 			mask = masks[i]

	# 			train_new = cf.fit(train)
	# 			error = util.get_MSE(train_new, mask)
	# 			print("MSE:", error, "parameters:", k, s)
	# 			mse.append(error)

	# 		plt.plot(s_grid, mse, label="k=" + str(k))
	# except KeyboardInterrupt:
	# 	pass

	# plt.legend()
	# plt.xlabel("s")
	# plt.ylabel("MSE Validation Error")
	# plt.savefig("cf.png", dpi=400)

	# # Set optimum hyperparameters
	# cf.k = 50
	# cf.s = 1

	# # Get the mean MSE over all the batches
	# e = 0
	# for train, mask, val in zip(train_mats, val_mats, masks):
	# 	train_new = cf.fit(train)
	# 	e += util.get_MSE(train_new, mask.astype(bool))

	# print("average MSE:", e/len(train_mats))

	A = util.load_data_matrix()
	cf = CollaborativeFiltering()
	A_new = cf.fit(A, verbose=True)
	recommendations = np.argsort(A_new[1, :])[:5]

	B = pickle.load( open('{}'.format('data/data_dicts.p'), 'rb'))

	for movie_id,rating in B['userId_rating'][2]:
	   if rating == 5 :
	       print(B['movieId_movieName'][movie_id] , ", rating:" , rating )

	l = recommendations
	k_list =[]
	for movie_column in l :
	   for k, v in B['movieId_movieCol'].items():
	       if v == movie_column:
	           k_list.append(k)
	print("")
	print("Recommendations")
	for movie_id in k_list :
	   print(B['movieId_movieName'][movie_id])

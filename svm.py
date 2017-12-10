# uses an SVM to predict movie ratings
import time
from sklearn import svm
import pandas as pd
import numpy as np
import util

class MovieSVM():


    def __init__(self, threshold, delta):
        self.threshold = threshold
        self.e = delta
        self.accuracy = []

    # fits the SVM according to the algorithm described
    def fit(self, A, V):
 
        # Create a negative of A 
        #N = self._buildNegative(A)
        #A = self._applyThreshold(A, self.threshold)

        T = np.copy(A)
        A = np.copy(A)
        for i in range(len(V)):
            for j in range(len(V[i])):
                if V[i, j] >= self.threshold:
                    V[i, j] = 1
                elif V[i, j] != 0:
                    V[i, j] = 0
                else:
                    V[i, j] = -1 
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i,j] >= self.threshold:
                    A[i,j] = 1
                    T[i,j] = 1
                elif A[i,j] != 0:
                    A[i,j] = 0
                    T[i,j] = 0
                else:
                    A[i,j] = np.random.randint(0, 2, size=1)[0]
                    T[i,j] = -1
        
        print(A.shape, np.count_nonzero(A==0), np.count_nonzero(A==1)) 
        print(T.shape, np.count_nonzero(T==0), np.count_nonzero(T==1)) 
        print(V.shape, np.count_nonzero(V==0), np.count_nonzero(V==1)) 
        #return []
        
        totalValidation = np.count_nonzero(V!=-1)
        iteration = 0
            
        svms = [ svm.SVC() for i in range(len(A[0]))  ]
        #print(len(svms))
        self.accuracy = []
        acc_prev, acc_k = 0, 2*self.e
        #print(np.delete(A, 1, axis=1).shape)    
        #print()
        self.train_accuracy = []
        while acc_k - acc_prev > self.e:
            train_correct = 0
            total_train = 0
            start_time = time.time()
            iteration += 1
            for i in range(len(svms)):
                print("fit:", str(i) + "/" + str(len(svms)), end='\r')
                X = np.delete(A, i, axis=1)
                Y = A[:,i]
        
                try:
                    svms[i].fit(X, Y)
                except:
                    dummy = 0
        
                A[:, i] = svms[i].predict(X)
                for j in range(len(A[:,i])):
                    if T[j,i] != -1 and A[j,i] != T[j,i]:
                        #A[j,i] = T[j,i]     
                        total_train += 1
                    elif T[j,i] != -1:
                        train_correct += 1   
                        total_train += 1
            self.train_accuracy.append((train_correct*1.0)/total_train)
            # calculate iteration accuracy
            countMatched = 0
            # go through each column, predict that column, check matching on V
            for i in range(len(svms)):
                print("Validate:", str(i) + "/" + str(len(svms)), end='\r')
                X = np.delete(V, i, axis=1)
                #print(X.shape)
                Y = V[:,i]
                Yhat = svms[i].predict(X)
                countMatched += np.sum(Yhat==Y)
            acc_prev = acc_k
            acc_k = (countMatched*1.0)/totalValidation
            
            self.accuracy.append(acc_k)
            print("\n - Iteration:", iteration, "\n - With Accuracy:", acc_k*100, "\n Difference:", acc_k - acc_prev, "\n Time (seconds):", time.time() - start_time)
        return self.accuracy, self.train_accuracy

                
        # while acc_k - acc_k-1 > e:
            # for each column, i
                # compute svm   -- X = [A[:,i-1],A[i+1,:]], Y = A[:,i] 
                # store svm in svms[i]
                # predict values
                # replace A[:,i] = predictions
            
            # predict values for each test point
            # acc_k = accuracy
            
    def countCorrect(self,T, A):
        count = np.sum(T==A)
        return count


    #def countTrue(A, N)

    # creates a matrix of the same shape as A
    #  All positions of values in A are replaced with -1
    #  All 0 positions in A are replaced with randomly chosen 0, 1
    def _buildNegative(self, A, thresh):
        #N = np.copy(A)
        N = np.random.randint(2, size=A.shape)
        N = N - 5*A
        N[N < 0] = -1
        return N

    def _applyTreshold(A, thresh):
        A[A >= thresh] = 2*thresh
        A[A < thresh] = 0
        return A

if __name__== "__main__":
    NUM_MOVIES = 1000
    Data = util.load_data_matrix()
    A = Data[:400, :NUM_MOVIES]
    movieSVM = MovieSVM(3.5, .01)
    V = Data[401:, :NUM_MOVIES]
    v_non_zero = np.count_nonzero(V)
    for i in range(len(A[0]) - 1, 0, -1):
        if np.count_nonzero(A[:,i]) == 0:
            A = np.delete(A, i, axis=1)
            V = np.delete(V, i, axis=1)
    
    accuracy, train_accuracy = movieSVM.fit(A, V)
    print("\n\n sparsity:", 1 - (v_non_zero*1.0)/(V.shape[0] * V.shape[1] * 1.0))
    print("\n\nFinished Accuracy Values:", accuracy)
    print("\n\nTraining Accuracy:", train_accuracy)


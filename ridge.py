# -*- coding: utf-8 -*-
"""

@author: Leo Laugier
"""
import numpy as np
import util

class Ridge():
    '''Weighted Ridge Regression'''
    def __init__(self, f = 200, alpha=40, lambd = 1, epsilon = 0.001):
        self.f = f
        self.alpha = alpha
        self.lambd = lambd
        self.epsilon = epsilon


    # Input: Rating matrix A
    # Hyperparameters: Vector Space Dimension: f ; alpha ; lambd
    def fit(self, A):
        k = len(A) #number of users
        n = len(A[0]) #number of movies
        # Construct the Preference matrix P
        # Construct the Confidence matrix C
        self.P = np.zeros((k,n))
        C = np.zeros((k,n))
        for u in range(k): # users
            for m in range(n): # movies
                if A[u][m]>0 : 
                    self.P[u][m] = 1
                else : 
                    self.P[u][m] = 0
                
                C[u][m] = 1 + self.alpha*A[u][m]
        print("P and C built")
        # Construct the C^u matrices and the  C^m matrices
        Cu = []
        Cm = []    
        for u in range(k):
            Cu.append(np.diag(C[u]))
            
        for m in range(n):
            Cm.append(np.diag(C[:,m]))
        print("Cu and Cm built")
        
        # Initialize X
        self.X = np.ones((k,self.f))
        Xprev = self.X+2*self.epsilon*np.ones((k,self.f))
        
        # Initialize Y
        self.Y = np.ones((self.f,n))
        Yprev = self.Y+2*self.epsilon*np.ones((self.f,n))
        
        while max(np.linalg.norm(self.X-Xprev) , np.linalg.norm(self.Y-Yprev) ) > self.epsilon:
            Xprev = self.X
            Yprev = self.Y
            for u in range(k): # users
                self.X[u] = np.transpose(np.dot( np.linalg.inv(np.dot(np.dot( self.Y , Cu[u]) , np.transpose(self.Y) ) + self.lambd*np.identity(self.f)) , np.dot(np.dot(self.Y, Cu[u]),self.P[u] )))
            
            for m in range(n): # movies
                self.Y[:,m] = np.dot( np.linalg.inv(np.dot(np.dot( np.transpose(self.X) , Cm[m]) , self.X ) + self.lambd*np.identity(self.f)) , np.dot(np.dot(np.transpose(self.X), Cm[m]),self.P[:,m] ))
        print("fitting done")
        
    # K is the number of recommended movies (taht have not been rated)   
    def predict(self, u, K = 5):
        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)
        k=0
        i = 0
        recommended_movies = []
        while k < K and i < len(indices) :
            if self.P[u][indices[i]] == 0 : 
                k += 1
                recommended_movies.append(indices[i])
            i += 1
            
        return recommended_movies
            
    
if __name__ == "__main__":
    A = util.load_data_matrix()
    # A = util.load_data_matrix()[:,:1000] # it's too painful for my laptop for all the movies 
    print(A, A.shape)
    r = Ridge()
    r.fit(A)
    recommendations = r.predict(1) # predicts the top K movies for user 1
    print(recommendations)

            

# -*- coding: utf-8 -*-
import numpy as np
import util
import matplotlib.pyplot as plt

class Ridge():
    '''Weighted Ridge Regression'''
    def __init__(self, f = 200, alpha=20, lambd = 1, epsilon = 0.001):
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
        self.X = np.random.rand(k,self.f)
        Xprev = self.X+2*self.epsilon*np.ones((k,self.f))
        
        # Initialize Y
        self.Y = np.random.rand(self.f,n)
        Yprev = self.Y+2*self.epsilon*np.ones((self.f,n))
        
        while max(np.linalg.norm(self.X-Xprev) , np.linalg.norm(self.Y-Yprev) ) > self.epsilon:
            Xprev = self.X
            Yprev = self.Y
            for u in range(k): # users
                self.X[u] = np.transpose(np.dot( np.linalg.inv(np.dot(np.dot( self.Y , Cu[u]) , np.transpose(self.Y) ) + self.lambd*np.identity(self.f)) , np.dot(np.dot(self.Y, Cu[u]),self.P[u] )))
            
            for m in range(n): # movies
                self.Y[:,m] = np.dot( np.linalg.inv(np.dot(np.dot( np.transpose(self.X) , Cm[m]) , self.X ) + self.lambd*np.identity(self.f)) , np.dot(np.dot(np.transpose(self.X), Cm[m]),self.P[:,m] ))
        print("fitting done")
        
    # K is the number of recommended movies   
    def predict(self, u, K = 300):
        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)
        
#        Recommended movies that have not been rated yet
#        k=0
#        i = 0
#        recommended_movies = []
#        while k < K and i < len(indices) :
#            if self.P[u][indices[i]] == 0 : 
#                k += 1
#                recommended_movies.append(indices[i])
#            i += 1
        
        recommended_movies = indices[:K].tolist()
            
        return recommended_movies
            
def rank(mat1, r):
    k = len(mat1) #number of users
    n = len(mat1[0]) #number of movies
    sum_numerator = 0
    sum_denominator = np.sum(mat1)
    for u in range(k):
        recommendations = r.predict(u)
        K = len(recommendations)
        rank_u = np.zeros(n)
        for m in range(n):
            if m in recommendations :
                rank_u[m] = recommendations.index(m)/(K-1)
        
        for m in range(n): 
            sum_numerator += mat1[u,m]*rank_u[m]
    
    return(sum_numerator / sum_denominator)
    
if __name__ == "__main__":
    
    '''Basic test of the algorithm'''
#    A = util.load_data_matrix()
#    A = util.load_data_matrix()[:,:100] # it's too painful for my laptop for all the movies 
#    print(A, A.shape)
#    r = Ridge()
#    r.fit(A)
#    recommendations = r.predict(1) # predicts the top K movies for user 1
#    print(recommendations)
    
    '''Choice of hyperparameters'''
    A = util.load_data_matrix()[:,:500]
    f_range = np.arange(100,400,20)
    ranks_f = []
    alpha_range = np.arange(10, 80, 10)
    ranks_alpha = []
    lambd_range = np.logspace(-1, 1, 10)
    ranks_lambd = []

    '''Choice of f'''
#    for f in f_range : 
#        r = Ridge(f)
#        r.fit(A)
#        x = rank(A, r)
#        ranks_f.append(x*100)
#        print(x)
#        
#    plt.plot(f_range,ranks_f)
#    plt.ylabel('expected percentile ranking (%)')
#    plt.xlabel('f')
#    plt.show()

    '''Choice of alpha'''
#    for alpha in alpha_range : 
#        r = Ridge(alpha = alpha)
#        r.fit(A)
#        x = rank(A, r)
#        ranks_alpha.append(x*100)
#        print(x)
#        
#    plt.plot(alpha_range,ranks_alpha)
#    plt.ylabel('expected percentile ranking (%)')
#    plt.xlabel('alpha')
#    plt.show()
    
    '''Choice of lambda'''
    for lambd in lambd_range : 
        r = Ridge(lambd = lambd)
        r.fit(A)
        x = rank(A, r)
        ranks_lambd.append(x*100)
        print(x)
        
    plt.semilogx(lambd_range,ranks_lambd)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('lambda')
    plt.show()
            

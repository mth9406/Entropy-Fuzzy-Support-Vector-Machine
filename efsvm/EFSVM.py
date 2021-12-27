from .utils import KNN_, Entropy, Kernel_

class EFSVM(object):

    def __init__(self,
                 k,
                 m,
                 beta,
                 C,
                 param=0,
                 type_='rbf'
                 ):
        
        self.k, self.m, self.beta = k,m,beta
        self.C, self.param = C, param
        self.type_ = type_
        self.support_vectors_ = None
        self.n_support_ = None
        
        # assert the beta's range
        if self.beta < 0 or self.beta > 1/(self.m-1):
            print("inappropriate beta")
        
    def fit(self, X, y):
        # param: rbf kernel's gamma
        # C: SVM param
        # k: neighbor 개수
        # m: subset 개수
        # beta: parameter  

        y = y.reshape(-1,1) *1. # column vector: matrix format.
        
        self.X, self.y = X, y

        # calculate neighbors
        self.neigh_, self.entropy_= KNN_(X, y, self.k)

        # divide the negative samples into m subsets
        # separating the negative samples 
        ## sub: size= m
        ## sub[l]= H_l: increasing order 

        sub= [[] for i in range(self.m)]
        FM= [1.0-self.beta*(l-1) for l in range(1,self.m+1)]
        H_= []
        for i,yv in enumerate(y):
            if yv == -1:
                H_.append((i,self.entropy_[i]))
        H_= np.array(H_,dtype=[('index',int),('entropy',float)])
        H_= np.sort(H_,order= 'entropy')
        H_min,H_max= H_[0][1],H_[-1][1]
        for l in range(self.m):
            thrUp, thrLow= H_min + l*(H_max-H_min)/self.m, H_min + (l-1)*(H_max-H_min)/self.m
            for i in range(len(H_)):
                if thrLow <= H_[i][1] < thrUp:
                    sub[l].append(H_[i][0])
        
        # assign fuzzy memberships s_i to each sample 
        s= np.ones(X.shape[0])
        for i in range(self.m):
            for idx in sub[i]:
                s[idx]= FM[i]

        # Adopt the SMO algorithm
        # SVM solver
        H = Kernel_(X,X,type=self.type_,params=self.param)*1.
        H *= y@y.T
        P= cvxopt_matrix(H)
        q= cvxopt_matrix(-np.ones((X.shape[0],1)))
        G= cvxopt_matrix(np.vstack((np.eye(X.shape[0]),-np.eye(X.shape[0]))))
        h= cvxopt_matrix(np.hstack((s*self.C,np.zeros(X.shape[0])))) 
        A= cvxopt_matrix(y.reshape(1,-1)) # 1, N
        b= cvxopt_matrix(np.zeros(1)) # 1


        sol= cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alphas= np.array(sol['x']).reshape(-1,1)
        self.S = ((self.alphas > 1e-4) & (self.alphas < self.C-1e-4)).flatten() # support vectors
        self.b = (y[self.S] - np.sum(Kernel_(X, X[self.S], type = self.type_,params= self.param)* y * self.alphas , axis = 0).reshape(-1,1))[0]       
        
    def predict(self, X_new):
        pred_sol= np.sign(np.sum(Kernel_(self.X, X_new, params= self.param,type= self.type_)* self.y * self.alphas, axis = 0).reshape(-1,1) + self.b)
        return pred_sol
    

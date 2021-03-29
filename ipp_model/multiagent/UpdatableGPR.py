import math
import numpy as np 
from numpy.linalg import inv
from .tools import euclidean, squared_euclidean
import copy

NAT_LOG_OF_TWO_PI = math.log(2*math.pi)

"""
Functionality

Inputs

Outputs
"""
def mrdivide(A, B):
    #out = A@inv(B)
    return A@inv(B)
    #return np.linalg.lstsq(B.T, A.T, rcond=-1)[0].T


"""
Functionality

Inputs

Outputs
"""
def diag_det(A):
    out = 1
    for x in range(len(A)):
        out = out * A[x,x]
    return out

"""
Functionality

Inputs

Outputs
"""
def log_diag_det(A):
    out = 0.0
    for x in range(len(A)):
        out = out + math.log(A[x,x])
    return out

class UpdatableGPR:

    """
    Functionality
        *to be written*

    Inputs
        loc: node locations (list of tuples of doubles)
        sig: per-cell standard deviation (double)
        ell: cross-cell correlation (double)

    Outputs
        n/a
    """
    def __init__(self, loc, sig, ell):
        self.loc = loc
        self.sig = sig
        self.ell = ell
        self.mu = None #len n vector
        self.val = None #len n vector
        self.cov = None #n-by-n matrix

        self.new_sample_needed = True
        self.recent_sample = None #len n vector

        self.generate()
        self.sample()


    """
    Functionality
        initialize both value vector, mean vector and covariance (kernel) matrix

    Inputs
        n/a

    Outputs
        n/a
    """
    def generate(self):
        n = len(self.loc)
        self.cov = (self.sig**2) * np.eye(n)

        if self.ell > 0:
            for i in range(n):
                iPos = self.loc[i]
                for j in range(i+1,n):
                    jPos = self.loc[j]
                    self.cov[i,j] = self.cov[j,i] = (self.sig**2) * math.exp(-1*euclidean(iPos, jPos)/self.ell)
        
        self.mu = np.full((n,1), 0.0)
        self.val = np.full((n,1), None)


    """
    Functionality

    Inputs

    Outputs
    """
    def sample(self):
        #if self.new_sample_needed:
        z = copy.deepcopy(self.val)
        a = self.unfound_values()#np.where(self.val == None)[0]
        n = len(a)
        rando = np.random.normal(size=(n,1))
        middle = np.linalg.cholesky(self.cov)@rando
        z[a] = self.mu + middle
        self.recent_sample = z
        #self.new_sample_needed = False

    """
    Functionality
        updates the value vector, mean vector and covariance (kernel) matrix given a list of samples

    Inputs
        data: new samples that have been obtained, this will be contained in a list of 2 lists
              the first list will contain ids (integer between 0 and len(self.loc)) and the second list will contain measurements (floating point values)
              the length of the lists much match

    Outputs
        nothing
    """
    def update(self, zB):
        n = len(self.loc)
        zB = np.array(zB)
        zB = zB[zB[:,0].argsort()]
        
        #Identify hidden components pre-update
        A0 = self.unfound_values()

        #Ignore observations seen before
        B, Iz, IB = np.intersect1d(zB[:,0].astype(int), A0, return_indices=True)
        zB = zB[Iz]

        if zB.shape[0] != 0:
            #Identify hidden components post-update
            IA = self.set_diff_indexes(A0,B)
            
            TauAB = mrdivide(self.cov[np.ix_(IA,IB)], self.cov[np.ix_(IB,IB)])
            
            self.cov = self.cov[np.ix_(IA,IA)] - TauAB@self.cov[np.ix_(IB,IA)]

            if len(zB[0]) == 2:
                newZ = np.reshape(zB[:,1], (zB[:,1].shape[0], 1))
                self.mu = self.mu[IA] + TauAB@(newZ-self.mu[IB])
                self.val[B.astype(int)] = newZ
            else:
                self.mu =  self.mu[IA]
                self.val[B.astype(int)] = self.mu[IB]
            self.sample()
    
    """
    Functionality
    
    Inputs

    Outputs
    """
    def predict(self, id):

        out = None
        if self.val[id][0] is not None:
            out = self.val[id][0]
        else:
            out = self.recent_sample[id][0]
        
        return out

    """
    Functionality

    Inputs

    Outputs
    """
    def variance(self, id):
        out = 0.0
        zB = np.array([(id, 0.0)])

        #Identify hidden components pre-update
        A0 = self.unfound_values()

        #Ignore observations seen before
        B, Iz, IB = np.intersect1d(zB[:,0].astype(int), A0, return_indices=True)
        zB = zB[Iz]

        if zB.shape[0] != 0:
            if self.val[id] == [None]:
                out = self.cov[IB[0], IB[0]] #MAKE SURE THAT THIS IS ACCURATE!!!

        return out


    """
    Functionality

    Inputs

    Outputs
    """
    def entropy(self, id, shift):
        var = self.variance(id)
        out = 0.5 * math.log(var*2*math.pi*math.e + shift)
        return out


    """
    Functionality

    Inputs

    Outputs
    """
    def mutual_metrics(self, id, method, shift):

        prioH = margH = postH = 0.0

        n = len(self.loc)
        zB = np.array([(id, 0.0)])

        #Identify hidden components pre-update
        A0 = self.unfound_values()

        #Ignore observations seen before
        B, Iz, IB = np.intersect1d(zB[:,0].astype(int), A0, return_indices=True)
        zB = zB[Iz]

        #Identify hidden components post-update
        IA = self.set_diff_indexes(A0,B)

        TauAB = mrdivide(self.cov[np.ix_(IA,IB)], self.cov[np.ix_(IB,IB)])
        post_cov = self.cov[np.ix_(IA,IA)] - TauAB@self.cov[np.ix_(IB,IA)]
        marg_cov = self.cov[np.ix_(IA,IA)]

        n0 = len(A0)
        m0 = n0 - 1

        if method == 'determinant':
            prioH = 0.5 * (n0 + n0*NAT_LOG_OF_TWO_PI + math.log(np.linalg.det(self.cov) + shift))
            margH = 0.5 * (m0 + m0*NAT_LOG_OF_TWO_PI + math.log(np.linalg.det(marg_cov) + shift))
            postH = 0.5 * (m0 + m0*NAT_LOG_OF_TWO_PI + math.log(np.linalg.det(post_cov) + shift))

        elif method == 'diagonal':
            prioH = 0.5 * (n0 + n0*NAT_LOG_OF_TWO_PI + log_diag_det(self.cov))
            margH = 0.5 * (m0 + m0*NAT_LOG_OF_TWO_PI + log_diag_det(marg_cov))
            postH = 0.5 * (m0 + m0*NAT_LOG_OF_TWO_PI + log_diag_det(post_cov))

        return prioH, margH, postH
    
    """
    Functionality

    Inputs

    Outputs
    """
    def unfound_values(self):
        return np.where(self.val == [None])[0]
    
    """
    Functionality

    Inputs

    Outputs
    """
    def set_diff_indexes(self, A0, B):
        diffs = np.setdiff1d(A0,B)
        IA = []
        for diff in diffs:
            IA.append(np.where(A0==diff)[0][0])
        return IA
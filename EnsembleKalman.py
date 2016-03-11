import pdb
import numpy as np
import numpy.linalg as lng
from AbstractFilter import AbstractFilter


class EnsembleKalmanFilter(AbstractFilter):

    def __init__(self, A, H, Q, R, x_0, B=0):
        AbstractFilter.__init__(self, A, H, Q, R, B)
        self.ens = np.matrix(x_0).T
        #pdb.set_trace()
        self.C = np.matrix(np.ones(self.dim))
        self.N = 1



    def _predict( self, z, u=0, return_cov=False, perm=True ):

        x_new, C_new = super(EnsembleKalmanFilter, self)._predict( z, u, return_cov=True )

        if perm:
            self.ens = x_new
            self.C = C_new
            self.u = u

        return x_new



    def _forecast( self, z, u=0 ):
        mu = np.zeros(self.dim)
        Sig = self.Q
        np.random.seed(self.N^2)
        w = np.mat(np.random.multivariate_normal(mu, Sig, 1)).T

        x_new_min = self.A * self.ens + w

        self.N = self.N + 1
        C_new = ((self.N-1)/self.N)*self.C + x_new_min * x_new_min.T/self.N

        return( x_new_min, C_new )



    def _correct( self, z, x_pred, P_pred ):
        mu = np.zeros(self.dim)
        Sig = self.R
        np.random.seed(self.N)
        v_k = np.mat(np.random.multivariate_normal(mu, Sig)).T
        
        K = P_pred * self.H.T * lng.inv( self.H * P_pred * self.H.T + self.R )

        x_k = x_pred + K * (np.mat(z).T + v_k - self.H * x_pred)
        return( x_k, P_pred )
        



    def getState( self, u=False ):
        if u:
            return x, P, u
        else:
            return x, P





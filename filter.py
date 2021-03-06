#from abc import ABCMeta, abstractmethod
import pdb
import numpy as np
import numpy.linalg as lng


class AbstractFilter:


    def __init__(self, A, H, Q, R, B=0, x_0=0, u_0=0):
        """
        Initializes the filter. Following the tutorial by Bishop and Welch
        the filter equations have the following form:

        x_k = A*x_k-1 + B*u_k-1 + w_k-1
        z_k = H*x_k + v_k

        where z is the true state of the process and z are available 
        measurements. Errors are assumed to be normal with mean 0 and
        covariances Q and R respectively.
        """

        if isinstance(x_0, int):
            self.x_0 = np.array([x_0])
            self.dim = 1
        else:
            self.dim = np.shape(x_0)[0]
    
        self.x = np.matrix(x_0).T
        self.u = np.matrix(u_0).T
        self.P = np.eye(self.dim)
        
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        if not B:
            self.B = np.matrix(np.zeros([self.dim, self.dim]))
        else:
            self.B = B





    def _predict( self, z, u=0, return_cov=False, perm=True ):
        """
        Peforms prediciton of the new state. By default this changes the
        internal state of the filter. If this is not deisred perm should
        be set to False.
        """
        x_new_min, P_new_min = self._update( z, u )
        x_new, P_new = self._correct( z, x_new_min, P_new_min )
        return x_new, P_new



    def _update( self, z, u ):
        raise NotImplementedError()



    def _correct( self, z, x_new_min, P_new_min ):
        raise NotImplementedError()



    def filtr(self, obs, inputs = None):
        """
        Returns a filtered sequence (i.e. the prediction of the true
        process based on supplied measurements and inputs
        """
        if inputs:
            data = zip(obs, inputs) if inputs else obs
            return np.array([self._predict(dp[0], dp[1]) for dp in data])
        else:
            return np.array([self._predict(dp) for dp in obs])





    def reset( self, x_0=0, u_0=0 ):
        """
        Resets the internal state of the filter. It should be used
        when the user wants to filter a new series using the same
        model
        """
        self.x_0 = x_0
        self.u_0 = u_0
        self.P = np.matrix(np.eye(self.dim))

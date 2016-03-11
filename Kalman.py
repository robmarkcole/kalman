import pdb
import numpy as np
import numpy.linalg as lng
from AbstractFilter import AbstractFilter


class KalmanFilter(AbstractFilter):

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
        AbstractFilter.__init__(self, A, H, Q, R, B)

        if self.dim>1:
            self.x_0 = np.array([x_0])

        self.x = np.matrix(x_0).T
        self.u = np.matrix(u_0).T
        self.P = np.eye(self.dim)





    def _forecast( self, z, u=0 ):
        x_new_min = self.A * self.x + self.B * self.u
        P_new_min = self.A * self.P * self.A.T + self.Q
        return( x_new_min, P_new_min )



    def _predict( self, z, u=0, return_cov=False, perm=True ):

        x_new, P_new = super(KalmanFilter, self)._predict( z, u, return_cov=True )

        if perm:
            self.x = x_new
            self.P = P_new
            self.u = u

        return x_new



    def _correct( self, z, x_pred, P_pred ):
        """
        Performs the measurement correction step. Should not be called from 
        outside of the class.
        """
        K = P_pred * self.H.T * lng.inv( self.H * P_pred * self.H.T + self.R )
        x_k = x_pred + K * (z - self.H * x_pred)
        P_k = (np.matrix(np.eye(self.dim)) - K*self.H) * P_pred
        #pdb.set_trace()
        return( x_k[0,0], P_k )
        



    def getState( self, u=False ):
        """
        Returns the internal state of the filter (defined by x and P)
        """
        if u:
            return x, P, u
        else:
            return x, P



    def reset( self, x_0=0, u_0=0 ):
        """
        Resets the internal state of the filter. It should be used
        when the user wants to filter a new series using the same
        model
        """
        self.x_0 = x_0
        self.u_0 = u_0
        self.P = np.matrix(np.eye(self.dim))


import matplotlib.pyplot as plt
import numpy as np
from kalman import KalmanFilter
import pdb



def generateYZ( seed, T, M, H, sig_w = 1, sig_eps = 0.5):
    """
    This function generates data from a linear state space model with measurement
    error. The model has the following form:

    y_t = A*y_t-1 + w_t
    z_t = H*y_t + e_t

    Both error terms (w, e) are assumed to be normal white noise
    """
    np.random.seed(seed)

    dim = M.shape[1]

    w = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), T)
    eps = np.random.multivariate_normal(np.zeros(dim), 0.5*np.eye(dim), T)

    y = np.zeros([T, dim])
    z = np.zeros([T, dim])

    for t in xrange(T):
        y[t] = np.dot(M,y[t-1]) + w[t]
        z[t] = np.dot(H,y[t]) + eps[t]

    return y, z






if __name__=="__main__":

    T = 100
    A = 0.5*np.matrix(np.eye(2))
    H = np.matrix(np.eye(2))

    y, z = generateYZ( 1988, T, A, H )

    KF = KalmanFilter(A, H, np.matrix(np.eye(2)), 0.5*np.matrix(np.eye(2)), x_0=np.zeros(2), u_0=np.zeros(2))

    y_filtered = KF.filtr(z)

    pdb.set_trace()


    plt.plot(y, 'b-')#, label="true process (y)")
    plt.plot(z, 'r--')#, label="measurements (z)")
    plt.plot(y_filtered, 'k-')#, label="KF prediction")
    plt.legend(["true process (y)", "measurements (z)", "KF prediction"])
    plt.show()





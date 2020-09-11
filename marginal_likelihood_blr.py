from scipy import linalg
import numpy as np

'''
This script calculates analytical (log) marginal likelihood function of simple-output Bayesian linear regression.
We do not use "reg.score_" because the current version of scikit-learn seems incorrect (see, https://github.com/scikit-learn/scikit-learn/issues/10748)
In this script, alpha is precision of weight, and lam is precision of noise.
'''

def log_marginalized_likelihood(alpha,lam,X,y):

    def AMatrix(alpha,lam,X):
        # eq 3.81 in PRML
        return alpha*np.eye(X.shape[1]) + lam*np.dot(X.T,X)

    def mnVector(alpha,lam,X,y):
        # eq 3.84 in PRML
        return lam * np.dot( linalg.inv(AMatrix(alpha,lam,X)), np.dot(X.T,y))

    def expectation_of_mnVector(alpha,lam,X,y):
        # eq 3.82 in PRML
        temp_mn=mnVector(alpha,lam,X,y)
        error=y-np.dot(X,temp_mn)
        first_term  = 0.5*lam*np.dot(error.T,error)
        second_term = 0.5*alpha*np.dot(temp_mn.T,temp_mn)
        return first_term, second_term

    # eq 3.86 in PRML
    temp1, temp2 = expectation_of_mnVector(alpha,lam,X,y)
    ret_val  = (0.5*X.shape[1]*np.log(alpha)
      + 0.5*X.shape[0]*np.log(lam)
     - temp1 - temp2
     - 0.5*np.log(linalg.det(AMatrix(alpha,lam,X)))
     - 0.5*X.shape[0]*np.log(2.*np.pi))
    return ret_val


def map_estimator(alpha,lam,X,y):
    def AMatrix(alpha,lam,X):
        # eq 3.81 in PRML
        return alpha*np.eye(X.shape[1]) + lam*np.dot(X.T,X)
    # eq 3.84 in PRML
    return lam * np.dot( linalg.inv(AMatrix(alpha,lam,X)), np.dot(X.T,y))


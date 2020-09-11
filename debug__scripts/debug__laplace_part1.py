import laplace_approximation as test_la
import numpy as np

'''
This script is for debuging parts of "laplace_approximation.py" by comparing the results of Bayesian linear regression. 
This script considers 1-dimensional input, 1-dimensional weight parameter, and 1-dimensional output.
'''


def so_model(x,w): # simple output linear regression model
    return x[0] * w[0]

w_true = np.array([10.])     # weight (parameter)
lam_true = np.array([100.0]) # noise precision (hyperparameter)
print("w_true =",w_true)
print("lam_true =",lam_true)



# training data
n=5
X = np.linspace(0., 1., n).reshape(n,1) # training input
print("X.shape =",X.shape)
if n <10:
    print("X =",X)


y=[] 
for i in range(0,n):
    v = so_model(X[i],w_true)
    v += np.random.normal(0.,1./np.sqrt(lam_true[0]))
    y.append([v]) # training output
y=np.array(y)
if n <10:
    print("y =",y)
print("y.shape =",y.shape)


prior_alpha = np.array([1.e-4]) # weight precision (hyperparameter)





# test 1 (map_estimation and marginal likelihood)
print("\n\n###  test 1  ###")
w_map = test_la.map_estimator(y,X,so_model,lam_true,prior_alpha,w_true)
print("approx_w_map     =",w_map) 
import marginal_likelihood_blr
print("analytical_w_map =",marginal_likelihood_blr.map_estimator(prior_alpha,lam_true,X,y).T[0]) # MAP estimator under true noise precision and prior weight precision
print("approx_log_ML      =", test_la.approximate_log_marginal_likelihood(y,X,w_map,so_model,lam_true,prior_alpha))
print("analytical_log_ML  =", marginal_likelihood_blr.log_marginalized_likelihood(prior_alpha,lam_true,X,y)[0])





# test 2 (empirical Bayes under Laplace approximation around previously obtained MAP estimation)
print("\n\n###  test 2  ###")
lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
eb_lam, eb_alpha  = test_la.empirical_bayes(y,X,w_map,so_model,lam_init,alpha_init)
from sklearn.linear_model import BayesianRidge
clf = BayesianRidge(fit_intercept=False,
                    copy_X=True,
                    alpha_1=0.,
                    alpha_2=0.,
                    lambda_1=0.,
                    lambda_2=0.,
                   )
clf.fit(X,y.reshape(n,))
print("empirical_bayes_alpha =",eb_alpha)
print("eb_alpha_sklearn      =",clf.lambda_)
print("empirical_bayes_lam =",eb_lam)
print("eb_lam_sklearn      =",clf.alpha_)
print("lam_true            =",lam_true)





# test 3 (Bayesian Regression ... repeating empirical Bayes and MAP estimation)
print("\n\n###  test 3  ###")
lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
w_init=0.1*w_true
w_map2, lam, alpha = test_la.maximized_approximate_log_marginal_likelihood(y,X,so_model,w_init,lam_init,alpha_init)
print("w_map          =",w_map2,", obtained under empirical Bayes hyperparameters.")
print("w_map (test 1) =",w_map ,", obtained under initial hyperparameters.")
print("w_map_sklearn  =",clf.coef_)
print("w_true         =",w_true)

print("lam      =",lam)
print("lam_true =",lam_true)

print("alpha =",alpha)





print("test finish!")

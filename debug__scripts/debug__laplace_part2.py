import laplace_approximation as test_la
import numpy as np

'''
This script is for debuging parts of "laplace_approximation.py" by comparing the results of Bayesian linear regression. 
This script considers 2-dimensional input, 2-dimensional weight parameter, and 1-dimensional output.
'''

key=0 # 0 (linear) or 1 (nonlinear)


def so_model(x,w):
    v1 = x[0] * w[0]       + x[1] * w[1]       # linear
    v2 = x[0] * w[0]* w[0] + x[1] * w[1]* w[1] # non-linear
    if 0==key:
        return v1
    else:
        return v2


w_true  = np.array([2., 5.])     # weight (parameter)
lam_true = np.array([1.]) # noise precision (hyperparameter)
if key>0:
    lam_true[1]=lam_true[0]
print("w_true =",w_true)
print("lam_true =",lam_true)


# training data
n=100
X = np.array( [np.linspace(0., 1., n), np.linspace(1., 3., n)]).T # training input
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






# test 1 (map estimation marginal likelihood)
print("\n\n###  test 1  ###")
w_map = test_la.map_estimator(y,X,so_model,lam_true,prior_alpha,w_true)
print("approx_w_map  =",w_map) 
#print("test_mo_ML =",test_la.approximate_log_marginal_likelihood(y,X,w_map,mo_model,lam_true,prior_alpha))
if 0==key:
    import marginal_likelihood_blr
    w_map_analytical=marginal_likelihood_blr.map_estimator(prior_alpha,lam_true,X,y).T[0]
    print("analytical_w_map=",w_map_analytical)
print("approx_log_ML      =", test_la.approximate_log_marginal_likelihood(y,X,w_map,so_model,lam_true,prior_alpha),", approximated around w_map")
if 0==key:
    print("approx_log_ML      =", test_la.approximate_log_marginal_likelihood(y,X,w_map_analytical,so_model,lam_true,prior_alpha),", approximated around w_map_analytical")
    print("analytical_log_ML  =", marginal_likelihood_blr.log_marginalized_likelihood(prior_alpha,lam_true,X,y)[0])


print("test finish!")

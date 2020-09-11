import laplace_approximation as test_la
import numpy as np

'''
This script is for debuging parts of "laplace_approximation.py" by comparing the results of Bayesian linear regression. 
This script considers 2-dimensional input, 2-dimensional weight parameter, and 2-dimensional output.
'''

key=1 # 0 (only for debug) or 1


def mo_model(x,w):
    v1 = x[0] * w[0]       + x[1] * w[1]
    v2 = x[0] * w[0]* w[0] + x[1] * w[1]* w[1]
    if 0==key:
        return v1, v1
    else:
        return v1, v2


w_true  = np.array([2., 5.])     # weight (parameter)
lam_true = np.array([1., 5.]) # noise precision (hyperparameter)
if 0==key:
    lam_true[1]=lam_true[0]
print("w_true =",w_true)
print("lam_true =",lam_true)



# training data
n=500
X = np.array( [np.linspace(0., 1., n), np.linspace(1., 3., n)]).T # training input
print("X.shape =",X.shape)
if n <10:
    print("X =",X)


y=[] 
for i in range(0,n):
    v = mo_model(X[i],w_true)
    v1, v2 =mo_model(X[i],w_true)
    v1 += np.random.normal(0.,1./np.sqrt(lam_true[0]))
    v2 += np.random.normal(0.,1./np.sqrt(lam_true[1]))
    if 0==key:
        y.append( [v1, v1] ) # training output
    else:
        y.append( [v1, v2] ) # training output
y=np.array(y)
print("y.shape =",y.shape)
if n<10:
    print("y =",y)


prior_alpha = np.array([1.e-5]) # weight precision (hyperparameter)





# test 1 (map estimation marginal likelihood)
print("\n\n###  test 1  ###")
w_map = test_la.map_estimator(y,X,mo_model,lam_true,prior_alpha,w_true)
print("approx_w_map  =",w_map) 
#print("test_mo_ML =",test_la.approximate_log_marginal_likelihood(y,X,w_map,mo_model,lam_true,prior_alpha))
if 0==key:
    import marginal_likelihood_blr
    X_double = np.vstack([X, X])
    y_double = np.hstack([y.T[0], y.T[0]]).reshape(2*n,1)
    w_map_analytical=marginal_likelihood_blr.map_estimator(prior_alpha,lam_true[0],X_double,y_double).T[0]
    print("analytical_w_map=",w_map_analytical)
print("approx_log_ML      =", test_la.approximate_log_marginal_likelihood(y,X,w_map,mo_model,lam_true,prior_alpha),", approximated around w_map")
if 0==key:
    print("approx_log_ML      =", test_la.approximate_log_marginal_likelihood(y,X,w_map_analytical,mo_model,lam_true,prior_alpha),", approximated around w_map_analytical")
    print("analytical_log_ML  =", marginal_likelihood_blr.log_marginalized_likelihood(prior_alpha,lam_true[0],X_double,y_double)[0])



# test 2 (empirical Bayes under Laplace approximation around previously obtained MAP estimation)
print("\n\n###  test 2  ###")
lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
eb_lam, eb_alpha  = test_la.empirical_bayes(y,X,w_map,mo_model,lam_init,alpha_init)
print("empirical_bayes_alpha =",eb_alpha)
print("empirical_bayes_lam =",eb_lam)
print("lam_true            =",lam_true)




# test 3 (Bayesian Regression ... repeating empirical Bayes and MAP estimation)
print("\n\n###  test 3  ###")
lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
w_init=0.1*w_true
w_map2, lam, alpha = test_la.maximized_approximate_log_marginal_likelihood(y,X,mo_model,w_init,lam_init,alpha_init)
print("w_map          =",w_map2,", obtained under empirical Bayes hyperparameters.")
print("w_map (test 1) =",w_map ,", obtained under initial hyperparameters.")
print("w_true         =",w_true)

print("lam      =",lam)
print("lam_true =",lam_true)

print("alpha =",alpha)


'''
print(X)
print(X_double)
y22a_double = np.vstack([y22a, y22a])
w2_map_analytical=marginal_likelihood_blr.map_estimator(prior_alpha,lam_true[0],X2_double,y22a_double).T[0]
print("w2_map_analytical=",w2_map_analytical)
print("analytical_marginal_likelihood given y22a = ", marginal_likelihood_blr.log_marginalized_likelihood(prior_alpha,lam_true[0],X2_double,y22a_double)[0])
print("test_so_marglike using w_map_analytical= ", test_la.so_log_approximate_marginal_likelihood(y22a_double,X2_double,w2_map_analytical,so_model_22a,lam_true[0],prior_alpha))
test_marginal_likelihood = test_la.mo_log_approximate_marginal_likelihood(y22_list,X2,w2_map_analytical,model_list,lam_true,prior_alpha)
print("[A] test_mo_marglike given analytical w_map_analytical=",test_marginal_likelihood)
test_marginal_likelihood = test_la.mo__log_approximate_marginal_likelihood(y22,X2,w2_map_analytical,mo_model_22,lam_true,prior_alpha)
print("[B] test_mo_marglike given analytical w_map_analytical=",test_marginal_likelihood)



from sklearn.linear_model import BayesianRidge
clf = BayesianRidge(fit_intercept=False,copy_X=True)
clf.fit(X2_double,y22a_double.reshape(2*n,))
print("eb_alpha_sklearn =",clf.lambda_)
print("eb_lam_sklearn  =",clf.alpha_)
print("w2_map_sklearn  =",clf.coef_)


lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
w_map=0.1*w_map
#w2_map=w2_map_analytical
eb_lam, eb_alpha = test_la.empirical_bayes(y,X,w_map,mo_model,lam_init,alpha_init)
print("[A] empirical_bayes_alpha =",eb_alpha)
print("[A] empirical_bayes_lam =",eb_lam)
print("w_map=",w_map)
print("lam_true =",lam_true)



# test 3 (comparing empirical Bayes)
print("\n\n###  test 3  ###")
lam_init=0.1*lam_true
alpha_init=0.1*prior_alpha
w_init=0.1*w2_true
w, lam, alpha = test_la.mo_maximized_log_approximate_marginal_likelihood(y22_list,X2,model_list,w_init,lam_init,alpha_init)
print("w = ",w)
print("lam = ",lam)
print("w2_true =",w2_true)
print("lam_true =",lam_true)
print("alpha = ",alpha)
'''

print("test finish!")

import math
import autograd.numpy as np
from autograd import hessian, grad
from scipy.optimize import minimize
import autograd.numpy.linalg as LA


'''
This script is for Bayesian (parametric) nonlinear regression using Laplace approximation.
The formulation are written in the supplementary manuscript.
'''

loop_num = 100

# common
ln_two_pi = np.log(2.*math.pi) # constant
regularization_coeff=1.0e-6


def optimization_eb_step(obj_func,w_init):

    # gradient-free optimization
    #res = minimize(obj_func, w_init, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

    # gradient-based optimization    
    df  = grad(obj_func)
    #res = minimize(obj_func, w_init, method='TNC', jac=df, options={'disp': False})
    ddf = hessian(obj_func)
    res = minimize(obj_func, w_init, method='trust-ncg', jac=df, hess=ddf, options={'disp': False})
    return res

def optimization_map_step(obj_func,w_init):

    # gradient-free optimization
    #res = minimize(obj_func, w_init, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

    # gradient-based optimization    
    df  = grad(obj_func)
    #res = minimize(obj_func, w_init, method='TNC', jac=df, options={'disp': False})
    ddf = hessian(obj_func)
    res = minimize(obj_func, w_init, method='trust-ncg', jac=df, hess=ddf, options={'disp': False})
    return res


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


def log_likelihood(y,X,w,model,lam):

    temp1=0.
    for i in range(y.shape[1]): # dimension of y (output)
        temp1 += ln_two_pi - np.log(lam[i])

    pred=[]
    for i in range(y.shape[0]): # number of data
        pred.append(model(X[i],w))
    pred=np.array(pred).reshape(y.shape[0],y.shape[1])
    edw = 0.
    for i in range(y.shape[1]): # dimension of y (output)
        error=y.T[i]-pred.T[i]
        edw += lam[i]*np.dot(error,error)

    return np.array([0.5 * ( - y.shape[0]*temp1  -  edw)])


def log_prior(w,alpha):
    log_p = 0.5* ( w.shape[0]*(np.log(alpha) - ln_two_pi) - alpha*np.dot(w.T,w))
    # regularization term
    for i in range(w.shape[0]): 
        log_p += regularization_coeff*np.log(w[i])
    return log_p


def map_estimator(y,X,model,lam,alpha,w_init):
    # optimize w_map under fixed lam and alpha.
    def obj_func(w):
        return -np.array(log_likelihood(y,X,w,model,lam) + log_prior(w,alpha))[0]
    res = optimization_map_step(obj_func,w_init.astype(float))
    return res.x


def hesse_w(y,X,w_map,model,lam,alpha):
    def temp_func(w):
        return -(log_likelihood(y,X,w,model,lam)+log_prior(w,alpha))
    temp_hesse=hessian(temp_func)
    return temp_hesse(w_map)[0]


def approximate_log_marginal_likelihood(y,X,w_map,model,lam,alpha):
    # see supplementary manuscript for details
    log_ml = log_likelihood(y,X,w_map,model,lam) + log_prior(w_map,alpha)
    log_ml += 0.5*w_map.shape[0]*ln_two_pi
    log_ml -= 0.5*np.log(   LA.det( hesse_w(y,X,w_map,model,lam,alpha) )    )
    return np.array( log_ml )


def empirical_bayes(y,X,w_map,model,lam_init,alpha_init):

    dimy=y.shape[1] # dimension of y (output)

    # caluculate hessian of square error sum at w_map for each output dimension, respectively.
    def temp_f(w):
        pred=[]
        for i in range(y.shape[0]): # number of data
            pred.append(model(X[i],w))
        pred=np.array(pred).reshape(y.shape[0],y.shape[1])
        ses_list=[]
        for i in range(dimy):
            error=y.T[i]-pred.T[i]
            ses_list.append(0.5*np.dot(error,error))
        return np.array(ses_list)
    mat_func = hessian(temp_f)
    hesse_ses = mat_func(w_map)

    temp_diag = np.eye(w_map.shape[0])
    for i in range(0,w_map.shape[0]):
        temp_diag[i,i] = regularization_coeff/(w_map[i]*w_map[i])

    temp_alpha=alpha_init
    temp_lam  =lam_init

    def obj_common(lam,alpha):
        # 0th order
        ret_val = (log_likelihood(y,X,w_map,model,lam) +log_prior(w_map,alpha))[0]
        
        # 2nd order
        #ret_val += 0.5*w_map.shape[0]*ln_two_pi # not related to optimization
        temp_mat2 = lam[0]*hesse_ses[0]
        for i in range(1,dimy):
            temp_mat2 += lam[i]*hesse_ses[i]
        det_hesse = LA.det(temp_mat2 + alpha[0]*np.eye(w_map.shape[0])+temp_diag)
        ret_val  -= 0.5*np.log(det_hesse)
        return -ret_val

    '''
    # optimize alpha under fixed w_map and lam.
    def obj_func2(z):
        lam=temp_lam
        alpha=z
        return obj_common(lam,alpha)
    res = optimization_eb_step(obj_func2,temp_alpha.astype(float))
    temp_alpha=res.x
   
    # optimize lam under fixed w_map and alpha.
    def obj_func3(z):
        lam=z
        alpha=temp_alpha
        return obj_common(lam,alpha)
    res = optimization_eb_step(obj_func3,temp_lam.astype(float))
    temp_lam=res.x
    '''
    # optimize lam and alpha under fixed w_map.
    def obj_func(z):
        lam=z[:dimy]
        alpha=z[dimy:]
        return obj_common(lam,alpha)
    z_init=np.concatenate([temp_lam, temp_alpha], axis=0).astype(float)
    res = optimization_eb_step(obj_func,z_init)
    temp_lam  =res.x[:dimy]
    temp_alpha=res.x[dimy:]
    #'''
    return temp_lam, temp_alpha


def maximized_approximate_log_marginal_likelihood(y_input,X_input,model,w_init,lam_init,alpha_init,warmup_num=20):
    temp_w = w_init
    temp_lam = lam_init
    temp_alpha = alpha_init
    from copy import deepcopy
    y=deepcopy(y_input)
    X=deepcopy(X_input)
    prev_ml = approximate_log_marginal_likelihood(y,X,temp_w,model,temp_lam,temp_alpha)
    print("temp_w, temp_lam, temp_alpha = ", temp_w, temp_lam, temp_alpha)
    print("approximate_marginal_likelihood =", prev_ml)
    for i in range(0,loop_num):
        if i<warmup_num:
            y=deepcopy(y_input[:50,:])
            X=deepcopy(X_input[:50,:])
        else:
            y=deepcopy(y_input)
            X=deepcopy(X_input)
        #print("Start MAP estimation")
        temp_w = map_estimator(y,X,model,temp_lam,temp_alpha,temp_w)   # map 
        print(i,"temp_w, temp_lam, temp_alpha = ", temp_w, temp_lam, temp_alpha)
        #print("Start empirical Bayes")
        temp_lam, temp_alpha = empirical_bayes(y,X,temp_w,model,temp_lam,temp_alpha) # empirical bayes
        print(i,"temp_w, temp_lam, temp_alpha = ", temp_w, temp_lam, temp_alpha)
        temp_ml = approximate_log_marginal_likelihood(y,X,temp_w,model,temp_lam,temp_alpha)
        print(i,"approximate_marginal_likelihood =", temp_ml)
        if i>warmup_num and abs(temp_ml-prev_ml)<1.0e-4:
            break
        prev_ml = temp_ml

    return temp_w, temp_lam, temp_alpha

    
'''
def empirical_bayes(y,X,w_map,model,lam_init,alpha_init): # simple version (not computationally efficient)
        
    dimy=y.shape[1]
    def obj_func(z):
        lam=z[:dimy]
        alpha=z[dimy:]
        return -approximate_log_marginal_likelihood(y,X,w_map,model,lam,alpha)[0]
    z_init=np.concatenate([lam_init, alpha_init], axis=0).astype(float)
    res = optimization_eb_step(obj_func,z_init)
    return res.x[:dimy], res.x[dimy:]
#'''


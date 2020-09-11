import autograd.numpy as np
import math
from sklearn.linear_model import BayesianRidge
from baselines import logger
from numpy.linalg import inv
import laplace_approximation

import warnings
warnings.simplefilter('ignore')

import cartpole_real_model
dt=cartpole_real_model.dt
m1=cartpole_real_model.m1
#m2=cartpole_real_model.m2
#l =cartpole_real_model.l
cz=cartpole_real_model.cz
#cth=cartpole_real_model.cth
cth=0.
g =cartpole_real_model.g

max_force=cartpole_real_model.max_force
init_std=cartpole_real_model.init_std

def ceom(t,v,uw): # continuous-time equations of motion
    zdot = v[1]
    thdot = v[3]
    u=uw[0]
    m2=uw[1]
    l=uw[2]
    costh=np.cos(v[2])
    sinth=np.sin(v[2])
    newzdot  = (u  -  m2*l*thdot*thdot*sinth  +  m2*g*costh*sinth  -  cz*zdot  -  cth*thdot*costh/l                       )  /    (m1 + m2*sinth*sinth)
    newthdot = (u*costh  +  (m1+m2)*g*sinth  -  m2*l*thdot*thdot*costh*sinth  -  cz*zdot*costh - (m1+m2)*cth*thdot/(m2*l) )  / (l*(m1 + m2*sinth*sinth))
    return np.array([v[1],newzdot,v[3],newthdot])


dimw=2

#'''
# ODE solver (Runge Kutta 4)
def mo_model(x,w):
    v  = np.array([x[0], x[1], x[2], x[3]])
    uw = np.array([x[4], w[0], w[1]])
    k1 = ceom(0.,v, uw)
    k2 = ceom(0.,v+k1*dt*0.5, uw )   
    k3 = ceom(0.,v+k2*dt*0.5, uw )   
    k4 = ceom(0.,v+k3*dt    , uw )
    v = (k1 + 2*(k2+k3) + k4) / 6.
    return v[0], v[1], v[2], v[3]
'''

# ODE solver (Modified Euler)
def mo_model(x,w):
    v  = np.array([x[0], x[1], x[2], x[3]])
    uw = np.array([x[4], w[0], w[1]])
    k1 = ceom(0.,v, uw)
    k2 = ceom(0.,v+k1*dt, uw )   
    v = (k1 + k2) / 2.
    return v[0], v[1], v[2], v[3]
#'''


#parameter_sampling_flag=0 # 0 ... every step parameter sampling  /  1 ... every episode parameter sampling (lazy sampling)  /  2 ... parameter sampling once (Bayesian DP)  /  3 ... posterior mean
#noiseless_flag=False
class CartPolePM():
    def __init__(self,
                 X,
                 Y,
                 init_state=None,
                 parameter_sampling_flag=0,
                 noiseless_flag=False,
                 ):
        self.dataXshape=X.shape
        self.dataYshape=Y.shape
        self.init_state_mean=np.array((0.,0.,np.pi,0.))
        '''
        if init_state is None:
            self.init_state_mean=np.array((0.,0.,np.pi,0.))
        else:
            self.init_state_mean=init_state
        '''

        w_init  = 0.1*np.ones(dimw)
        #lam_init = np.array([10000., 10000., 10000., 10000.])
        lam_init = np.array([100., 100., 100., 100.])
        alpha_init = np.array([5.])

        unload_flag=1
        import os
        if os.path.isfile("./result_apply/temp_param_cartpole.csv"):
            param_load = np.loadtxt('./result_apply/temp_param_cartpole.csv', delimiter=',')
            if abs(param_load[0]-self.dataXshape[0])<1:
                unload_flag=0
                w = np.array([param_load[1],param_load[2]])
                lam = np.array([param_load[3],param_load[4],param_load[5],param_load[6]])
                alpha = np.array([param_load[7]])
        if unload_flag:
            w, lam, alpha = laplace_approximation.maximized_approximate_log_marginal_likelihood(Y,X,mo_model,w_init,lam_init,alpha_init)

        param_save = np.array([self.dataXshape[0],w[0],w[1],lam[0],lam[1],lam[2],lam[3],alpha[0]])
        np.savetxt('./result_apply/temp_param_cartpole.csv', param_save, delimiter=',')

        self.prec_weight = alpha
        self.post_mean = w
        self.post_var  = inv( laplace_approximation.hesse_w(Y,X,w,mo_model,lam,alpha) )
        self.noise_var1 = 1./lam[0]
        self.noise_var2 = 1./lam[1]
        self.noise_var3 = 1./lam[2]
        self.noise_var4 = 1./lam[3]
        self.lam_memo = lam

        self.log_evidence_memo = laplace_approximation.approximate_log_marginal_likelihood(Y,X,w,mo_model,lam,alpha)

        self.parameter_sampling_flag=parameter_sampling_flag
        self.noiseless_flag=noiseless_flag
        if 2==self.parameter_sampling_flag:
            self.parameter_sample =  np.random.multivariate_normal(self.post_mean, self.post_var)
            print("self.parameter_sample =",self.parameter_sample)
        if 3==self.parameter_sampling_flag:
            self.parameter_sample =  self.post_mean

        self.sqrt_noise_var1=np.sqrt( self.noise_var1 )
        self.sqrt_noise_var2=np.sqrt( self.noise_var2 )
        self.sqrt_noise_var3=np.sqrt( self.noise_var3 )
        self.sqrt_noise_var4=np.sqrt( self.noise_var4 )

        if self.noiseless_flag:
            self.sqrt_noise_var1=0.
            self.sqrt_noise_var2=0.
            self.sqrt_noise_var3=0.
            self.sqrt_noise_var4=0.

    def k_fold_cv(self, X, Y, k=10):
        indx = np.array([i for i in range(self.dataXshape[0])])
        indx = np.random.permutation(indx)
        indx_subset = np.split(indx,k)

        ret_val = 0.
        for ki in range(k):
            temp_ind = [i for i in range(X.shape[0]) if i not in indx_subset[ki]]
            temp_X = X[temp_ind]
            temp_Y = Y[temp_ind]

            w  = self.post_mean
            lam = self.lam_memo
            alpha = self.prec_weight

            w, lam, alpha = laplace_approximation.maximized_approximate_log_marginal_likelihood(temp_Y,temp_X,mo_model,w,lam,alpha,warmup_num=0)
            train_ml = laplace_approximation.approximate_log_marginal_likelihood(temp_Y,temp_X,w,mo_model,lam,alpha)
            total_ml = laplace_approximation.approximate_log_marginal_likelihood(Y,X,w,mo_model,lam,alpha)
            ret_val += total_ml - train_ml
            print(ki,"ave_cv",ret_val/(ki+1),"  temp_cv",total_ml - train_ml)
        logger.log("k =",k)
        logger.log("PM_CV =",ret_val/(1.*k))
        return ret_val/(1.*k)

    def log_evidence_of_new_data(self, X, Y):
        return laplace_approximation.approximate_log_marginal_likelihood(Y,X,self.post_mean,mo_model,self.lam_memo,self.prec_weight)

    def onestepdynamics(self, z_input, zdot_input, th_input, thdot_input, u_input):
        if 0==self.parameter_sampling_flag:
            temp_w = np.random.multivariate_normal(self.post_mean, self.post_var)
        if 1==self.parameter_sampling_flag or 2==self.parameter_sampling_flag or 3==self.parameter_sampling_flag:
            temp_w = self.parameter_sample
        v1, v2, v3, v4 = mo_model([z_input, zdot_input, th_input, thdot_input, u_input], temp_w )
        v1 += self.sqrt_noise_var1 * np.random.normal(0., 1.)
        v2 += self.sqrt_noise_var2 * np.random.normal(0., 1.)
        v3 += self.sqrt_noise_var3 * np.random.normal(0., 1.)
        v4 += self.sqrt_noise_var4 * np.random.normal(0., 1.)
        return (z_input + v1*dt),  (zdot_input + v2*dt), (th_input + v3*dt), (thdot_input + v4*dt)


    def custom_reset(self, envp):
        if 1==self.parameter_sampling_flag:
            self.parameter_sample = np.random.multivariate_normal(self.post_mean, self.post_var)
        envp.state = self.init_state_mean + envp.np_random.normal(scale=init_std, size=(4,))

    def wrap_env(self, envp):
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset

    def log_evidence(self):
        return self.log_evidence_memo

    def logger_parameter(self):
        logger.log("\ncartpole_pm (Bayesian Non-Linear Regression using Laplace Approximation)")
        logger.log("dataX.shape =",self.dataXshape)
        logger.log("dataY.shape =",self.dataYshape)

        logger.log("precision of weight =",self.prec_weight)
        logger.log("post_mean =",self.post_mean)
        logger.log("post_var =",self.post_var)
        logger.log("noise_var1 =",self.noise_var1)
        logger.log("noise_var2 =",self.noise_var2)
        logger.log("noise_var3 =",self.noise_var3)
        logger.log("noise_var4 =",self.noise_var4)

        logger.log("log_evidence =",self.log_evidence())
        logger.log("init_state_mean =",self.init_state_mean)
        logger.log("parameter_sampling_flag =",self.parameter_sampling_flag)
        logger.log("noiseless_flag =",self.noiseless_flag)
        logger.log("dt =",dt)
        logger.log("m1 =",m1)
        logger.log("g =",g)
        logger.log("cz =",cz)
        logger.log("cth=",cth)
        logger.log("init_std =",init_std)


# test
if __name__ == '__main__':


    import gym, custom_gym, time
    env = gym.make('CustomCartPole-v0')

    real_dynamics = cartpole_real_model.CartPoleDynamics()
    real_dynamics.wrap_env(env.env)


    dn ="./data_debug/"
    fn1="debug_input.csv"
    fn2="debug_output.csv"
    cartpole_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    dataX = np.loadtxt(dn+fn1, delimiter=',')
    dataY = np.loadtxt(dn+fn2, delimiter=',')

    test_class = CartPolePM(dataX,dataY)
    test_class.wrap_env(env.env)
    test_class.logger_parameter()

    #'''
    episode_count=0
    while episode_count<1:
        ob = env.reset()
        env.env.state = np.array([0.,0.,0.,0.])
        print("init_state =",env.env.state)

        while True:
            #ac = env.action_space.sample()
            ac = np.array([0.])
            ob, rew, new, _ = env.step(ac)
            env.render()
            time.sleep(0.02)
            if new:
                episode_count +=1
                break
    #'''
    env.close()


from scipy import linalg
import numpy as np
from sklearn.linear_model import BayesianRidge
from baselines import logger
from marginal_likelihood_blr import log_marginalized_likelihood


parameter_sampling_flag=0 # 0 ... every step parameter sampling  /  1 ... every episode parameter sampling (lazy sampling)  /  2 ... parameter sampling once (Bayesian DP)


# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.
# This script is only for bebugging parametric regression using laplace approximation. This is not used for obtaining result in the paper and supp.



# remove
''' 
def blr(npx, npy):
    model1 = BayesianRidge( fit_intercept=False, # assume linear model, not affine model
                            copy_X=True)
    model1.fit(npx, npy)
    temp1 = log_marginalized_likelihood(model1.lambda_, model1.alpha_, npx, npy)
    return model1, temp1
''' 
# remove


#def so_model_11(x,w): # simple output linear regression model # add
#    th=x[0]    # add
#    thdot=x[1] # add
#    u=x[2]     # add
#    return ( w * np.sin(th + np.pi) + alpha3 * u ) # add

def so_model_11_minus(x,minus_w): # simple output linear regression model # add
    # The difference from the above is only the sign of weight parameter.
    # When using regularization term, parameters of interest should be positive.
    th=x[0]    # add
    thdot=x[1] # add
    u=x[2]     # add
    return ( -minus_w * np.sin(th + np.pi) + alpha3 * u ) # add
    

import pendulum_real_model
dt    =pendulum_real_model.dt
alpha3=pendulum_real_model.alpha3

class PendulumPM():
    def __init__(self,
                 X,
                 Y,
                 init_state=None,
                 ):
        self.datasize=X.shape
        if init_state is None:
            self.init_state_mean=np.array([np.pi, 0.])
        else:
            self.init_state_mean=init_state

        y1=Y[:,0] - X[:,1]
        self.noise_var1=np.dot(y1,y1)/y1.shape[0]
        self.log_evidence_y1 = - 0.5 * y1.shape[0] *( 1. + np.log( 2.*np.pi*(self.noise_var1)))

        # remove
        '''  
        blr_x = -np.sin(X[:,0]).reshape((Y[:,0].shape[0]),1) # x' = - sin(theta)
        blr_y = Y[:,1]- alpha3*X[:,2]                        # y' = y2 - alpha3*u
        self.blr_m, self.log_evidence_y2 = blr(blr_x, blr_y)

        self.post_mean   = self.blr_m.coef_
        self.post_var    = self.blr_m.sigma_[0]
        self.prec_weight = self.blr_m.lambda_
        self.noise_var2   = 1./self.blr_m.alpha_
        ''' 
        # remove

        beta_init=np.array([0.1])  # add
        alpha_init=np.array([0.1]) # add
        w_init=np.array([-2.])      # add

        import laplace_approximation # add
        #w, beta, alpha = laplace_approximation.maximized_approximate_log_marginal_likelihood(Y[:,1].reshape(X.shape[0],1),X,so_model_11,w_init,beta_init,alpha_init) # add
        #self.log_evidence_y2 = laplace_approximation.approximate_log_marginal_likelihood(Y[:,1].reshape(X.shape[0],1),X,w,so_model_11,beta,alpha)                    # add
        minus_w_init=-w_init      # add
        print("minus_w_init=",minus_w_init)
        minus_w, beta, alpha = laplace_approximation.maximized_approximate_log_marginal_likelihood(Y[:,1].reshape(X.shape[0],1),X,so_model_11_minus,minus_w_init,beta_init,alpha_init) # add
        w = -minus_w # add
        self.log_evidence_y2 = laplace_approximation.approximate_log_marginal_likelihood(Y[:,1].reshape(X.shape[0],1),X,minus_w,so_model_11_minus,beta,alpha)                    # add

        self.post_mean    = w # add
        #self.post_var     = 1./laplace_approximation.hesse_of_w(Y[:,1].reshape(X.shape[0],1),X,w,so_model_11,beta,alpha) # add
        self.post_var     = 1./laplace_approximation.hesse_w(Y[:,1].reshape(X.shape[0],1),X,minus_w,so_model_11_minus,beta,alpha) # add
        self.prec_weight  = alpha   # add
        self.noise_var2   = 1./beta # add

        self.post_var_square   = np.sqrt(self.post_var)
        self.noise_var1_square = np.sqrt(self.noise_var1)
        self.noise_var2_square = np.sqrt(self.noise_var2)

        self.thdot_clip_value=15. # To accelerate planning process, we use this clipping value. 
                                  # More large value or no clipping is also OK, while it requires more simulation samples for planning.
                                  # For main results, this value has no effect to the finally planned policy. The effect is only to limit the search space.

        if 2==parameter_sampling_flag:
            self.parameter_sample = self.post_mean + np.random.normal(0., self.post_var_square)


    def onestepdynamics(self, th, thdot, u):
        newth = th + (thdot +np.random.normal(0., self.noise_var1_square)) * dt

        if 0==parameter_sampling_flag:
            alpha1 = self.post_mean + np.random.normal(0., self.post_var_square)
        elif 1==parameter_sampling_mode or 2==parameter_sampling_mode:
            alpha1 = self.parameter_sample
        newthdot = (thdot + ( alpha1 * np.sin(th + np.pi) + alpha3 * u + np.random.normal(0., self.noise_var2_square) ) * dt)[0]
        return newth, newthdot


    def custom_reset(self, envp):
        if 1==parameter_sampling_flag:
            self.parameter_sample = self.post_mean + np.random.normal(0., self.post_var_square)
        returnv =self.init_state_mean + envp.np_random.normal(scale=1., size=2) # variance = scale^2
        return returnv


    def wrap_env(self, envp):
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset
        envp.max_speed   = self.thdot_clip_value

    def log_evidence(self):
        return self.log_evidence_y1 + self.log_evidence_y2


    def logger_parameter(self):
        #logger.log("\npendulum_pm (Bayesian Linear Regression)") 
        logger.log("\npendulum_pm_another (Bayesian Linear Regression using Laplace Approximation)") # add
        logger.log("thdot_clip_value =",self.thdot_clip_value)
        logger.log("alpha3 =",alpha3)
        logger.log("dataX.shape =",self.datasize)

        logger.log("precision of weight =",self.prec_weight)
        logger.log("post_mean =",self.post_mean)
        logger.log("post_var =",self.post_var)
        logger.log("noise_var1 =",self.noise_var1)
        logger.log("noise_var2 =",self.noise_var2)

        logger.log("log_evidence =",self.log_evidence())

        logger.log("init_state_mean =",self.init_state_mean)

        logger.log("parameter_sampling_flag =",parameter_sampling_flag)


# test
if __name__ == '__main__':


    import gym, custom_gym
    env = gym.make('CustomPendulum-v0')

    real_dynamics = pendulum_real_model.PendulumDynamics()
    real_dynamics.wrap_env(env.env)


    dn ="./data_debug2/"
    fn1="debug_input.csv"
    fn2="debug_output.csv"
    #pendulum_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    dataX = np.loadtxt(dn+fn1, delimiter=',')
    dataY = np.loadtxt(dn+fn2, delimiter=',')

    test_class = PendulumPM(dataX,dataY)
    test_class.wrap_env(env.env)
    test_class.logger_parameter()

    import pendulum_pm # add
    test2_class = pendulum_pm.PendulumPM(dataX,dataY) # add
    test2_class.wrap_env(env.env)  # add
    test2_class.logger_parameter() # add  

    ''' # add
    episode_count=0
    while episode_count<1:
        env.reset()
        env.env.state= np.array([[np.pi],[0.]])

        while True:
            ac = env.action_space.sample()
            ob, rew, new, _ = env.step(ac)
            env.render()
            if new:
                episode_count +=1
                break

    ''' # add
    env.close()


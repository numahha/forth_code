import numpy as np
from sklearn.linear_model import BayesianRidge
from baselines import logger
from marginal_likelihood_blr import log_marginalized_likelihood

import warnings
warnings.simplefilter('ignore')

parameter_sampling_flag=0 # 0 ... every step parameter sampling  /  1 ... every episode parameter sampling (lazy sampling)  /  2 ... parameter sampling once (Bayesian DP)  /  3 ... posterior mean



def blr(npx, npy):
    model1 = BayesianRidge( fit_intercept=False, # assume linear model, not affine model
                            copy_X=True)
    model1.fit(npx, npy)
    temp1 = log_marginalized_likelihood(model1.lambda_, model1.alpha_, npx, npy)
    return model1, temp1


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

        #logger.log("self.valid_score =",self.k_fold_cv(X, Y))

        y1=Y[:,0] - X[:,1]
        self.noise_var1=np.dot(y1,y1)/y1.shape[0]
        self.log_evidence_y1 = - 0.5 * y1.shape[0] *( 1. + np.log( 2.*np.pi*(self.noise_var1)))

        blr_x = -np.sin(X[:,0]).reshape((Y[:,0].shape[0]),1) # x' = - sin(theta)
        blr_y = Y[:,1]- alpha3*X[:,2]                        # y' = y2 - alpha3*u
        self.blr_m, self.log_evidence_y2 = blr(blr_x, blr_y)

        self.post_mean   = self.blr_m.coef_
        self.post_var    = self.blr_m.sigma_[0]
        self.prec_weight = self.blr_m.lambda_
        self.noise_var2   = 1./self.blr_m.alpha_

        self.post_var_square   = np.sqrt(self.post_var)
        self.noise_var1_square = np.sqrt(self.noise_var1)
        self.noise_var2_square = np.sqrt(self.noise_var2)

        self.thdot_clip_value=15. # To accelerate planning process, we use this clipping value. 
                                  # More large value or no clipping is also OK, while it requires more simulation samples for planning.
                                  # For main results, this value has no effect to the finally planned policy. The effect is only to limit the search space.

        if 2==parameter_sampling_flag:
            self.parameter_sample = self.post_mean + self.post_var_square*np.random.normal(0., 1.)
            print("self.parameter_sample =",self.parameter_sample)
        if 3==parameter_sampling_flag:
            self.parameter_sample =  self.post_mean

    def k_fold_cv(self, X, Y, k=10):
        indx = np.array([i for i in range(X.shape[0])])
        indx = np.random.permutation(indx)
        indx_subset = np.split(indx,k)
        ret_val = 0.

        for ki in range(k):
            temp_ind = [i for i in range(X.shape[0]) if i not in indx_subset[ki]]
            temp_X = X[temp_ind]
            temp_Y = Y[temp_ind]

            temp_y1=temp_Y[:,0] - temp_X[:,1]
            temp_noise_var1=np.dot(temp_y1,temp_y1)/temp_y1.shape[0]
            temp_log_evidence_y1 = - 0.5 * temp_y1.shape[0] *( 1. + np.log( 2.*np.pi*(temp_noise_var1)))

            temp_blr_x = -np.sin(temp_X[:,0]).reshape((temp_Y[:,0].shape[0]),1) # x' = - sin(theta)
            temp_blr_y = temp_Y[:,1]- alpha3*temp_X[:,2]                        # y' = y2 - alpha3*u
            temp_blr_m, temp_log_evidence_y2 = blr(temp_blr_x, temp_blr_y)

            train_ml = temp_log_evidence_y1 + temp_log_evidence_y2

            y1=Y[:,0] - X[:,1]
            total_log_evidence_y1 = - 0.5 * ( y1.shape[0] * np.log( 2.*np.pi*(temp_noise_var1)) + np.dot(y1,y1)/temp_noise_var1)

            blr_x = -np.sin(X[:,0]).reshape((Y[:,0].shape[0]),1) # x' = - sin(theta)
            blr_y = Y[:,1]- alpha3*X[:,2]                        # y' = y2 - alpha3*u
            total_log_evidence_y2 = log_marginalized_likelihood(temp_blr_m.lambda_, temp_blr_m.alpha_, blr_x, blr_y)
            
            total_ml = total_log_evidence_y1 + total_log_evidence_y2
            ret_val += total_ml - train_ml
            print(total_ml-train_ml,total_ml,train_ml)
        logger.log("k=",k)
        logger.log("PM_CV =",ret_val/(1.*k))
        return ret_val/(1.*k)

    def log_evidence_of_new_data(self, X, Y):
        y1=Y[:,0] - X[:,1]
        temp_log_evidence_y1 = - 0.5 * ( y1.shape[0] * np.log( 2.*np.pi*(self.noise_var1)) + np.dot(y1,y1)/self.noise_var1)

        blr_x = -np.sin(X[:,0]).reshape((Y[:,0].shape[0]),1) # x' = - sin(theta)
        blr_y = Y[:,1]- alpha3*X[:,2]                        # y' = y2 - alpha3*u
        temp_log_evidence_y2 = log_marginalized_likelihood(self.blr_m.lambda_, self.blr_m.alpha_, blr_x, blr_y)
        return temp_log_evidence_y1 + temp_log_evidence_y2


    def onestepdynamics(self, th, thdot, u):
        newth = th + (thdot +self.noise_var1_square*np.random.normal(0., 1.)) * dt

        if 0==parameter_sampling_flag:
            alpha1 = self.post_mean + self.post_var_square*np.random.normal(0., 1.)
        elif 1==parameter_sampling_flag or 2==parameter_sampling_flag or 3==parameter_sampling_flag:
            alpha1 = self.parameter_sample
        newthdot = (thdot + ( alpha1 * np.sin(th + np.pi) + alpha3 * u + self.noise_var2_square*np.random.normal(0., 1.) ) * dt)[0]
        return newth, newthdot


    def custom_reset(self, envp):
        if 1==parameter_sampling_flag:
            self.parameter_sample = self.post_mean + self.post_var_square*np.random.normal(0., 1.)
        returnv =self.init_state_mean + envp.np_random.normal(scale=.1, size=2) # variance = scale^2
        return returnv


    def wrap_env(self, envp):
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset
        envp.max_speed   = self.thdot_clip_value

    def log_evidence(self):
        return self.log_evidence_y1 + self.log_evidence_y2


    def logger_parameter(self):
        logger.log("\npendulum_pm (Bayesian Linear Regression)")
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


    dn ="./data_debug/"
    fn1="debug_input.csv"
    fn2="debug_output.csv"
    pendulum_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    dataX = np.loadtxt(dn+fn1, delimiter=',')
    dataY = np.loadtxt(dn+fn2, delimiter=',')

    test_class = PendulumPM(dataX,dataY)
    test_class.wrap_env(env.env)
    test_class.logger_parameter()

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

    env.close()


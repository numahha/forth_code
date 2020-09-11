import GPy
import time
import numpy as np
from baselines import logger

import warnings
warnings.simplefilter('ignore')

def gpr(x, yi):
    kernel = GPy.kern.RBF(x.shape[1],ARD=True)
    model = GPy.models.GPRegression(x, yi, kernel)
    model.optimize(messages=False, max_iters=3e5)
    model.Gaussian_noise.variance.constrain_bounded(1e-10,1.)
    model.optimize(messages=False, max_iters=3e5)
    return model

def gpr_without_optimize(x, yi):
    kernel = GPy.kern.RBF(x.shape[1],ARD=True)
    model = GPy.models.GPRegression(x, yi, kernel)
    #model.optimize(messages=True, max_iters=3e5)
    return model

import cartpole_real_model
dt=cartpole_real_model.dt
init_std=cartpole_real_model.init_std

#def dim5_input(z_input, zdot_input, th_input, thdot_input, u_input):
#    return np.array([zdot_input, np.cos(th_input), np.sin(th_input), thdot_input, u_input])
def dim6_input(z_input, zdot_input, th_input, thdot_input, u_input):
    return np.array([z_input, zdot_input, np.cos(th_input), np.sin(th_input), thdot_input, u_input])


#sampling_and_noise_flag=0 # 0 ... posterior sampling and gaussian noise  /  1 ... posterior mean and gaussian noise  /  2 ... posterior mean
class CartPoleNPM():
    def __init__(self,
                 X, # [z,zdot,th,thdot,u]
                 Y,
                 init_state=None,
                 sampling_and_noise_flag=0,
                 ):

        #logger.log("self.valid_score =",self.k_fold_cv(X, Y))

        X = np.insert(X, 3, np.sin(X[:,2]), axis=1) # [z,zdot,th,sinth,thdot,u]
        X[:,2] = np.cos(X[:,2])                     # [z,zdot,costh,sinth,thdot,u]
        #X=X[:,1:6]                                  # [zdot,costh,sinth,thdot,u] # if you want to use 5-dimensional input.

        #print(X[0,:]) # for debug
        self.gpr_m1 = gpr(X, Y[:,0].reshape((Y[:,0].shape[0]),1))
        self.gpr_m2 = gpr(X, Y[:,1].reshape((Y[:,1].shape[0]),1))
        self.gpr_m3 = gpr(X, Y[:,2].reshape((Y[:,2].shape[0]),1))
        self.gpr_m4 = gpr(X, Y[:,3].reshape((Y[:,3].shape[0]),1))
        self.datasize=X.shape

        self.input_func = dim6_input

        self.sqrt_noise_var1=np.sqrt( self.gpr_m1.Gaussian_noise.variance.values )
        self.sqrt_noise_var2=np.sqrt( self.gpr_m2.Gaussian_noise.variance.values )
        self.sqrt_noise_var3=np.sqrt( self.gpr_m3.Gaussian_noise.variance.values )
        self.sqrt_noise_var4=np.sqrt( self.gpr_m4.Gaussian_noise.variance.values )

        self.init_state_mean=np.array([0., 0., np.pi, 0.])
        '''
        if init_state is None:
            self.init_state_mean=np.array([0., 0., np.pi, 0.])
        else:
            self.init_state_mean=init_state
        '''
        self.sampling_and_noise_flag=sampling_and_noise_flag
        if 0==self.sampling_and_noise_flag:
            self.onestepdynamics = self.onestepdynamics_posterior_sampling_with_noise
        if 1==self.sampling_and_noise_flag:
            self.onestepdynamics = self.onestepdynamics_posterior_mean_with_noise
        if 2==self.sampling_and_noise_flag:
            self.onestepdynamics = self.onestepdynamics_posterior_mean_without_noise

    def k_fold_cv(self, X, Y, k=10):
        indx = np.array([i for i in range(X.shape[0])])
        indx = np.random.permutation(indx)
        indx_subset = np.split(indx,k)
        X = np.insert(X, 3, np.sin(X[:,2]), axis=1) # [z,zdot,th,sinth,thdot,u]
        X[:,2] = np.cos(X[:,2])                     # [z,zdot,costh,sinth,thdot,u]
        ret_val = 0.

        total_m1 = gpr_without_optimize(X, Y[:,0].reshape((Y[:,0].shape[0]),1))
        total_m2 = gpr_without_optimize(X, Y[:,1].reshape((Y[:,1].shape[0]),1))
        total_m3 = gpr_without_optimize(X, Y[:,2].reshape((Y[:,2].shape[0]),1))
        total_m4 = gpr_without_optimize(X, Y[:,3].reshape((Y[:,3].shape[0]),1))

        for ki in range(k):
            temp_ind = [i for i in range(X.shape[0]) if i not in indx_subset[ki]]
            temp_X = X[temp_ind]
            temp_Y = Y[temp_ind]
            temp_m1 = gpr(temp_X, temp_Y[:,0].reshape((temp_Y[:,0].shape[0]),1))
            temp_m2 = gpr(temp_X, temp_Y[:,1].reshape((temp_Y[:,1].shape[0]),1))
            temp_m3 = gpr(temp_X, temp_Y[:,2].reshape((temp_Y[:,2].shape[0]),1))
            temp_m4 = gpr(temp_X, temp_Y[:,3].reshape((temp_Y[:,3].shape[0]),1))
            #temp_m1 = gpr(X, Y[:,0].reshape((Y[:,0].shape[0]),1))
            #temp_m2 = gpr(X, Y[:,1].reshape((Y[:,1].shape[0]),1))
            #temp_m3 = gpr(X, Y[:,2].reshape((Y[:,2].shape[0]),1))
            #temp_m4 = gpr(X, Y[:,3].reshape((Y[:,3].shape[0]),1))

            train_ml = temp_m1.log_likelihood() + temp_m2.log_likelihood() + temp_m3.log_likelihood() + temp_m4.log_likelihood() 
            total_m1[:] = temp_m1.param_array
            total_m2[:] = temp_m2.param_array
            total_m3[:] = temp_m3.param_array
            total_m4[:] = temp_m4.param_array
            total_ml = total_m1.log_likelihood() + total_m2.log_likelihood() + total_m3.log_likelihood() + total_m4.log_likelihood() 
            ret_val += total_ml - train_ml
            print(total_ml - train_ml)
        logger.log("k =",k)
        logger.log("NPM_CV =",ret_val/(1.*k))
        return ret_val/(1.*k)

    def onestepdynamics_posterior_sampling_with_noise(self, z_input, zdot_input, th_input, thdot_input, u_input):
        X = self.input_func(z_input, zdot_input, th_input, thdot_input, u_input)
        X=X[:,None].T
        # predict with noise, see GPy reference ``https://gpy.readthedocs.io/en/deploy/GPy.core.html''
        y1_mu, y1_sigma=self.gpr_m1.predict(X)
        y2_mu, y2_sigma=self.gpr_m2.predict(X)
        y3_mu, y3_sigma=self.gpr_m3.predict(X)
        y4_mu, y4_sigma=self.gpr_m4.predict(X)

        # posterior sampling and gaussian noise
        newz     = ( y1_mu + np.sqrt(y1_sigma)*np.random.normal(0.,1.) )*dt + z_input
        newzdot  = ( y2_mu + np.sqrt(y2_sigma)*np.random.normal(0.,1.) )*dt + zdot_input
        newth    = ( y3_mu + np.sqrt(y3_sigma)*np.random.normal(0.,1.) )*dt + th_input
        newthdot = ( y4_mu + np.sqrt(y4_sigma)*np.random.normal(0.,1.) )*dt + thdot_input
        return newz[0,0], newzdot[0,0], newth[0,0], newthdot[0,0]


    def onestepdynamics_posterior_mean_with_noise(self, z_input, zdot_input, th_input, thdot_input, u_input):
        X = self.input_func(z_input, zdot_input, th_input, thdot_input, u_input)
        X=X[:,None].T
        # predict with noise, see GPy reference ``https://gpy.readthedocs.io/en/deploy/GPy.core.html''
        y1_mu, y1_sigma=self.gpr_m1.predict(X)
        y2_mu, y2_sigma=self.gpr_m2.predict(X)
        y3_mu, y3_sigma=self.gpr_m3.predict(X)
        y4_mu, y4_sigma=self.gpr_m4.predict(X)

        # gaussian noise
        newz     = ( y1_mu + self.sqrt_noise_var1*np.random.normal(0.,1.) )*dt + z_input
        newzdot  = ( y2_mu + self.sqrt_noise_var2*np.random.normal(0.,1.) )*dt + zdot_input
        newth    = ( y3_mu + self.sqrt_noise_var3*np.random.normal(0.,1.) )*dt + th_input
        newthdot = ( y4_mu + self.sqrt_noise_var4*np.random.normal(0.,1.) )*dt + thdot_input
        print("hello2")
        return newz[0,0], newzdot[0,0], newth[0,0], newthdot[0,0]


    def onestepdynamics_posterior_mean_without_noise(self, z_input, zdot_input, th_input, thdot_input, u_input):
        X = self.input_func(z_input, zdot_input, th_input, thdot_input, u_input)
        X=X[:,None].T
        # predict with noise, see GPy reference ``https://gpy.readthedocs.io/en/deploy/GPy.core.html''
        y1_mu, y1_sigma=self.gpr_m1.predict(X)
        y2_mu, y2_sigma=self.gpr_m2.predict(X)
        y3_mu, y3_sigma=self.gpr_m3.predict(X)
        y4_mu, y4_sigma=self.gpr_m4.predict(X)

        # gaussian noise
        newz     =  y1_mu*dt + z_input
        newzdot  =  y2_mu*dt + zdot_input
        newth    =  y3_mu*dt + th_input
        newthdot =  y4_mu*dt + thdot_input
        return newz[0,0], newzdot[0,0], newth[0,0], newthdot[0,0]

    def custom_reset(self, envp):
        envp.state=self.init_state_mean + envp.np_random.normal(scale=init_std, size=(4,))


    def wrap_env(self, envp):
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset


    def log_evidence(self):
        return self.gpr_m1.log_likelihood() + self.gpr_m2.log_likelihood() + self.gpr_m3.log_likelihood() + self.gpr_m4.log_likelihood()


    def logger_parameter(self):
        logger.log("\ncartpole_npm (Gausssian Process Regression)")
        logger.log("dataX.shape =",self.datasize)

        logger.log("model_y1.kern.variance.values = "         ,self.gpr_m1.kern.variance.values)
        logger.log("model_y1.kern.lengthscale.values = "      ,self.gpr_m1.kern.lengthscale.values)
        logger.log("model_y1.Gaussian_noise.variance.values =",self.gpr_m1.Gaussian_noise.variance.values)
        logger.log("model_y2.kern.variance.values = "         ,self.gpr_m2.kern.variance.values)
        logger.log("model_y2.kern.lengthscale.values = "      ,self.gpr_m2.kern.lengthscale.values)
        logger.log("model_y2.Gaussian_noise.variance.values =",self.gpr_m2.Gaussian_noise.variance.values)
        logger.log("model_y3.kern.variance.values = "         ,self.gpr_m3.kern.variance.values)
        logger.log("model_y3.kern.lengthscale.values = "      ,self.gpr_m3.kern.lengthscale.values)
        logger.log("model_y3.Gaussian_noise.variance.values =",self.gpr_m3.Gaussian_noise.variance.values)
        logger.log("model_y4.kern.variance.values = "         ,self.gpr_m4.kern.variance.values)
        logger.log("model_y4.kern.lengthscale.values = "      ,self.gpr_m4.kern.lengthscale.values)
        logger.log("model_y4.Gaussian_noise.variance.values =",self.gpr_m4.Gaussian_noise.variance.values)

        logger.log("log_evidence =",self.log_evidence())

        logger.log("init_state_mean =",self.init_state_mean)

        logger.log("sampling_and_noise_flag =",self.sampling_and_noise_flag)
        logger.log("init_std =",init_std)


# test
if __name__ == '__main__':


    import gym, custom_gym
    env = gym.make('CustomCartPole-v0')
    real_dynamics = cartpole_real_model.CartPoleDynamics()
    real_dynamics.wrap_env(env.env)

    dn ="./data_debug/"
    fn1="debug_input.csv"
    fn2="debug_output.csv"
    cartpole_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    dataX = np.loadtxt(dn+fn1, delimiter=',')
    dataY = np.loadtxt(dn+fn2, delimiter=',')

    test_class = CartPoleNPM(dataX,dataY)
    test_class.wrap_env(env.env)
    test_class.logger_parameter()

    episode_count=0
    while episode_count<1:
        ob = env.reset()
        ####env.env.state= np.array([[np.pi],[0.]])

        while True:
            ac = env.action_space.sample()
            ob, rew, new, _ = env.step(ac)
            env.render()
            time.sleep(0.1)
            if new:
                episode_count +=1
                break

    env.close()


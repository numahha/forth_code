import numpy as np
from baselines import logger

# change switch 1, 2, 3


def custom_cost(th, thdot, u):

    ''' switch 1
    # cost function for main results.
    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    return angle_normalize(th)**2 + 0.1*thdot**2 + .001*(u**2)
    '''
    # cost function for comparison with PILCO
    costs = (1.-np.cos(th))**2 + (np.sin(th))**2
    return 1. - np.exp( - 2.*costs )
    #'''


def custom_reset_zerostate(envp):
    temp=envp.np_random.normal(scale=.1, size=2) # variance = scale^2
    temp[0] += np.pi
    return temp


dt=0.1

''' switch 2
# below is for main results
alpha1 = -3.0 
alpha2 = -0.0 # This is c*. Change this value !! The range in this paper is c* in [-0.5,0.0].
alpha3 = 1.
noisevar1=0.01
noisevar2=0.01
max_torque=1.0
'''
# below is for comparison with PILCO
g  = 9.82
m  = 1.
l  = 1.
alpha1 =-(m*l*g/2.)/((m*l**2)/3.)
alpha2 =-0.01/((m*l**2)/3.)
alpha3 =1./((m*l**2)/3.)
noisevar1=0.0001
noisevar2=0.01
max_torque=2.5
#'''



sqrt_noise_var1=np.sqrt(noisevar1)
sqrt_noise_var2=np.sqrt(noisevar2)

class PendulumDynamics():

    def __init__(self):
        self.thdot_clip_value=1000. # technical assumption for bounded state action space.

    def onestepdynamics(self, th_input, thdot_input, u_input):

        ''' switch 3
        # next state for main result
        newthdot = thdot_input + ( alpha1 * np.sin(th_input + np.pi) + alpha2 * thdot_input + alpha3 * u_input + sqrt_noise_var2*np.random.normal(0., 1.) ) * dt 
        newth    = th_input + (thdot_input + sqrt_noise_var1*np.random.normal(0., 1.)) * dt
        v = [newth, newthdot]
        '''
        # next state for comparison with PILCO
        def my_func(t,v,u): # for dopri5
            th=v[0]
            thdot=v[1]            
            newthdot = ( alpha1 * np.sin(th + np.pi) + alpha2 * thdot + alpha3 * u )
            newth = (thdot)
            return [newth,newthdot]
        from scipy.integrate import ode
        solver=ode(my_func)
        solver.set_integrator('dopri5').set_initial_value([th_input, thdot_input],0.0)
        solver.set_f_params(u_input)
        solver.integrate(dt)
        v=solver.y
        #'''

        return v[0], v[1]

    def custom_reset(self, envp):
        returnv = custom_reset_zerostate(envp)
        return returnv

    def custom_obs_pilco(self, envp):                               
        theta, thetadot = envp.state                               
        # PILCO assumes noiseless observation for planning but noisy observation for rollout.
        theta    += sqrt_noise_var1*envp.np_random.normal(0.,1.)
        thetadot += sqrt_noise_var2*envp.np_random.normal(0.,1.)
        return np.array([np.cos(theta), np.sin(theta), thetadot])   

    def wrap_env(self, envp):
        envp.cost        = custom_cost
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset
        envp.max_torque  = max_torque
        envp.max_speed   = self.thdot_clip_value
        envp.obs_func    = self.custom_obs_pilco

    def logger_parameter(self):
        logger.log("\npendulum_real")
        logger.log("thdot_clip_value =",self.thdot_clip_value)
        logger.log("alpha1 =",alpha1)
        logger.log("alpha2 =",alpha2)
        logger.log("alpha3 =",alpha3)
        logger.log("noisevar1 =",noisevar1)
        logger.log("noisevar2 =",noisevar2)
        logger.log("max_torque =",max_torque)



def generate_test_samples(dirname="./data_debug/",
                          filename1='debug_input.csv',
                          filename2='debug_output.csv'):

    import gym, custom_gym
    env = gym.make('CustomPendulum-v0')
    real_dynamics = PendulumDynamics()
    real_dynamics.wrap_env(env.env)
    logger.configure(dirname)
    real_dynamics.logger_parameter()

    logger.log("generate samples by random policy")
    episode_count=0
    inputdata=[]
    outputdata=[]


    while episode_count<1:
        ob = env.reset()
        env.env.state= np.array([[np.pi],[0.]])
        print("init_state =",env.env.state)
        th, thdot = env.env.state
        prevth, prevthdot = th, thdot

        while True:
            ac = env.action_space.sample()
            ac = np.clip(ac,-max_torque, max_torque)
            inputdata.append([th[0], thdot[0], ac])

            ob, rew, new, _ = env.step(ac)
            th, thdot = env.env.state
            outputdata.append([(th[0]-prevth[0])/dt, (thdot[0]-prevthdot[0])/dt])
            prevth, prevthdot = th, thdot
            env.render()
            if new or len(inputdata)>100:
                episode_count +=1
                break


    np_input=np.array(inputdata)
    np_output=np.array(outputdata)
    np.savetxt(dirname+filename1,np_input,delimiter=',')
    np.savetxt(dirname+filename2,np_output,delimiter=',')
    env.close()


if __name__ == '__main__':
    generate_test_samples()



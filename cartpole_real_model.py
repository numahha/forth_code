import numpy as np
import math
from baselines import logger


dt=0.1
m1=.5
m2=.2
l=.5
cz=0.0
cth=0.025 # modeling error
g=9.8

noisevar1=0.0001
noisevar2=0.0001
noisevar3=0.0001
noisevar4=0.0001

#noisevar1=0.0
#noisevar2=0.0
#noisevar3=0.0
#noisevar4=0.0

max_force=5.0

init_std=1.

sqrt_noise_var1=np.sqrt(noisevar1)
sqrt_noise_var2=np.sqrt(noisevar2)
sqrt_noise_var3=np.sqrt(noisevar3)
sqrt_noise_var4=np.sqrt(noisevar4)


def custom_reset_zerostate(envp):
    envp.state=envp.np_random.normal(scale=init_std, size=(4,))
    #envp.state = np.array([0.,0.,0.,0.])
    envp.state[2] += np.pi

def ceom(t,v,u): # continuous-time equations of motion
    zdot = v[1]
    thdot = v[3]
    costh=np.cos(v[2])
    sinth=np.sin(v[2])
    newzdot  = (u[0]  -  m2*l*thdot*thdot*sinth  +  m2*g*costh*sinth  -  cz*zdot  -  cth*thdot*costh/l                       )  /    (m1 + m2*sinth*sinth)
    newthdot = (u[0]*costh  +  (m1+m2)*g*sinth  -  m2*l*thdot*thdot*costh*sinth  -  cz*zdot*costh - (m1+m2)*cth*thdot/(m2*l) )  / (l*(m1 + m2*sinth*sinth))
    return np.array([v[1],newzdot,v[3],newthdot])

# ODE solver (Runge Kutta 4)
def delta_state(z_input, zdot_input, th_input, thdot_input, u_input):
    x = np.array([z_input, zdot_input, th_input, thdot_input])
    k1 = ceom(0.,x, [u_input])
    k2 = ceom(0.,x+k1*dt*0.5, [u_input])
    k3 = ceom(0.,x+k2*dt*0.5, [u_input])
    k4 = ceom(0.,x+k3*dt, [u_input])
    return x + ((k1 + 2*(k2+k3) + k4)*dt / 6.)

'''

# ODE solver (Modified Euler)
def delta_state(z_input, zdot_input, th_input, thdot_input, u_input):
    x = np.array([z_input, zdot_input, th_input, thdot_input])
    k1 = ceom(0.,x, [u_input])
    k2 = ceom(0.,x+k1*dt, [u_input])
    return x + ((k1 + k2)*dt / 2.)

#'''

class CartPoleDynamics():

    #def __init__(self):


    def onestepdynamics(self, z_input, zdot_input, th_input, thdot_input, u_input):
        v =delta_state(z_input, zdot_input, th_input, thdot_input, u_input)

        v[0] += sqrt_noise_var1 * np.random.normal(0., 1.) * dt
        v[1] += sqrt_noise_var2 * np.random.normal(0., 1.) * dt
        v[2] += sqrt_noise_var3 * np.random.normal(0., 1.) * dt
        v[3] += sqrt_noise_var4 * np.random.normal(0., 1.) * dt
        return v[0], v[1], v[2], v[3]


    def custom_reset(self, envp):
        custom_reset_zerostate(envp)


    def wrap_env(self, envp):
        envp.dynamics    = self.onestepdynamics
        envp.reset_state = self.custom_reset


    def logger_parameter(self):
        logger.log("\ncartpole_real")

        logger.log("dt =",dt)
        logger.log("m1 =",m1)
        logger.log("m2 =",m2)
        logger.log("l  =",l)
        logger.log("g  =",g)
        logger.log("cz =",cz)
        logger.log("cth=",cth)
        logger.log("noisevar1 =",noisevar1)
        logger.log("noisevar2 =",noisevar2)
        logger.log("noisevar3 =",noisevar3)
        logger.log("noisevar4 =",noisevar4)
        logger.log("max_force =",max_force)
        logger.log("init_std =",init_std)



def energy(z, zdot, th, thdot):
    kinetic_energy    = 0.5*(m1+m2)*zdot*zdot
    kinetic_energy   += 0.5*m2*( l*l*thdot*thdot  -  2.*l*zdot*thdot*np.cos(th) )
    potential_energy  = m2*l*g*np.cos(th)
    return kinetic_energy + potential_energy


def generate_test_samples(dirname="./data_debug/",
                          filename1='debug_input.csv',
                          filename2='debug_output.csv'):

    import gym, custom_gym, time
    env = gym.make('CustomCartPole-v0')
    real_dynamics = CartPoleDynamics()
    real_dynamics.wrap_env(env.env)
    logger.configure(dirname)
    real_dynamics.logger_parameter()

    logger.log("generate samples by random policy")
    episode_count=0
    inputdata=[]
    outputdata=[]

    while episode_count<1:
        ob = env.reset()
        print("init_state =",env.env.state)
        z, zdot, th, thdot = env.env.state
        prevz, prevzdot, prevth, prevthdot = z, zdot, th, thdot

        while True:
            # simulate onestep
            #ac = env.action_space.sample()
            #ac = np.clip(ac,-max_force, max_force)
            ac = np.array([0.])
            inputdata.append([z, zdot, th, thdot, ac])
            ob, rew, new, _ = env.step(ac)
            z, zdot, th, thdot = env.env.state
            print("energy =",energy(z, zdot, th, thdot))

            outputdata.append([(z-prevz)/dt, (zdot-prevzdot)/dt, (th-prevth)/dt, (thdot-prevthdot)/dt])
            prevz, prevzdot, prevth, prevthdot = z, zdot, th, thdot

            env.render()
            time.sleep(0.1)
            if new or len(inputdata)>50:
                episode_count +=1
                break


    np_input=np.array(inputdata)
    np_output=np.array(outputdata)
    np.savetxt(dirname+filename1,np_input,delimiter=',')
    np.savetxt(dirname+filename2,np_output,delimiter=',')
    np.savetxt(dirname+'current_state.csv',env.env.state,delimiter=',')
    env.close()


if __name__ == '__main__':
    generate_test_samples()

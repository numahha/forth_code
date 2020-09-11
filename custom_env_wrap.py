import numpy as np
from baselines import logger

import pendulum_real_model
import pendulum_pm
import pendulum_npm
import pendulum_real_model_pilco

import cartpole_real_model
import cartpole_pm
import cartpole_npm


'''
model_switch 
(both pendlum and cartpole)    0 ... true model  /  1 ... nonparametric model  /  2 ... parametric model  /  3 ... selected model
(only pendulum)                11 and 111 ... nonparametric model for comparizon with PILCO  
'''


def custom_pendulum_wrap(env,
                         model_switch=0, 
                         dirname="./data_debug/",
                         filename1="debug_input.csv",
                         filename2="debug_output.csv",
                         ):
    # 1 #
    if model_switch < 9:
        real_dynamics = pendulum_real_model.PendulumDynamics()
    else:
        real_dynamics = pendulum_real_model_pilco.PendulumDynamics()
    real_dynamics.logger_parameter()    
    real_dynamics.wrap_env(env)


    # 2 #
    if model_switch>0:
        dataX = np.loadtxt(dirname+filename1, delimiter=',')
        dataY = np.loadtxt(dirname+filename2, delimiter=',')
        if model_switch < 9:
            init_s=np.loadtxt(dirname+'current_state.csv',delimiter=',')
        else:
            init_s=None
    if 1==model_switch or 3==model_switch  or 4==model_switch or 11==model_switch or 111==model_switch:
        npm_class = pendulum_npm.PendulumNPM(dataX,dataY,init_state=init_s)
        npm_class.logger_parameter()
    if 2==model_switch or 3==model_switch or 4==model_switch:
        pm_class = pendulum_pm.PendulumPM(dataX,dataY,init_state=init_s)
        pm_class.logger_parameter()


    # 3 #
    best_fit_model = 0 
    if 1==model_switch or 2==model_switch or 5==model_switch:
        best_fit_model = model_switch
    if 3==model_switch:
        log_evidence = pm_class.log_evidence() - npm_class.log_evidence()
        if log_evidence>0.:
            best_fit_model = 2
        else:
            best_fit_model = 1
        logger.log("log_evidence_diff =",log_evidence)

    if 4==model_switch:
        npmcv =  npm_class.k_fold_cv(dataX,dataY,k=10)
        pmcv = pm_class.k_fold_cv(dataX,dataY,k=10)
        cv=pmcv-npmcv
        if cv>0.:
            best_fit_model = 2
        else:
            best_fit_model = 1
        

    # 4 #
    if 1==best_fit_model:
        npm_class.wrap_env(env)
    if 2==best_fit_model:
        pm_class.wrap_env(env)

    if 11==model_switch or 111==model_switch: # only for comparison with pilco
        npm_class.thdot_clip_value = 50.  # only for comparison with pilco 
        # In this case, the resulting real-world velocity exceeds the max_speed given for accelerating learning process for main results.
        npm_class.wrap_env_pilco(env) # only for comparison with pilco
        npm_class.logger_parameter()
        best_fit_model=model_switch


    logger.log("flag in custom_pendulum_wrap =",model_switch)
    logger.log("actually_selected_model=",best_fit_model)
    if (best_fit_model+model_switch)>0: 
        logger.log("init_state=",init_s)
    return best_fit_model


def custom_cartpole_wrap(env,
                         model_switch=0, 
                         dirname="./data_debug/",
                         filename1="debug_input.csv",
                         filename2="debug_output.csv",
                         ):
    # 1 #
    real_dynamics = cartpole_real_model.CartPoleDynamics()
    real_dynamics.logger_parameter()    
    real_dynamics.wrap_env(env)


    # 2 #
    if model_switch>0:
        dataX = np.loadtxt(dirname+filename1, delimiter=',')
        dataY = np.loadtxt(dirname+filename2, delimiter=',')
        #init_s= np.loadtxt(dirname+'current_state.csv',delimiter=',')
        init_s=None
    if 1==model_switch or 3==model_switch or 4==model_switch:
        npm_class = cartpole_npm.CartPoleNPM(dataX,dataY,init_state=init_s)
        npm_class.logger_parameter()
    if 2==model_switch or 3==model_switch or 4==model_switch:
        pm_class = cartpole_pm.CartPolePM(dataX,dataY,init_state=init_s)
        pm_class.logger_parameter()


    # 3 #
    best_fit_model = 0 
    if 1==model_switch or 2==model_switch:
        best_fit_model = model_switch
    if 3==model_switch:
        log_evidence = pm_class.log_evidence() - npm_class.log_evidence()
        if log_evidence>0.:
            best_fit_model = 2
        else:
            best_fit_model = 1
        logger.log("log_evidence_diff =",log_evidence)
    if 4==model_switch:
        npmcv =  npm_class.k_fold_cv(dataX,dataY,k=10)
        pmcv = pm_class.k_fold_cv(dataX,dataY,k=10)
        cv=pmcv-npmcv
        if cv>0.:
            best_fit_model = 2
        else:
            best_fit_model = 1


    # 4 #
    if 1==best_fit_model:
        npm_class.wrap_env(env)
    if 2==best_fit_model:
        pm_class.wrap_env(env)


    logger.log("flag in custom_cartpole_wrap =",model_switch)
    logger.log("actually_selected_model=",best_fit_model)
    if (best_fit_model+model_switch)>0: 
        logger.log("init_state=",init_s)
    return best_fit_model

def custom_wrap(env_id,
                env,
                model_switch=0, 
                dirname="./data_debug/",
                filename1="debug_input.csv",
                filename2="debug_output.csv",
                ):
    if env_id=="CustomPendulum-v0" or env_id=="CustomPendulum-v1":
        print("model_switch =",model_switch)
        best_fit_model = custom_pendulum_wrap(env,
                             model_switch=model_switch,
                             dirname=dirname,
                             filename1=filename1,
                             filename2=filename2)

    if env_id=="CustomCartPole-v0":
        best_fit_model = custom_cartpole_wrap(env,
                             model_switch=model_switch,
                             dirname=dirname,
                             filename1=filename1,
                             filename2=filename2)
    return best_fit_model

if __name__ == '__main__': # for debug

    import gym, custom_gym
    dn="./data_debug/"
    fn1="debug_input.csv"
    fn2="debug_output.csv"
 
    #'''
    env_id="CustomPendulum-v0"
    pendulum_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    '''
    env_id='CustomCartPole-v0'
    cartpole_real_model.generate_test_samples(dirname=dn,filename1=fn1,filename2=fn2)
    #'''

    env = gym.make(env_id)
    custom_wrap(env_id,env.env,
                model_switch=3, # change here !
                dirname=dn,filename1=fn1,filename2=fn2)

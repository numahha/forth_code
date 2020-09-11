import gym, custom_gym
import numpy as np
import argparse
import pendulum_real_model
import tensorflow as tf
from baselines import logger
from baselines.common.policies import build_policy
import baselines.common.tf_util as U

def apply_policy_to_sim_environment(timesteps,
                                     save_path,
                                     loop_num,
                                     num_hidden,
                                    sim_env_type):

    print("loop_num =",loop_num)
    env_id   = 'CustomPendulum-v0'
    env = gym.make(env_id)

    from custom_env_wrap import custom_wrap                #modify
    custom_wrap(env_id,                                    #modify
                env.env,                                   #modify
                model_switch=sim_env_type,                 #modify
                dirname="./result_apply/",                 #modify
                filename1="real_world_samples_input.csv",  #modify
                filename2="real_world_samples_output.csv") #modify

    episode_count=0
    inputdata=[]
    outputdata=[]
    costdata=[]

    dt=pendulum_real_model.dt
    env.reset()


    env.env.state= np.loadtxt('./result_apply/current_state.csv',delimiter=',').reshape(2,1)
    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
               allow_soft_placement=True,
               inter_op_parallelism_threads=cpus_per_worker,
               intra_op_parallelism_threads=cpus_per_worker
    ))
    logger.log("num_hidden =",num_hidden)
    policy = build_policy(env, "mlp", value_network='copy', num_hidden=num_hidden)
    with tf.variable_scope("pi"):
        pi = policy()
    U.initialize()
    pi.load("./result_apply/policy"+str(loop_num)+"/policy/")



    th, thdot = env.env.state
    prevth, prevthdot = th, thdot
    ob=env.env.obs_func(env.env)
    while True:
        # simulate onestep
        ac, vpred, _, _ = pi.step(ob, stochastic=True)
        #ac=np.array([1.0]) # constant control input for debug
        ac = np.clip(ac, -pendulum_real_model.max_torque, pendulum_real_model.max_torque)[0]
        inputdata.append([th[0], thdot[0], ac[0]])

        ob, rew, new, _ = env.step(ac)
        th, thdot = env.env.state
        th = [th]
        thdot = [thdot]
        print(th, thdot,prevth, prevthdot)
        outputdata.append([(th[0]-prevth[0])/dt, (thdot[0]-prevthdot[0])/dt])
        prevth, prevthdot = th, thdot

        costdata.append(rew)

        env.render()
        if len(outputdata)>=timesteps:
            break

    np_input=np.array(inputdata)
    np_output=np.array(outputdata)
    np_cost=np.array(costdata)
    np.savetxt(save_path+'simulation_samples_input.csv',np_input,delimiter=',')
    np.savetxt(save_path+'simulation_samples_output.csv',np_output,delimiter=',')
    np.savetxt(save_path+'simulation_cost.csv',np_cost,delimiter=',')
    print("np_cost.sum() =",np_cost.sum())
    np.savetxt(save_path+'simulation_total_cost.csv',np.array([np_cost.sum()]),delimiter=',',fmt ='%.6f')

    env.close()

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--timesteps', type=int, default=50)
    arg_parser.add_argument('--loop_num', type=int, default=0)
    arg_parser.add_argument('--num_hidden', type=int, default=32)
    arg_parser.add_argument('--sim_env_type', type=int, default=1)
    args = arg_parser.parse_args()
    save_path="./result_apply/policy"+str(args.loop_num)+"/simulation"+str(args.sim_env_type)+"/"
    logger.configure(save_path)
    apply_policy_to_sim_environment(timesteps=args.timesteps,
                                     save_path=save_path,
                                     loop_num=args.loop_num,
                                     num_hidden=args.num_hidden,
                                    sim_env_type=args.sim_env_type
                                     )

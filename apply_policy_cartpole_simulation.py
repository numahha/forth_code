import gym, custom_gym
import numpy as np
import argparse
import cartpole_real_model
import tensorflow as tf
from baselines import logger
from baselines.common.policies import build_policy
import baselines.common.tf_util as U
import time


def apply_policy_to_sim_environment(timesteps,
                                    save_path,
                                    loop_num,
                                    num_hidden,
                                    sim_env_type):

    print("loop_num =",loop_num)
    env_id   = 'CustomCartPole-v0'
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

    dt=cartpole_real_model.dt
    env.reset()


    env.env.state= np.array([0., 0., np.pi,0.])
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


    z, zdot, th, thdot = env.env.state
    prevz, prevzdot, prevth, prevthdot = z, zdot, th, thdot
    ob=env.env._get_obs()
    while True:
        # simulate onestep
        ac, vpred, _, _ = pi.step(ob, stochastic=True)
        ac=ac[0]
        ac = np.clip(ac,-cartpole_real_model.max_force, cartpole_real_model.max_force)
        print(z, zdot, th, thdot, ac)
        inputdata.append([z, zdot, th, thdot, ac])

        ob, rew, new, _ = env.step(ac)
        z, zdot, th, thdot = env.env.state
        outputdata.append([(z-prevz)/dt, (zdot-prevzdot)/dt, (th-prevth)/dt, (thdot-prevthdot)/dt])
        prevz, prevzdot, prevth, prevthdot = z, zdot, th, thdot


        costdata.append(rew)
        time.sleep(0.1)
        env.render()
        if len(outputdata)>=timesteps:
            print(np.array(costdata).sum())
            break
    np_input=np.array(inputdata)
    np_output=np.array(outputdata)
    np_cost=np.array(costdata)
    np.savetxt(save_path+'simulation_samples_input.csv',np_input,delimiter=',')
    np.savetxt(save_path+'simulation_samples_output.csv',np_output,delimiter=',')
    np.savetxt(save_path+'simulation_cost.csv',np_cost,delimiter=',')
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

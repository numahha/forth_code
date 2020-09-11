import gym
import custom_gym
import time
import numpy as np

#env_id='CustomPendulum-v0'
env_id='CustomCartPole-v0'


cartpole_zoomout=50


env = gym.make(env_id)
for j in range(1,10):
    data = np.loadtxt("./result_apply/real_world_samples_input_"+str(j)+".csv", delimiter=',')
    #cost = np.loadtxt("./result_apply/cost.csv", delimiter=',')
    env.reset()
    for i in range(data.shape[0]):
        #print(i,data[i], cost[i])
        print(j,i,data[i])
        env.env.state=data[i,:-1]
        if env_id=='CustomCartPole-v0':
            env.env.state[0] /= cartpole_zoomout
            #env.env.length_scale = 1./cartpole_zoomout
        env.render()
        time.sleep(0.05)
        #if new or (t>200):
        #    break

    #print("cost_sum =",cost.sum())

env.close()

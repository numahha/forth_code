import gym
import custom_gym
import time


env_switch  =2 #  -1 ... pendulum  /  -2 ... cartpole  /  1 ... custom_pendulum  /  2 ... custom_cartpole


if -1==env_switch:
    env = gym.make('Pendulum-v0')
if -2==env_switch:
    env = gym.make('CartPole-v0')
if 1==env_switch:
    env = gym.make('CustomPendulum-v0')
if 2==env_switch:
    env = gym.make('CustomCartPole-v0')



ob = env.reset()
#print(dir(env))
#print(dir(env.spec))
#print("max_episode_steps =",env.spec.max_episode_steps)

t=0
while True:
    env.render()
    time.sleep(0.01)
    ac = env.action_space.sample()
    print("t =",t)
    print("  ob =",ob)
    print("  s  =",env.env.state)
    print("  ac =",ac)
    ob, rew, new, _ = env.step(ac)
    t += 1
    if new or (t>200):
        break



env.close()

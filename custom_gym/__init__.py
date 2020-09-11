import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CustomPendulum-v0',
    entry_point='custom_gym.envs:CustomPendulumEnv',
    #timestep_limit=200,
    max_episode_steps=200,
)

register(
    id='CustomPendulum-v1', # this is only for comparison with PILCO
    entry_point='custom_gym.envs:CustomPendulumEnv2',
    #timestep_limit=40,
    max_episode_steps=40,
)

register(
    id='CustomCartPole-v0',
    entry_point='custom_gym.envs:CustomCartPoleEnv',
    max_episode_steps=50,
    #timestep_limit=200,
    #reward_threshold=195.0,
)


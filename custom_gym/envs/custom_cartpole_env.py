"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CustomCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        '''
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        '''

        self.max_force=5.0
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        high = np.array([
            1000.1,
            1000.1,
            1.,
            1.,
            1000.1])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.dynamics = default_dynamics
        self.cost = default_cost
        self.reset_state = default_reset_state

        self.length_scale=1 # only for view

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        action = np.clip(action, -self.max_force, self.max_force)[0]
        z, z_dot, theta, theta_dot = self.state
        self.state = self.dynamics(z, z_dot, theta, theta_dot, action)

        '''
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)E                
        
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        '''

        done = False
        cost = self.cost(z, z_dot, theta, theta_dot, action)
        return self._get_obs(), -cost, done, {}

    def _get_obs(self):
        z, z_dot, theta, theta_dot = self.state
        #return np.array(self.state)
        return np.array(( z, z_dot, np.cos(theta), np.sin(theta), theta_dot))

    def reset(self):
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        #return np.array(self.state)
        self.reset_state(self)
        return self._get_obs()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        #world_width = self.x_threshold*2
        world_width = 2.4*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.length_scale
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART

        if cartx>scale+screen_width:
            cartx = cartx - screen_width
        if cartx>scale+screen_width:
            cartx = cartx - screen_width
        if cartx<0.:
            cartx = cartx + screen_width
        if cartx<0.:
            cartx = cartx + screen_width


        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
def default_dynamics(x, x_dot, theta, theta_dot, action):
    self_gravity = 9.8
    self_masscart = 1.0
    self_masspole = 0.1
    self_total_mass = (self_masspole + self_masscart)
    self_length = 0.5 # actually half the pole's length
    self_polemass_length = (self_masspole * self_length)
    #self_force_mag = 10.0
    self_tau = 0.02  # seconds between state updates
    self_kinematics_integrator = 'euler'
    #force = self_force_mag if action==1 else -self_force_mag
    force = action
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + self_polemass_length * theta_dot * theta_dot * sintheta) / self_total_mass
    thetaacc = (self_gravity * sintheta - costheta* temp) / (self_length * (4.0/3.0 - self_masspole * costheta * costheta / self_total_mass))
    xacc  = temp - self_polemass_length * thetaacc * costheta / self_total_mass
    if self_kinematics_integrator == 'euler':
        x  = x + self_tau * x_dot
        x_dot = x_dot + self_tau * xacc
        theta = theta + self_tau * theta_dot
        theta_dot = theta_dot + self_tau * thetaacc
    else: # semi-implicit euler
        x_dot = x_dot + self_tau * xacc
        x  = x + self_tau * x_dot
        theta_dot = theta_dot + self_tau * thetaacc
        theta = theta + self_tau * theta_dot
    return np.array((x,x_dot,theta,theta_dot))

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def default_cost(z, zdot, th, thdot, u):
    #return 1 - np.cos(th)
    return angle_normalize(th)**2 + 0.*(thdot**2) + 0.*(zdot**2) + 0.001*(z**2)

def default_reset_state(envp):
    envp.state = envp.np_random.uniform(low=-0.05, high=0.05, size=(4,))


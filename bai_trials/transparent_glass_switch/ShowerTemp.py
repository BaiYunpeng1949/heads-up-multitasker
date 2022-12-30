import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random


class ShowerTemp(Env):
    def __init__(self):
        """
        This is a pipeline-validity testing class.

        Reference:
            https://github.com/nicknochnack/ReinforcementLearningCourse/blob/main/Project%203%20-%20Custom%20Environment.ipynb
        """
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=0, high=100, shape=(1,))
        # Set start temp
        self.state = 38 + random.randint(-3, 3)
        # Set shower length
        self.shower_length = 60

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1
        # Reduce shower length by 1 second
        self.shower_length -= 1

        # Calculate reward
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1

        # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        # Reset shower time
        self.shower_length = 60
        return self.state


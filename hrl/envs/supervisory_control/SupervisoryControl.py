import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from collections import deque
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from stable_baselines3 import PPO

from huc.utils.rendering import Camera, Context
from huc.utils.write_video import write_video


class SupervisoryControl(Env):

    def __init__(self):
        """
        Model the supervisory control of attention allocation in the context of walking and reading from smart glasses.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize the beliefs - the reading progress belief and walking progress belief
        self._reading_belief = None
        self._walking_belief = None
        self._reading_weight = None
        self._walking_weight = None


    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def step(self, action):
        pass

    def _get_obs(self):
        pass

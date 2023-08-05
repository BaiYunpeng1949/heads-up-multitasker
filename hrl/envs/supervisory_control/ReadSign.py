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

from hrl.utils.rendering import Camera, Context
from hrl.utils.write_video import write_video
from hrl.envs.supervisory_control.OcularMotorControl import OcularMotorControl
from hrl.envs.supervisory_control.LocomotionControl import LocomotionControl


class ReadSign(Env):

    def __init__(self):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def step(self, action):
        pass

    def _get_obs(self):
        pass

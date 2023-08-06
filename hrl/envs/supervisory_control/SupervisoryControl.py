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
from hrl.envs.supervisory_control.WordSelection import WordSelection
from hrl.envs.supervisory_control.ReadSign import ReadSign


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
        self._reading_task_belief = None
        self._walking_task_belief = None
        self._reading_task_weight_range = [0, 1]
        self._reading_task_weight = None
        self._walking_task_weight = None
        self._reading_task_termination_threshold = 300   # The number of cells/words that needed to be read

        # Initialize the walking task environment related parameters - TODO maybe determine more metrics
        self._SMART_GLASSES = -1
        self._BACKGROUND = 1
        self._attention = None
        self._prev_attention = None
        # Step-wise background information update frequency levels
        self._bg_info_freq_lvs = {
            'low': 20,
            'medium': 10,
            'high': 5,
        }
        self._bg_info_freq_lv = None
        self._rho_bg_info_freq = 0.1
        self._bg_info_freq = None
        self._bg_info_freq_noise = None

        # Initialize the RL training related parameters
        self._steps = None
        self.ep_len = 200
        self._epsilon = 1e-100

        # Define the observation space
        self._num_stateful_info = 16
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space - the attention allocation: reading on smart glasses (0) or reading on the environment (1)
        self.action_space = Box(low=-1, high=1, shape=(1,))

        # Initialize the pre-trained middle level RL models when testing the supervisory control
        if self._config['rl']['mode'] == 'test':
            # Initialize the RL middle level task environments
            self._read_sg_env = WordSelection()
            self._read_bg_env = ReadSign()

            # Load the pre-trained RL middle level task models - reading on smart glasses
            read_sg_checkpoints_dir_name = ""
            read_sg_loaded_model_name = ""
            read_sg_model_path = os.path.join(root_dir, 'training', 'saved_models',
                                              read_sg_checkpoints_dir_name, read_sg_loaded_model_name)
            self._read_sg_model = PPO.load(read_sg_model_path)
            self._read_sg_tuples = None
            self._read_sg_params = None
            self.read_sg_imgaes = None

            # Load the pre-trained RL middle level task models - reading on the background/environment
            read_bg_checkpoints_dir_name = ""
            read_bg_loaded_model_name = ""
            read_bg_model_path = os.path.join(root_dir, 'training', 'saved_models',
                                              read_bg_checkpoints_dir_name, read_bg_loaded_model_name)
            self._read_bg_model = PPO.load(read_bg_model_path)
            self._read_bg_tuples = None
            self._read_bg_params = None
            self.read_bg_imgaes = None

    def reset(self):

        self._steps = 0

        # For the test mode, evaluate with the pre-trained RL middle level task models
        if self._config['rl']['mode'] == 'test':
            pass
        # For the training mode, only focus on this model, deploy a lot of stochasticity
        else:
            # Randomly initialize the attention allocation
            self._attention = np.random.choice([self._SMART_GLASSES, self._BACKGROUND])
            self._prev_attention = self._attention

            # Randomly initialize the background information update frequency level
            self._bg_info_freq = np.random.choice(list(self._bg_info_freq_lvs.values()))
            levels = list(self._bg_info_freq_lvs.keys())
            self._bg_info_freq_lv = np.random.choice(levels)
            self._bg_info_freq = self._bg_info_freq_lvs[self._bg_info_freq_lv]

            # Randomly initialize the reading task weight and walking task weight - describes the perceived importance of the two tasks
            self._reading_task_weight = np.random.uniform(self._reading_task_weight_range[0],
                                                          self._reading_task_weight_range[1])
            self._walking_task_weight = 1 - self._reading_task_weight

            # Initialize the reading task belief and walking task belief
            # The reading task belief contains information including the cost of switching the task estimation -
            #   it describes how easily it is to pick up the word correctly after the attention switch,
            #   the higher the belief value, the easier it is to pick up the word
            self._reading_task_belief = 0
            # The walking task belief contains information including the possibility of the lane instruction has changed,
            #   the higher the belief value, the higher the possibility of the lane instruction has changed
            self._walking_task_belief = 0

        return self._get_obs()

    def render(self, mode="human"):
        pass

    def step(self, action):

        # Action a
        action = np.normalize(action, -1, 1, -1, 1)
        if action <= 0:
            self._attention = self._SMART_GLASSES
        else:
            self._attention = self._BACKGROUND

        self._steps += 1

        # State s'
        finish_reading = False
        # Continue with the previous task
        if self._prev_attention == self._attention:
            pass
        # Switch to the other task
        else:
            pass
            # TODO apply cost here, it can include: 1. the time cost of switching the attention, 2. the cost of facing risks of not picking up the word correctly

        # Update the background information updates according to the background information update frequency level
        # TODO

        # Update beliefs
        # TODO

        # Reward r
        # TODO define the reward function here - read as much as possible but also maintain the environmental awareness
        #   if missed the environmental awareness, then penalize the agent;
        #   if interrupt too much to the reading task, then penalize the agent
        reward_reading = 0  # TODO related to the reading progress and the
        reward_walking = 0  # TODO related to the actual environmental awareness
        reward = self._reading_task_weight * reward_reading + self._walking_task_weight * reward_walking

        # Termination
        terminate = False
        if self._steps >= self.ep_len or finish_reading:
            terminate = True

        return self._get_obs(), reward, terminate, {}

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_obs(self):
        pass

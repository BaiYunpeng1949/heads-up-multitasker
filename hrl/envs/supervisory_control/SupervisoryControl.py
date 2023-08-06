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
        One step in this task can be defined as many seconds in the real world.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Register the reading content layout - has to be consistent with the layout in the WordSelection environment
        self._reading_content_per_page = {
            'num_rows': 3,
            'num_cols': 4,
            'num_words': 12,
        }

        # Initialize the beliefs - the reading progress belief and walking progress belief
        self._reading_task_belief = None
        self._walking_task_belief = None
        self._reading_task_weight_range = [0.5, 1]
        self._reading_task_weight = None
        self._walking_task_weight = None
        self._total_reading_words = 300   # The number of cells/words that needed to be read
        self._reading_words_progress = None
        self._reading_positions = {
            'margins': -1,  # The beginning and the end of the reading task
            'middle': 1,
        }
        self._reading_position = None

        self._background_events = {
            'left_lane': -1,
            'right_lane': 1,
        }
        self._background_event = None
        self._prev_background_event_step = None

        self._agent_on_lane = None

        self._background_last_checks = {
            'recent_checked': -1,
            'distant_checked': 1,
        }
        self._last_check_step_wise_threshold = 10
        self._background_last_check = None
        self._prev_background_last_check_step = None

        self._perceived_background_events = {
            'recent_event': -1,
            'distant_event': 1,
        }
        self._perceived_background_event_step_wise_threshold = 10
        self._perceived_background_event = None
        self._prev_perceived_background_event = None
        self._prev_perceived_background_event_step = None

        # Initialize the walking task environment related parameters - TODO maybe determine more metrics
        self._SMART_GLASSES = -1
        self._BACKGROUND = 1
        self._attention_switch_to_background = None
        # Step-wise background information update frequency levels
        self._bg_event_intervals = {
            'short': 10,
            'long': 20,
        }
        self._bg_event_interval_level = None
        self._rho_bg_event_interval_noise = 0.25
        self._bg_event_interval = None
        self._bg_event_interval_noise = None

        # Initialize the reward shaping related parameters
        self._attention_switch_time_cost = None
        self._word_selection_time_cost = None
        self._word_selection_error_cost = None
        self._lane_change_time_cost = None

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
            # Embed the middle level task models into the supervisory control model
            pass
        # For the training mode, only focus on this model, deploy a lot of stochasticity
        else:
            # Reset the reading task related parameters
            self._reading_words_progress = 0
            self._reading_position = self._reading_positions['beginning']

            # Reset the walking and background related parameters
            self._background_event = np.random.choice(list(self._background_events.values()))
            self._agent_on_lane = self._background_event
            self._prev_background_event_step = 0
            self._background_last_check = self._background_last_checks['just_checked']
            self._prev_background_last_check_step = 0
            self._perceived_background_event = self._perceived_background_events['short_event']
            self._prev_perceived_background_event = self._perceived_background_event
            self._prev_perceived_background_event_step = 0

            # Randomly initialize the attention allocation
            self._attention_switch_to_background = False

            # Randomly initialize the background information update frequency level
            levels = list(self._bg_event_intervals.keys())
            self._bg_event_interval_level = np.random.choice(levels)
            self._bg_event_interval = self._bg_event_intervals[self._bg_event_interval_level]
            self._bg_event_interval_noise = self._bg_event_interval * self._rho_bg_event_interval_noise

            # TODO debug delete later
            print(f"\n"
                  f"bg_info_freq: {self._bg_event_interval}, "
                  f"bg_info_freq_lv: {self._bg_event_interval_level}"
                  f"bg_info_freq_noise: {self._bg_event_interval_noise}\n")

            # Randomly initialize the reading task weight and walking task weight - describes the perceived importance of the two tasks
            self._reading_task_weight = np.random.uniform(self._reading_task_weight_range[0],
                                                          self._reading_task_weight_range[1])
            self._walking_task_weight = 1 - self._reading_task_weight

            # Initialize the reading task belief and walking task belief,
            #  the belief describes the possibility of sampling a state, here, the state is
            # The reading task belief contains information including the cost of switching the task estimation -
            #   it describes how easily it is to pick up the word correctly after the attention is switched back to words,
            #   the higher the belief value, the easier it is to pick up the word
            self._reading_task_belief = 0
            # The walking task belief contains information including the possibility of the lane instruction has changed,
            #   the higher the belief value, the higher the possibility of the lane instruction has changed
            self._walking_task_belief = 0

            # Reset the reward shaping related parameters
            self._word_selection_time_cost = 0
            self._word_selection_error_cost = 0
            self._attention_switch_time_cost = 0
            self._lane_change_time_cost = 0

        return self._get_obs()

    def render(self, mode="rgb_array"):
        pass

    def step(self, action):

        # Action a - decision-making on every time step
        action = np.normalize(action, -1, 1, -1, 1)
        if action <= 0:
            self._attention_switch_to_background = False
        else:
            self._attention_switch_to_background = True

        # State s'
        self._steps += 1
        reward = 0
        terminate = False
        finish_reading = False

        # Update the background event updates, we assume when the background events are updated,
        #   if the agent switch its attention on it, the agent can observe it immediately
        if self._steps - self._prev_background_event_step >= self._bg_event_interval:
            prev_background_event = self._background_event
            # From the current background event, sample a new background event
            while True:
                self._background_event = np.random.choice(list(self._background_events.values()))
                if self._background_event != prev_background_event:
                    break
            self._prev_background_event_step = self._steps

        # Continue reading
        if not self._attention_switch_to_background:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                pass
                # TODO Call the pre-trained RL models - Maybe a simple MDP reading model + ocular motor control
            else:
                # Continue the reading task, read one more cells/words
                self._reading_words_progress += 1
                # Update the reading position
                self._get_reading_position()

        # Switch attention to the environment
        else:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                pass
                # TODO Call the pre-trained RL models
                #  1.RHS: sign read + ocular motor control + locomotion control
                #  2.LHS: word selection + ocular motor control
            # All other non-testing modes
            else:
                # Update the background last check
                self._prev_background_last_check_step = self._steps
                # If the instruction in the background has changed, the agent will update its state
                if self._prev_perceived_background_event != self._perceived_background_event:
                    self._prev_perceived_background_event = self._perceived_background_event
                    self._prev_perceived_background_event_step = self._steps

                    # The background information has been read, then move to the correct lane
                    self._agent_on_lane = self._perceived_background_event

                    # Resume reading - word selection
                    # TODO how to describe it without using the pre-trained model?
                    #  1. need to get the selection time (wasted time) and the error for reward shaping
                    #  Use the pre-trained model's data - simulated results - it looks like a model-based approach,
                    #  we just need to distinguish between the layouts and reading positions
                    self._word_selection_time_cost = 0
                    self._word_selection_error_cost = 0
            # TODO apply cost here, it can include: 1. the time cost of switching the attention,
            #  2. the cost of facing risks of not picking up the word correctly

        # Check and update the last checks
        if self._steps - self._prev_background_last_check_step >= self._background_last_check:
            self._background_last_check = self._background_last_checks['recent_checked']
        else:
            self._background_last_check = self._background_last_checks['distant_checked']

        # Check and update the perceived background events
        if self._steps - self._prev_perceived_background_event_step >= self._perceived_background_event:
            self._perceived_background_event = self._perceived_background_events['recent_event']
        else:
            self._perceived_background_event = self._perceived_background_events['distant_event']

        # Update beliefs
        # TODO

        # Reward r
        # TODO define the reward function here - read as much as possible but also maintain the environmental awareness
        #   if missed the environmental awareness, then penalize the agent;
        #   if interrupt too much to the reading task, then penalize the agent
        reward_reading = 0  # TODO related to the reading progress and the
        reward_walking = 0  # TODO related to the actual environmental awareness
        reward += self._reading_task_weight * reward_reading + self._walking_task_weight * reward_walking

        # Check the reading progress
        if self._reading_words_progress >= 1:
            finish_reading = True

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

    def _get_reading_position(self):
        # Get the remains in a page
        page_remain = self._reading_words_progress % self._reading_content_per_page['num_words_per_page']
        # If the remains is 0, means the agent has finished reading a page, then the reading position is at the margin
        if page_remain == 0:
            self._reading_position = self._reading_positions['margins']
        else:
            # If the remains is not 0, then determine its position from a sentence's perspective
            sentence_remain = page_remain % self._reading_content_per_page['num_cols']
            # If the remained word is the first or last word in a sentence, then the reading position is at the margin
            if sentence_remain == 1 or sentence_remain == self._reading_content_per_page['num_cols']:
                self._reading_position = self._reading_positions['margins']
            # If the remained word is in the middle of a sentence, then the reading position is at the middle
            else:
                self._reading_position = self._reading_positions['middle']

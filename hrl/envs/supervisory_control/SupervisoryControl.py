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

        self._L0 = 'L0'
        self._L50 = 'L50'
        self._L100 = 'L100'
        self._reading_content_layouts = {
            self._L0: -1,
            self._L50: 0,
            self._L100: 1,
        }
        self._reading_content_layout_name = None

        # These are parameters normalized from the empirical results
        self._word_selection_time_costs = {
            self._L0: 1,
            self._L50: -0.33,
            self._L100: -1,
        }
        self._word_selection_time_cost = None
        # self._word_selection_time_cost_noise = 0.25     # Not needed now

        self._word_selection_error_costs = {
            self._L0: 1,
            self._L50: 0.11,
            self._L100: -1,
        }
        self._word_selection_error_cost = None
        # self._word_selection_error_cost_noise = 0.25      # Not needed now

        # Initialize the reading and walking tasks related parameters
        self._reading_task_weight_range = [0.5, 1]
        self._reading_task_weight = None
        self._walking_task_weight = None
        self._total_reading_words = 300   # The number of cells/words that needed to be read
        self._reading_progress_word_wise = None
        self._prev_reading_progress_word_wise = None
        self._MARGINS = 'margins'
        self._MIDDLE = 'middle'
        self._reading_positions = {
            self._MARGINS: -1,  # The beginning and the end of the reading task
            self._MIDDLE: 1,
        }
        self._reading_position = None
        self._reading_position_cost_factors = {
            self._MARGINS: 0.5,
            self._MIDDLE: 1,
        }
        self._reading_position_cost_factor = None

        self._background_events = {
            'left_lane': -1,
            'right_lane': 1,
        }
        self._background_event = None
        self._prev_background_event_step = None

        self._walking_lane = None

        self._RECENT = 'recent'
        self._DISTANT = 'distant'
        self._background_last_check_flags = {
            self._RECENT: -1,
            self._DISTANT: 1,
        }
        self._background_last_check_step_wise_threshold = 10
        self._background_last_check_flag = None
        self._prev_background_last_check_step = None

        self._background_last_check_with_different_event_flags = {
            self._RECENT: -1,
            self._DISTANT: 1,
        }
        self._background_last_check_with_different_event_step_wise_threshold = 10
        self._background_last_check_with_different_event_flag = None
        self._observed_background_event = None
        self._prev_observed_background_event = None
        self._prev_background_last_check_with_different_event_step = None

        # Initialize the walking task environment related parameters
        self._attention_switch_to_background = None
        # Step-wise background information update frequency levels
        self._SHORT = 'short'
        self._LONG = 'long'
        self._background_event_intervals = {
            self._SHORT: 10,
            self._LONG: 20,
        }
        self._background_event_interval_level = None
        self._rho_bg_event_interval_noise = 0.25
        self._background_event_interval = None
        self._bg_event_interval_noise = None

        # Initialize the belief of the current state
        self._beliefs = None

        # Initialize the RL training related parameters
        self._steps = None
        self.ep_len = 200
        self._epsilon = 1e-100

        # Define the observation space
        self._num_stateful_info = 10
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
            self._reading_progress_word_wise = 0
            self._prev_reading_progress_word_wise = 0
            self._reading_position = self._reading_positions[self._MARGINS]
            self._reading_position_cost_factor = self._reading_position_cost_factors[self._MARGINS]
            self._reading_content_layout_name = np.random.choice(list(self._reading_content_layouts.keys()))
            self._word_selection_time_cost = self._word_selection_time_costs[self._reading_content_layout_name]
            self._word_selection_error_cost = self._word_selection_error_costs[self._reading_content_layout_name]

            # Reset the walking and background related parameters
            self._background_event = np.random.choice(list(self._background_events.values()))
            self._walking_lane = self._background_event
            self._prev_background_event_step = 0
            self._observed_background_event = self._background_event
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]
            self._prev_background_last_check_step = 0
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]
            self._prev_observed_background_event = self._background_last_check_with_different_event_flag
            self._prev_background_last_check_with_different_event_step = 0

            # Randomly initialize the attention allocation
            self._attention_switch_to_background = False

            # Randomly initialize the background information update frequency level
            self._background_event_interval_level = np.random.choice(list(self._background_event_intervals.keys()))
            self._background_event_interval = self._background_event_intervals[self._background_event_interval_level]
            # self._bg_event_interval_noise = self._bg_event_interval * self._rho_bg_event_interval_noise     # Not need this now

            # Randomly initialize the reading task weight and walking task weight - describes the perceived importance of the two tasks
            self._reading_task_weight = np.random.uniform(self._reading_task_weight_range[0],
                                                          self._reading_task_weight_range[1])
            self._walking_task_weight = 1 - self._reading_task_weight

            self._update_beliefs()

            # Initialize the reading task belief and walking task belief,
            #  the belief describes the possibility of sampling a state, here, the state is
            # The reading task belief contains information including the cost of switching the task estimation -
            #   it describes how easily it is to pick up the word correctly after the attention is switched back to words,
            #   the higher the belief value, the easier it is to pick up the word
            # The walking task belief contains information including the possibility of the lane instruction has changed,
            #   the higher the belief value, the higher the possibility of the lane instruction has changed

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

        # Update the background events
        self._update_background_events()

        # Apply actions
        # Continue reading
        if not self._attention_switch_to_background:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                pass
                # TODO Call the pre-trained RL models - Maybe a simple MDP reading model + ocular motor control
            else:
                self._prev_reading_progress_word_wise = self._reading_progress_word_wise
                # Continue the reading task, read one more cells/words
                self._reading_progress_word_wise += 1
                # Update the reading position
                self._update_reading_position()
        # Switch attention to the environment, interrupt the reading task
        else:
            # Testing mode
            if self._config['rl']['mode'] == 'test':
                pass
                # TODO Call the pre-trained RL models
                #  1.RHS: sign read + ocular motor control + locomotion control
                #  2.LHS: word selection + ocular motor control
            # All other non-testing modes
            else:
                # Update the background last check - no matter whether observe a changed event instruction of not
                self._prev_background_last_check_step = self._steps
                # Update the observation of the background event
                self._observed_background_event = self._background_event
                # If the instruction in the background has changed, the agent will update its state
                if self._observed_background_event != self._prev_observed_background_event:
                    self._prev_observed_background_event = self._observed_background_event
                    self._prev_background_last_check_with_different_event_step = self._steps

                    # The background information has been read, then move to the correct lane
                    self._walking_lane = self._background_event

                    # Resume reading - word selection
                    self._update_word_selection_costs()

        # Update the step relevant flags
        self._update_background_check_flags()

        # Update beliefs
        self._update_beliefs()

        # Reward r
        reward = self._get_reward()

        # Termination
        terminate = False
        if self._steps >= self.ep_len or self._reading_progress_word_wise >= self._total_reading_words:
            terminate = True

        return self._get_obs(), reward, terminate, {}

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_obs(self):
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        event_update_freq_norm = -1 if self._background_event_interval_level == self._SHORT else 1
        # Get the belief observation - normalize to [-1, 1]
        belief = self._beliefs.copy()

        stateful_info = np.array([remaining_ep_len_norm, event_update_freq_norm, *belief])

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError(f"The shape of stateful information observation is not correct! "
                             f"The allocated shape is: {self._num_stateful_info}, "
                             f"the actual shape is: {stateful_info.shape[0]}")

        return stateful_info

    def _update_background_events(self):
        # Update the background event updates, we assume when the background events are updated,
        #   if the agent switch its attention on it, the agent can observe it immediately
        if self._steps - self._prev_background_event_step >= self._background_event_interval:
            prev_background_event = self._background_event
            # From the current background event, sample a new background event
            while True:
                self._background_event = np.random.choice(list(self._background_events.values()))
                if self._background_event != prev_background_event:
                    break
            self._prev_background_event_step = self._steps

    def _update_reading_position(self):
        # Get the remains in a page
        page_remain = self._reading_progress_word_wise % self._reading_content_per_page['num_words']
        # If the remains is 0, means the agent has finished reading a page, then the reading position is at the margin
        if page_remain == 0:
            reading_position_key = self._MARGINS
        else:
            # If the remains is not 0, then determine its position from a sentence's perspective
            sentence_remain = page_remain % self._reading_content_per_page['num_cols']
            # If the remained word is the first or last word in a sentence, then the reading position is at the margin
            if sentence_remain == 1 or sentence_remain == self._reading_content_per_page['num_cols']:
                reading_position_key = self._MARGINS
            # If the remained word is in the middle of a sentence, then the reading position is at the middle
            else:
                reading_position_key = self._MIDDLE
        # Update the reading position and the reading position cost factor
        self._reading_position = self._reading_positions[reading_position_key]
        self._reading_position_cost_factor = self._reading_position_cost_factors[reading_position_key]

    def _update_word_selection_costs(self):
        # Start from training with simple cases: deterministic word selection costs.
        #   Then move to the more complex cases where costs varies across trials
        self._word_selection_time_cost = self._word_selection_time_cost
        self._word_selection_error_cost = self._word_selection_error_cost

    def _update_background_check_flags(self):
        # Check and update the last checks
        if self._steps - self._prev_background_last_check_step >= self._background_last_check_step_wise_threshold:
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]
        else:
            self._background_last_check_flag = self._background_last_check_flags[self._RECENT]

        # Check and update the perceived background events
        if self._steps - self._prev_background_last_check_with_different_event_step >= self._background_last_check_with_different_event_step_wise_threshold:
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]
        else:
            self._background_last_check_with_different_event_flag = self._background_last_check_with_different_event_flags[self._RECENT]

    def _update_beliefs(self):
        # Update the LHS's reading related beliefs, containing the reading progress, reading position, estimated costs of switching attention
        reading_progress_norm = self.normalise(self._reading_progress_word_wise, 0, self._total_reading_words, -1, 1)
        reading_position_norm = self._reading_position
        reading_content_layout_norm = self._reading_content_layouts[self._reading_content_layout_name]
        word_selection_time_cost_norm = self._word_selection_time_cost * self._reading_position_cost_factor
        word_selection_error_cost_norm = self._word_selection_error_cost * self._reading_position_cost_factor
        reading_related_beliefs = [reading_progress_norm, reading_position_norm, reading_content_layout_norm,
                                   word_selection_time_cost_norm, word_selection_error_cost_norm]

        # Update the RHS's environmental awareness/Walking related beliefs
        background_last_check_norm = self._background_last_check_flag
        background_last_check_with_different_event_norm = self._background_last_check_with_different_event_flag
        walking_lane_norm = self._walking_lane
        walking_related_beliefs = [background_last_check_norm, background_last_check_with_different_event_norm, walking_lane_norm]

        # Update the beliefs
        self._beliefs = reading_related_beliefs + walking_related_beliefs

    def _get_reward(self, reward=0):
        # Customized reward function, coefficients are to be tuned/modified
        if self._reading_progress_word_wise > self._prev_reading_progress_word_wise:
            reward_reading_making_progress = 1
        else:
            reward_reading_making_progress = 0

        if self._attention_switch_to_background:
            reward_attention_switch_time_cost = -0.05   # Can be proportional to the time cost
            reward_word_selection_time_cost = -self._reading_position_cost_factor * self.normalise(self._word_selection_time_cost, -1, 1, 0.1, 0.25)
            reward_word_selection_error_cost = -self._reading_position_cost_factor * self.normalise(self._word_selection_error_cost, -1, 1, 0.1, 0.25)
        else:
            reward_attention_switch_time_cost = 0
            reward_word_selection_time_cost = 0
            reward_word_selection_error_cost = 0

        if self._walking_lane == self._background_event:
            reward_walk_on_correct_lane = 1
        else:
            reward_walk_on_correct_lane = -1

        reward += \
            self._reading_task_weight * (reward_reading_making_progress + reward_word_selection_time_cost + reward_word_selection_error_cost) + \
            self._walking_task_weight * reward_walk_on_correct_lane + \
            reward_attention_switch_time_cost

        return reward

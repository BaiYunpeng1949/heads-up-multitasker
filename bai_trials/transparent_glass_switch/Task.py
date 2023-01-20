import os
import numpy as np


class Task:

    def _config(self, configs):
        """
        This method set all the task needed configurations from the YAML config file.
        """
        self._config_task = configs['task_spec']['scripts']

        self._height = configs['mujoco_env']['render']['height']
        self._width = configs['mujoco_env']['render']['width']
        self._config_task['x_sample'] = int(self._height / 2 - 5)
        self._config_task['y_sample'] = int(self._width / 2 - 5)

        # Focal point moving distance configurations.
        self._demo_dist_configs = {
            'min_dist': 0,
            'max_dist': 1,  # TODO This could be a tunable parameter in the design space.
        }
        self._demo_mapping_range = self._demo_dist_configs['max_dist'] - self._demo_dist_configs['min_dist']
        # The demo mode's simulation time.
        self._demo_sim_time = configs['task_spec']['demo_sim_time']

    def _init(self):
        """
        This method initializes all the runtime data, especially states.
        """
        # The task's finite state machine.
        self._task_states = {  # task finite state machine
            'current_glass_display_id': 0,
            'current_env_color_id': 0,
            'start_step_glass_display_timestamp': 0,  # the current glass display started frame's timestamp, in seconds
            'start_step_env_color_timestamp': 0,  # the current env class started frame's timestamp, in seconds
            'previous_glass_display_id': 0,
            'previous_env_color_id': 0,
            'previous_step_timestamp': 0,
        }

    def __init__(self, configs):
        """
        This class defines the RL task game, specifies scripts, and manipulate all the runtime data.
        """
        # Set the configs from configurations read from the YAML file in the simulation class.
        self._config(configs=configs)

        # Initialize the runtime data.
        self._init()

    def reset(self):
        """
        This method resets all the runtime data, especially the task game states.
        """
        self._init()

    def step(self):
        """
        This method updates the task game states according to the pre-designed .
        """

    def __del__(self):
        pass

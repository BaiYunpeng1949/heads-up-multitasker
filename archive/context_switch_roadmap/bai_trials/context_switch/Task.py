import numpy as np


class Task:

    def _load_config(self, config):
        """
        This method set all the task needed configurations from the YAML config file.
        """
        self._config = config
        self._config_task = config['task']

        self._height = config['mj_env']['render']['height']
        self._width = config['mj_env']['render']['width']

        # The observation point - a temporary point.
        offset = 2
        self._sample_x = int(self._height / 2 - offset)
        self._sample_y = int(self._width / 2 - offset)

    def _init_data(self):
        """
        This method initializes all the runtime data, especially states.
        """
        # The task's finite state machine.
        self._task_status = {  # task finite state machine
            'current_glass_display_id': 0,
            'current_env_color_id': 0,
            'start_step_glass_display_timestamp': 0,  # the current glass display started frame's timestamp, in seconds
            'start_step_env_color_timestamp': 0,  # the current env class started frame's timestamp, in seconds
            'previous_glass_display_id': 0,
            'previous_env_color_id': 0,
            'previous_step_timestamp': 0,
        }

        # States.
        self._states = {
            'optimal_score': 0,
            'total_time_on_glass_B': 0,
            'total_time_on_env_red': 0,
            'total_time_on_glass_X': 0,
            'total_time_miss_glass_B': 0,
            'total_time_miss_env_red': 0,
            'total_time_miss_glass_X': 0,
            'total_time_glass_B': 0,
            'total_time_glass_X': 0,
            'total_time_env_red': 0,
            'total_time_intermediate': 0,
            'num_on_glass': 0,
            'num_on_env': 0,
            'current_on_level': 0,
        }

    def __init__(self, config):
        """
        This class defines the RL task game, specifies scripts, and manipulate all the runtime data.
        """
        # Set the configs from configurations read from the YAML file in the simulation class.
        self._load_config(config=config)

        # Initialize the runtime data.
        self._init_data()

    def reset(self):
        """
        This method resets all the runtime data, especially the task game states.
        """
        self._init_data()

    def step(self, action):
        """
        This method updates the task game states according to the pre-designed .
        """
        pass

    def update(self, action, time):
        """
        Update the task scenarios.
        """
        current_step_timestamp = time
        elapsed_time = current_step_timestamp - self._task_status['previous_step_timestamp']

        # Check the glass status.
        current_glass_display_duration = current_step_timestamp - self._task_status[
            'start_step_glass_display_timestamp']
        if current_glass_display_duration > self._config_task['glass_display_duration']:
            # A deterministic scenario.
            current_glass_display_id = int(
                (self._task_status['current_glass_display_id'] + 1) % len(self._config_task['glass_display_choices'])
            )

            # Update the task state machine.
            self._task_status['start_step_glass_display_timestamp'] = current_step_timestamp
            self._task_status['previous_glass_display_id'] = self._task_status['current_glass_display_id']
            self._task_status['current_glass_display_id'] = current_glass_display_id
        else:
            # Get the exact glass display time:
            if self._task_status['current_glass_display_id'] == 1:
                self._states['total_time_glass_B'] += elapsed_time
            elif self._task_status['current_glass_display_id'] == 2:
                self._states['total_time_glass_X'] += elapsed_time
            else:
                pass

        # Check the environment status.
        current_env_color_duration = current_step_timestamp - self._task_status['start_step_env_color_timestamp']
        if current_env_color_duration > self._config_task['env_color_duration']:
            current_env_color_id = int((self._task_status['current_env_color_id'] + 1) %
                                       len(self._config_task['env_color_choices']))
            # Update the task state machine.
            self._task_status['start_step_env_color_timestamp'] = current_step_timestamp
            self._task_status['previous_env_color_id'] = self._task_status['current_env_color_id']
            self._task_status['current_env_color_id'] = current_env_color_id
        else:
            # Get the exact env color time:
            if self._task_status['current_env_color_id'] == 1:
                self._states['total_time_env_red'] += elapsed_time
            else:
                pass

        if action == 0:  # look at the smart glass lenses
            alpha = 0.5
            color_id = 0
            self._states['num_on_glass'] += 1
        elif action == 1:  # look at the env
            alpha = 0
            color_id = self._task_status['current_env_color_id']
            self._states['num_on_env'] += 1
        else:
            alpha = 0
            color_id = 0

        # Update the timer.
        self._task_status['previous_step_timestamp'] = current_step_timestamp

        task_status = self._task_status.copy()

        return alpha, color_id, task_status, elapsed_time

    def make_decision(self, action, rgb, elp_time):
        """
        Make decisions based on the current scenarios.
        """
        sample_point_rgb = rgb[self._sample_x, self._sample_y, :]

        # Get the current visual content.
        perceived_content = self.identify_visual_content(
            sample_point=sample_point_rgb,
            reference=self._config_task,
            task_status=self._task_status,
        )

        # Make decisions.
        if perceived_content == 'on_env_grey':
            # Check if lost the B glass display.
            if self._task_status['current_glass_display_id'] == 1:
                self._states['total_time_miss_glass_B'] += elp_time
                self._states['current_on_level'] = -3
            elif self._task_status['current_glass_display_id'] == 2:
                self._states['total_time_miss_glass_X'] += elp_time
                self._states['current_on_level'] = -1
            elif self._task_status['current_glass_display_id'] == 0:
                self._states['current_on_level'] = 0
        elif perceived_content == 'on_env_red':
            self._states['total_time_on_env_red'] += elp_time
            # Check if lost the B glass display.
            if self._task_status['current_glass_display_id'] == 1:
                self._states['total_time_miss_glass_B'] += elp_time
                self._states['current_on_level'] = -3
            elif self._task_status['current_glass_display_id'] == (0 or 2):
                self._states['current_on_level'] = 2
        elif perceived_content == 'on_env_green':
            # Check if lost the B glass display.
            if self._task_status['current_glass_display_id'] == 1:
                self._states['total_time_miss_glass_B'] += elp_time
                self._states['current_on_level'] = -3
            elif self._task_status['current_glass_display_id'] == 2:
                self._states['total_time_miss_glass_X'] += elp_time
                self._states['current_on_level'] = -1
            elif self._task_status['current_glass_display_id'] == 0:
                self._states['current_on_level'] = 0
        elif perceived_content == 'on_env_blue':
            # Check if lost the B glass display.
            if self._task_status['current_glass_display_id'] == 1:
                self._states['total_time_miss_glass_B'] += elp_time
                self._states['current_on_level'] = -3
            elif self._task_status['current_glass_display_id'] == 2:
                self._states['total_time_miss_glass_X'] += elp_time
                self._states['current_on_level'] = -1
            elif self._task_status['current_glass_display_id'] == 0:
                self._states['current_on_level'] = 0
        elif perceived_content == 'on_glass_nothing':
            # Check if lost the red env.
            if self._task_status['current_env_color_id'] == 1:
                self._states['total_time_miss_env_red'] += elp_time
                self._states['current_on_level'] = -2
            elif self._task_status['current_env_color_id'] == (0 or 2 or 3):
                self._states['current_on_level'] = 0
        elif perceived_content == 'on_glass_B':
            self._states['total_time_on_glass_B'] += elp_time
            self._states['current_on_level'] = 3
        elif perceived_content == 'on_glass_X':
            self._states['total_time_on_glass_X'] += elp_time
            # Check if lost the red env.
            if self._task_status['current_env_color_id'] == 1:
                self._states['total_time_miss_env_red'] += elp_time
                self._states['current_on_level'] = -2
            elif self._task_status['current_env_color_id'] == (0 or 2 or 3):
                self._states['current_on_level'] = 1
        else:
            self._states['total_time_intermediate'] += elp_time
            self._states['current_on_level'] = 0

        # Calculate the optimal rewards.
        task_states = self._task_status.copy()
        if task_states['current_glass_display_id'] == 1:
            self._states['optimal_score'] += 5
        elif task_states['current_glass_display_id'] == 2:
            if task_states['current_env_color_id'] != 1:
                self._states['optimal_score'] += 1
            else:
                self._states['optimal_score'] += 3
        else:
            if task_states['current_env_color_id'] == 1:
                self._states['optimal_score'] += 3
            else:
                self._states['optimal_score'] += 0

        # Debug printings.
        # if self._config['rl']['mode'] == 'debug':
        #     print('action: {}   rgb: {}     glass id: {}    env id: {}      perceived result: {}'
        #           .format(action, sample_point_rgb, self._task_status['current_glass_display_id'],
        #                   self._task_status['current_env_color_id'], perceived_content))

    @staticmethod
    def identify_visual_content(sample_point, reference, task_status):
        ref = reference.copy()
        dist_threshold = 6
        content = 'none'

        for comparison in ref['on_env_grey']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                if task_status['current_glass_display_id'] != 0:
                    content = 'on_env_grey'
                # elif (sample_point == states['on_env_grey'][0]).all():
                #     content = 'on_env_grey'

        for comparison in ref['on_env_green']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                content = 'on_env_green'

        for comparison in ref['on_env_blue']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                content = 'on_env_blue'

        for comparison in ref['on_glass_nothing']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                if task_status['current_env_color_id'] != 0:
                    content = 'on_glass_nothing'
                # elif (sample_point == states['on_glass_nothing'][1]).all():
                #     content = 'on_glass_nothing'

        for comparison in ref['on_glass_X']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                content = 'on_glass_X'

        for comparison in ref['on_env_red']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                content = 'on_env_red'

        for comparison in ref['on_glass_B']:
            if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                content = 'on_glass_B'

        return content

    @property
    def get_sample(self):
        """
        Gets the temporary observation sample point's x and y coordinators.

        Returns:
            sample's x and y coordinators.
        """
        return self._sample_x, self._sample_y

    @property
    def states(self):
        """
        Gets the states.
        """
        return self._states

    def __del__(self):
        pass

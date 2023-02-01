class Task:
    def _load_config(self):
        try:
            self._conf_mj_env = self._config['mj_env']
            self._conf_rl = self._config['rl']
            self._conf_task = self._config['task']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

    def _init_data(self):
        # Ground truth: the task status.
        self._ground_truth = {
            'unexplored_grids': [0, 1, 2, 3],   # TODO maybe add to the observation space.
            'explored_grids': [],               # TODO maybe add to the observation space.
            'fixing_steps': 0,
            'wasted_steps:': 0,
        }

        # States.
        self._states = {
            'grid_id': 0,   # The grid that the focus is currently on.
            'pre_grid_id': 0    # The previous grid that the focus was on.
        }

    def __init__(self, config):
        # Configure the settings.
        self._config = config
        self._load_config()

        # Initialize the runtime data.
        self._init_data()

    def reset(self):
        self._init_data()

    def update(self, action):
        # Update the previous grid.
        pre_grid_id = self._states['grid_id']
        self._states['pre_grid_id'] = pre_grid_id

        # Update the current grid.
        if action == 0:     # Move the focus west.
            if pre_grid_id <= 0:    # Already on the westest grid.
                self._states['grid_id'] = pre_grid_id + 0
            else:
                self._states['grid_id'] = pre_grid_id - 1
        elif action == 1:   # Stay still.
            self._states['grid_id'] = pre_grid_id + 0
        elif action == 2:   # Move the focus east.
            if pre_grid_id >= 3:   # Already on the eastest grid.
                self._states['grid_id'] = pre_grid_id + 0
            else:
                self._states['grid_id'] = pre_grid_id + 1
        else:
            raise ValueError('Invalid action: Only west and east are allowed.')

    def make_decisions(self, **kwargs):
        grid_id = self._states['grid_id']
        pre_grid_id = self._states['pre_grid_id']

        # Update the fixing steps.
        if grid_id == pre_grid_id:
            self._ground_truth['fixing_steps'] += 1
        else:
            self._ground_truth['fixing_steps'] = 0

        # Update the wasted steps.
        if grid_id in self._ground_truth['explored_grids']:
            self._ground_truth['wasted_steps'] += 1
        elif grid_id in self._ground_truth['unexplored_grids']:
            self._ground_truth['wasted_steps'] = 0

        # Update the explored/unexplored grids set.
        if grid_id in self._ground_truth['unexplored_grids']:   # If the agent is focusing on an unexplored grid.
            if self._ground_truth['fixing_steps'] >= kwargs['fix_steps']:
                self._ground_truth['unexplored_grids'].remove(grid_id)
                self._ground_truth['explored_grids'].append(grid_id)
            else:
                pass

    @property
    def states(self):
        return self._states

    @property
    def ground_truth(self):
        return self._ground_truth

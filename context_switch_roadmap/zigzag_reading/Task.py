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
        self._ground_truth = {  # TODO I found teaching agent to learn to focus is very hard, maybe I should first teach him traverse.
                                #  The dynamic reading fixation should be coupled with some event, like comprehending a letter, only when more fixing time, more information will be disclosed.
            'unexplored_grids': [0, 1, 2, 3],
            'explored_grids': [],
            'fixing_steps': 0,
            'wasted_steps': 0,
            'grids_steps':  [0, 0, 0, 0],
            'move_to_unexp': 0,
            'move_to_exp': 0,
            'num_loops': 0,
        }

        # States.
        self._states = {
            'grid_id': 0,   # The grid that the focus is currently on.
            'pre_grid_id': 0,    # The previous grid that the focus was on.
            'finished_one_grid': False,
            'finished_one_loop': False,
        }

    def __init__(self, config):
        # Configure the settings.
        self._config = config
        self._load_config()

        # Initialize the runtime data.
        self._init_data()

    def reset(self):
        self._init_data()

    def step(self, action, **kwargs):
        self._update(action=action)
        self._make_decisions(threshold=kwargs['fix_steps'])

    def _update(self, action):
        # Update the previous grid.
        pre_grid_id = self._states['grid_id']
        self._states['pre_grid_id'] = pre_grid_id

        # Update the current grid.
        if action == 0:     # Move the focus west.
            if pre_grid_id <= 0:    # Already on the westest grid.
                self._states['grid_id'] += 0
            else:
                self._states['grid_id'] -= 1
        elif action == 1:   # Stay still.
            self._states['grid_id'] += 0
        elif action == 2:   # Move the focus east.
            if pre_grid_id >= 3:   # Already on the eastest grid.
                self._states['grid_id'] += 0
            else:
                self._states['grid_id'] += 1
        else:
            raise ValueError('Invalid action: Only west and east are allowed.')

    def _make_decisions(self, threshold):
        grid_id = self._states['grid_id']
        pre_grid_id = self._states['pre_grid_id']
        explored_grids = self._ground_truth['explored_grids']

        # TODO dynamically initializations.
        self._states['finished_one_grid'] = False
        self._states['finished_one_loop'] = False
        if len(self._ground_truth['unexplored_grids']) <= 0:
            self._ground_truth['unexplored_grids'] = [0, 1, 2, 3]

        # Update the wasted steps and fixing steps. The wasted_steps and fixing_steps should be orthogonal.
        if grid_id == len(explored_grids):
            self._ground_truth['wasted_steps'] = 0
            self._ground_truth['fixing_steps'] += 1
        else:
            self._ground_truth['wasted_steps'] += 1
            self._ground_truth['fixing_steps'] = 0

        # Update the explored/unexplored grids set.
        if grid_id in self._ground_truth['unexplored_grids']:   # If the agent is focusing on an unexplored grid.
            if self._ground_truth['fixing_steps'] >= threshold:
                self._ground_truth['unexplored_grids'].remove(grid_id)
                self._ground_truth['explored_grids'].append(grid_id)

                self._states['finished_one_grid'] = True
            else:
                pass

        # Update the logs.
        for i in range(len(self._ground_truth['grids_steps'])):
            if grid_id == i:
                self._ground_truth['grids_steps'][i] += 1

        # Update the loop.
        if len(self._ground_truth['unexplored_grids']) <= 0:    # All grids are traversed.
            # print('TODO oh one look is over!')  # TODO debug delete later.
            num_loops = self._ground_truth['num_loops']
            num_loops += 1
            unexplored_grids = self._ground_truth['unexplored_grids']

            # Refresh the useless buffers.
            self._init_data()   # Reset the scene and read from the 1st grid.

            # Reserve the useful buffers.
            self._states['finished_one_loop'] = True
            self._states['finished_one_grid'] = True
            self._ground_truth['num_loops'] = num_loops
            self._ground_truth['unexplored_grids'] = unexplored_grids

    @property
    def states(self):
        return self._states

    @property
    def ground_truth(self):
        return self._ground_truth


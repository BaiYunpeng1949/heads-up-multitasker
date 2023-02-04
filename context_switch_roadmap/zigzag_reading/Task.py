import numpy as np

from utils import debug


class Task:
    def _load_config(self):
        try:
            self._conf_mj_env = self._config['mj_env']
            self._conf_rl = self._config['rl']
            self._conf_task = self._config['task']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

        # Configure initializers.
        grids = debug.DEBUG['grids']
        self._h, self._w = np.array(grids).shape
        grid_ids = np.reshape(np.arange(np.prod(np.array(grids.copy()).shape), dtype=int), (self._h, self._w)).tolist()
        self._coords_to_ids = {}    # The mapping from coordinates to grid ids.
        for i, row in enumerate(grid_ids):
            for j, grid_id in enumerate(row):
                self._coords_to_ids[tuple((i, j))] = grid_id
        # The reverse mapping from grid ids to coordinates.
        #  Hence that I don't have to iterate in each action steps to save computation resources.
        self._ids_to_coords = {grid_id: coord for coord, grid_id in self._coords_to_ids.items()}

        self._unexplored_grid_ids = []
        for row in grids:
            for grid in row:
                if debug.DEBUG['keyword'] not in grid:
                    self._unexplored_grid_ids.append(int(grid.split('-')[-1]))

    def _init_data(self, **kwargs):
        # Ground truth: the task status. Most are local variables.
        self._ground_truth = {
            # TODO I found teaching agent to learn to focus is very hard, maybe I should first teach him traverse.
            #  The dynamic reading fixation should be coupled with some event, like comprehending a letter,
            #  only when more fixing time, more information will be disclosed.
            'unexplored_grid_ids': self._unexplored_grid_ids.copy(),
            'explored_grid_ids': [],
            'pre_fixing_steps': 0,
            'fixing_steps': 0,
            'wasted_steps': 0,
            'move_to_unexp': 0,
            'move_to_exp': 0,
            'num_loops': 0 if 'num_loops' not in kwargs else kwargs['num_loops'],
            # TODO add the traverse path length later.
        }

        # States.
        self._states = {
            'grid_id': 0 if 'grid_id' not in kwargs else kwargs['grid_id'],   # The grid that the focus is currently on.
            'pre_grid_id': 0 if 'pre_grid_id' not in kwargs else kwargs['pre_grid_id'],    # The previous grid that the focus was on.
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
        self._make_decisions(threshold=kwargs['fix_steps_threshold'])

    def _update(self, action):
        # Update the previous grid and previous fixing steps.
        self._states['pre_grid_id'] = self._states['grid_id']
        self._ground_truth['pre_fixing_steps'] = self._ground_truth['fixing_steps']

        # Data initialization - remake if one loop is finished: restart from the beginning.
        if len(self._ground_truth['unexplored_grid_ids']) <= 0:    # If all the non-vacant grids are traversed.
            # Refresh everything, but preserve some variables, such as a global counter num_loops.
            self._init_data(
                num_loops=self._ground_truth['num_loops'],
                grid_id=self._states['grid_id'],
                pre_grid_id=self._states['pre_grid_id'],
            )

        # Get the horizontal and vertical offsets by the input action.
        ofs_h, ofs_w = 0, 0
        try:
            ofs_h = action[0] - debug.DEBUG['dim_actions'] // 2  # The vertical/height offset: move to -1, 0, 1.
            ofs_w = action[1] - debug.DEBUG['dim_actions'] // 2  # The horizontal/width offset: move to -1, 0, 1.
        except ValueError:
            print(
                'Invalid action space. The action should be a 2D space. '
                'Check the task configuration in the ZigzagReadingEnv file.'
            )

        # Update the grid that the agent looks on. And then cater for the special cases.
        coord = self._ids_to_coords[self._states['grid_id']]   # Get the current grid's coordinate. Note: type tuple.
        coord = np.clip(
            a=(coord[0]+ofs_h, coord[1]+ofs_w),
            a_min=0,
            a_max=[self._h-1, self._w-1],
        )
        self._states['grid_id'] = self._coords_to_ids[tuple(coord)]

    def _make_decisions(self, threshold):
        grid_id = self._states['grid_id']
        pre_grid_id = self._states['pre_grid_id']
        unexplored_grid_ids = self._ground_truth['unexplored_grid_ids']

        # Update the wasted steps and fixing steps. The wasted_steps and fixing_steps should be orthogonal.
        if grid_id == min(unexplored_grid_ids):     # Determine whether it is the assigned next grid.
            self._ground_truth['wasted_steps'] = 0
            if pre_grid_id != grid_id:
                self._ground_truth['fixing_steps'] = 1
            else:
                self._ground_truth['fixing_steps'] += 1
        else:
            self._ground_truth['wasted_steps'] += 1
            # Check if many steps are wasted jumping from the target grid and the adjacent grid.
            if 0 < self._ground_truth['pre_fixing_steps'] < threshold:
                self._ground_truth['fixing_steps'] = - self._ground_truth['pre_fixing_steps']
            else:
                self._ground_truth['fixing_steps'] = 0

        # Update the explored/unexplored grids set.
        if grid_id in unexplored_grid_ids and self._ground_truth['fixing_steps'] >= threshold:   # If the agent is focusing on an unexplored grid.
            self._ground_truth['unexplored_grid_ids'].remove(grid_id)
            self._ground_truth['explored_grid_ids'].append(grid_id)

            self._states['finished_one_grid'] = True
        else:
            self._states['finished_one_loop'] = False
            self._states['finished_one_grid'] = False

        # Update the loop.
        if len(self._ground_truth['unexplored_grid_ids']) <= 0:    # All grids are traversed.
            self._ground_truth['num_loops'] += 1

            # Reserve the useful buffers.
            self._states['finished_one_loop'] = True
            self._states['finished_one_grid'] = True
        else:
            self._states['finished_one_loop'] = False

    @property
    def states(self):
        return self._states

    @property
    def ground_truth(self):
        return self._ground_truth

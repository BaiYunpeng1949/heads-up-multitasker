# TODO after debugging, these will be integrated into the configuration file.
DEBUG = {
    'dim_actions': 3,   # west/north; still; and east/south.
    'seed': 1997,
    # TODO, the grids layout might be programmed after these trials (use explicit ground truth --> use implicit pixels as observations).
    'grids': [
        ['grid-0',        'grid-1',        'grid-2',        'grid-3'],
        ['vacant-grid-4', 'vacant-grid-5', 'vacant-grid-6', 'vacant-grid-7'],
        ['grid-8',        'grid-9',       'grid-10',       'grid-11'],
        ['vacant-grid-12', 'vacant-grid-13', 'vacant-grid-14', 'vacant-grid-15'],
        ['grid-16',        'grid-17',       'grid-18',       'grid-19'],
    ],
    'keyword': 'vacant',

    'cam_name': 'single-eye',
    'ball_name': 'focus',
    'fix_steps_threshold': 25,      # The step periods that a fixation action takes.

    'obs_width': 80,                # The observations' portion pixels width.
    'obs_height': 80,               # The observations' portion pixels height.
    'explored_grid_ids': [],

    'grey': [0.2, 0.2, 0.2],
    'red': [0.5, 0, 0],
}

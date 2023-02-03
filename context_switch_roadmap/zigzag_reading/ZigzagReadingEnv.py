import cv2
import numpy as np
import math

import mujoco
from mujoco.glfw import glfw

from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete


from Task import Task

_DEBUG = {
    'num_actions': 3,   # 2 actions: 0 for west, 1 for still, and 2 for east.
    'seed': 1997,
    'grids': ['grid-1', 'grid-2', 'grid-3', 'grid-4'],
    'cam_name': 'single-eye',
    'ball_name': 'focus',
    'fix_steps': 50,     # The step periods that a fixation action takes.
    'obs_width': 80,   # The observations' portion pixels width.
    'obs_height': 80,  # The observations' portion pixels height.
    'explored_grids': [],
    'grey': [0.2, 0.2, 0.2],
    'num_loops': 2000,
}


class ZigzagReadingEnv(Env):

    def _load_config(self):
        try:
            self._conf_mj_env = self._config['mj_env']
            self._conf_rl = self._config['rl']
            self._conf_task = self._config['task']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

        # MuJoCo.
        self._mj_filename = self._conf_mj_env['filename']

        self._width = self._conf_mj_env['render']['width']
        self._height = self._conf_mj_env['render']['height']

        self._rgb = self._conf_mj_env['render']['rgb']
        self._depth = self._conf_mj_env['render']['depth']

        self._window_visible = self._conf_mj_env['render']['is_window_visible']
        if (self._conf_rl['mode'] == ('train' or 'test')) and (self._window_visible == 1):
            self._window_visible = 0
            raise Warning(
                'The training and testing modes does not allow visible rendering. The window has been reset invisible.'
            )
        if self._window_visible == 1:
            self._framebuffer = mujoco.mjtFramebuffer.mjFB_WINDOW.value
        else:
            self._framebuffer = mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value

        # RL.
        self._num_steps = self._conf_rl['train']['num_steps']

    def _init_data(self):
        # MuJoCo.
        self._rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8) if self._rgb else None
        self._depth_buffer = np.empty((self._height, self._width), dtype=np.float32) if self._depth else None
        self._rgb_images = []

        # RL.
        self._elp_steps = 0

    def __init__(self, config):
        """
        Initialize an instance of the MuJoCo environment within gym.Env architecture.
        This time the application scenario is: zigzag reading.

        Args:
            config: the configurations.
        """
        # Load the configurations.
        self._config = config
        self._load_config()

        # Initialize RL.
        self.action_space = Discrete(n=_DEBUG['num_actions'], seed=_DEBUG['seed'])
        self.observation_space = Dict({
            # 'visual_rgb': Box(low=0, high=255, shape=(_DEBUG['obs_height'], _DEBUG['obs_width'], 3)),   # TODO include the visual later.
            # 'focus': Discrete(n=len(_DEBUG['grids'])),   # Where is the focus at, 0 for grid-1.
            # 'pre_focus': Discrete(n=len(_DEBUG['grids'])),
            # 'fixing_steps': Discrete(n=(self._num_steps+1), seed=_DEBUG['seed']),
            # 'wasted_steps': Discrete(n=(self._num_steps+1), seed=_DEBUG['seed']),
            'pre_now_focus': MultiDiscrete([len(_DEBUG['grids']), len(_DEBUG['grids'])]),
            'unexplored_grids_binary': MultiBinary(len(_DEBUG['grids'])),
        })

        # Initialize MuJoCo.
        self._m = mujoco.MjModel.from_xml_path(filename=self._mj_filename)
        self._d = mujoco.MjData(self._m)

        # Initialize the visual perception in MuJoCo.
        self._cam = mujoco.MjvCamera()  # The abstract camera.
        cam_name = _DEBUG['cam_name']
        if isinstance(_DEBUG['cam_name'], str):
            cam_id = mujoco.mj_name2id(
                m=self._m,
                type=mujoco.mjtObj.mjOBJ_CAMERA,
                name=cam_name
            )
        else:
            cam_id = -1997
        if cam_id < -1:
            raise ValueError('camera_id cannot be smaller than -1.')
        if cam_id >= self._m.ncam:
            raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.format(self._m.ncam, cam_id))
        if cam_id == -1:
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # TODO debug here. Generalize this part later along with the implementation of the visual perception.
        self._cam.lookat = [0, 0, 0]
        self._cam.distance = 8
        self._cam.azimuth = 90.0
        self._cam.elevation = -8

        self._opt = mujoco.MjvOption()  # Visualization options.
        mujoco.mjv_defaultOption(opt=self._opt)

        glfw.init()  # Windows platform MuJoCo only supports GLFW.
        glfw.window_hint(glfw.VISIBLE, self._window_visible)
        self._window = glfw.create_window(  # Create an OpenGL context with GLFW.
            width=self._width,
            height=self._height,
            title='ZigzagReading',
            monitor=None,
            share=None,
        )
        glfw.make_context_current(window=self._window)

        self._con = mujoco.MjrContext(self._m, mujoco.mjtFontScale.mjFONTSCALE_150.value)   # Create a MuJoCo context.
        mujoco.mjr_setBuffer(
            framebuffer=self._framebuffer,
            con=self._con
        )

        self._scn = mujoco.MjvScene(self._m, maxgeom=10000)   # Create a scene in MuJoCo.

        self._viewport = mujoco.MjrRect(    # Create a viewport.
            left=0,
            bottom=0,
            width=self._width,
            height=self._height
        )

        # Initialize the corresponding task.
        self._task = Task(config=config)

        # Initialize runtime data.
        self._init_data()

        # Confirm settings.
        mujoco.mj_forward(m=self._m, d=self._d)

    def reset(self):
        # Reset the simulated environment.
        mujoco.mj_resetData(m=self._m, d=self._d)

        # Reset the task.
        self._task.reset()

        # Reset the data.
        self._init_data()

        # Confirm settings.
        mujoco.mj_forward(m=self._m, d=self._d)

        # Get the reset observations. In this task, it has to start from 0.
        obs = self._get_obs()

        return obs

    def step(self, action):
        # Advance the simulation
        mujoco.mj_step(m=self._m, d=self._d, nstep=1)  # nstep=self._run_parameters["frame_skip"]

        # Update environment.
        obs, reward, done, info = self._update(action=action)

        # Update the simulated steps.
        self._elp_steps += 1

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        if self._window:
            if glfw.get_current_context() == self._window:
                glfw.make_context_current(None)
            # Destroy the window.
            glfw.destroy_window(self._window)
            self._window = None

        glfw.terminate()

    def _update(self, action):
        # ---------------------- Done ----------------------
        done = False

        # ---------------------- Task: Update ----------------------
        self._task.step(action=action, fix_steps=_DEBUG['fix_steps'])

        # ---------------------- MuJoCo Render: Update ----------------------
        # Update the MuJoCo model - change the color of explored/traversed grid.
        explored_grids = self._task.ground_truth['explored_grids']
        if len(explored_grids) > len(_DEBUG['explored_grids']):
            new_grids = [grid_id for grid_id in explored_grids if grid_id not in _DEBUG['explored_grids']]
            new_grid = new_grids[0]
            # Change the color of that.
            for grid_name_str in _DEBUG['grids']:
                if str(new_grid) in grid_name_str:
                    self._m.geom(grid_name_str).rgba[0:3] = _DEBUG['grey']
            # Update the explored_grids buffer.
            _DEBUG['explored_grids'] = explored_grids
        # Update the focus-ball's movement.
        grid_id = self._task.states['grid_id']
        for grid_name_str in _DEBUG['grids']:
            if str(grid_id) in grid_name_str:
                self._d.qpos[0:3] = self._m.geom(grid_name_str).pos[0:3]
        # Render.
        self._update_render()

        # ---------------------- Observations ----------------------
        obs = self._get_obs()

        # ---------------------- Reward ----------------------
        reward = self._reward_function()

        # ---------------------- Info ----------------------
        info = self._task.ground_truth.copy()
        info['elp_steps'] = self._elp_steps
        info['optimal_loops'] = int(self._num_steps / (_DEBUG['fix_steps'] * len(_DEBUG['grids']) + (len(_DEBUG['grids']) - 2)))
        info['achievement'] = np.round(info['num_loops'] / info['optimal_loops'], 4)

        # ---------------------- Done -------- --------------
        if self._elp_steps >= self._num_steps:
            done = True

        # TODO debug area
        if self._conf_rl['mode'] == 'test':
            print('\nThe action is: {}; the previous grid id is: {}'
                  '\nThe current grid is: {}, the fixing steps is: {}, the current reward is: {}.'
                  '\nThe unexplored grids are: {}, the explored grids are: {}'
                  ''.
                  format(action, self._task.states['pre_grid_id'],
                         grid_id, self._task.ground_truth['fixing_steps'], reward,
                         self._task.ground_truth['unexplored_grids'], self._task.ground_truth['explored_grids']))

        return obs, reward, done, info

    def _get_obs(self):
        offset_h = _DEBUG['obs_height'] / 2
        offset_w = _DEBUG['obs_width'] / 2
        obs_idx_h = [int(self._height / 2 - offset_h), int(self._height / 2 + offset_h)]
        obs_idx_w = [int(self._width / 2 - offset_w), int(self._width / 2 + offset_w)]
        visual_rgb = self._rgb_buffer.copy()[obs_idx_h[0]:obs_idx_h[1], obs_idx_w[0]:obs_idx_w[1], :]
        focus = self._task.states['grid_id']
        pre_focus = self._task.states['pre_grid_id']
        fixing_steps = self._task.ground_truth['fixing_steps']
        wasted_steps = self._task.ground_truth['wasted_steps']

        # Get the unexplored grids.
        ref = np.arange(len(_DEBUG['grids'])).tolist()
        unexplored_grids = self._task.ground_truth['unexplored_grids']
        unexplored_grids_binaries = [1 if grid_id in unexplored_grids else 0 for grid_id in ref]

        obs = {
            # 'visual_rgb': visual_rgb, # TODO add the visual perception back later.
            # 'focus': focus,
            # 'pre_focus': pre_focus,
            # 'fixing_steps': fixing_steps,
            # 'wasted_steps': wasted_steps,
            'pre_now_focus': [pre_focus, focus],
            'unexplored_grids_binary': unexplored_grids_binaries,
        }

        return obs

    def _reward_function(self):
        g_t = self._task.ground_truth
        f_s = g_t['fixing_steps']
        w_s = g_t['wasted_steps']

        basic_reward = 1
        gain_fixing = 1
        gain_wasting = -1
        gain_finish_grid = _DEBUG['fix_steps'] + 2
        gain_finish_loop = len(_DEBUG['grids']) + 2

        if self._task.states['finished_one_grid']:
            if self._task.states['finished_one_loop']:
                reward = basic_reward * gain_finish_grid * gain_finish_loop
            else:
                reward = basic_reward * gain_finish_grid
        else:
            reward = gain_fixing * min(basic_reward, f_s) + gain_wasting * min(basic_reward, w_s)

        return reward

    def _update_render(self):
        """
        Update the scene and render for 1 step.
        Be called in the internal method: _update.
        """
        # Update the camera.
        # No need to update in this application since I am using a fixed camera.

        # Update the scene.
        mujoco.mjv_updateScene(
            m=self._m,
            d=self._d,
            opt=self._opt,
            pert=None,
            cam=self._cam,
            catmask=mujoco.mjtCatBit.mjCAT_ALL.value,
            scn=self._scn
        )

        # Render the scene.
        mujoco.mjr_render(
            viewport=self._viewport,
            scn=self._scn,
            con=self._con
        )

        # Read the current step (1 frame)'s pixels.
        mujoco.mjr_readPixels(
            rgb=self._rgb_buffer,
            depth=self._depth_buffer,
            viewport=self._viewport,
            con=self._con
        )

        # Manage the buffers in the non-headless mode.
        if self._window_visible == 1:
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        else:
            pass

    @property
    def num_steps(self):
        """
        This property gets the number of steps assigned in the environment.
        """
        return self._num_steps

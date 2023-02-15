import numpy as np
import math

import mujoco
from mujoco.glfw import glfw

from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete


from Task import Task
from utils import debug


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

        # Env.
        grids = debug.DEBUG['grids']
        self._vacant_grid_ids = []
        for row in grids:
            for grid in row:
                if debug.DEBUG['keyword'] in grid:
                    self._vacant_grid_ids.append(int(grid.split('-')[-1]))
        grids_layout_shape = np.array(debug.DEBUG['grids'].copy()).shape
        self._num_grids = np.prod(grids_layout_shape)
        self._h_grids, self._w_grids = grids_layout_shape[0], grids_layout_shape[1]
        self._ref_grids = np.arange(self._num_grids).tolist()

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
        self.action_space = MultiDiscrete(nvec=[debug.DEBUG['dim_actions'], debug.DEBUG['dim_actions']])
        self.observation_space = Dict({
            'rgb': Box(low=-1.0, high=1.0, shape=(3, debug.DEBUG['obs_height'], debug.DEBUG['obs_width'])),
            'fix_waste_steps': MultiDiscrete([self._num_steps, self._num_steps]),
            'pre_now_focus': MultiDiscrete([self._num_grids, self._num_grids]),
            # 'grids_status': MultiDiscrete([3] * self._num_grids),   # 0 for not traversed, 1 for traversed, 2 for vacant.
        })

        # Initialize MuJoCo.
        self._m = mujoco.MjModel.from_xml_path(filename=self._mj_filename)
        self._d = mujoco.MjData(self._m)

        # Initialize the visual perception in MuJoCo.
        self._cam = mujoco.MjvCamera()  # The abstract camera.
        cam_name = debug.DEBUG['cam_name']
        if isinstance(debug.DEBUG['cam_name'], str):
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

        # TODO debug here. Generalize this part later along with the implementation of the visual perception.
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat = [0, 0, 2.5]
        self._cam.distance = 6
        self._cam.azimuth = 90.0
        self._cam.elevation = 0

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
        self._task.step(action=action, fix_steps_threshold=debug.DEBUG['fix_steps_threshold'])

        # ---------------------- MuJoCo Render: Update ----------------------
        # Update the MuJoCo model - change the color of explored/traversed grid.
        # TODO finalize this part later to facilitate video making.
        explored_grid_ids = self._task.ground_truth['explored_grid_ids'].copy()
        if len(explored_grid_ids) > len(debug.DEBUG['explored_grid_ids']):
            new_grids = [grid_id for grid_id in explored_grid_ids if grid_id not in debug.DEBUG['explored_grid_ids']]
            new_grid = new_grids[0]

            # Change the color of that.
            for row in debug.DEBUG['grids']:
                for grid_name_str in row:
                    if new_grid == int(grid_name_str.split('-')[-1]):
                        self._m.geom(grid_name_str).rgba[0:3] = debug.DEBUG['grey']
            # Update the explored_grid_ids buffer.
            debug.DEBUG['explored_grid_ids'] = explored_grid_ids
        elif len(explored_grid_ids) < len(debug.DEBUG['explored_grid_ids']):    # One loop has been finished.
            for row in debug.DEBUG['grids']:
                for grid_name_str in row:
                    if debug.DEBUG['keyword'] not in grid_name_str:
                        self._m.geom(grid_name_str).rgba[0:3] = debug.DEBUG['red']
            # Update the explored_grid_ids buffer.
            debug.DEBUG['explored_grid_ids'] = explored_grid_ids

        # Update the focus-ball's movement.
        grid_id = self._task.states['grid_id']
        for row in debug.DEBUG['grids']:
            for grid_name_str in row:
                if grid_id == int(grid_name_str.split('-')[-1]):
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
        num_vacant_grids = len(self._vacant_grid_ids)
        num_non_vacant_grids = self._num_grids - num_vacant_grids
        forward_steps = debug.DEBUG['fix_steps_threshold'] * num_non_vacant_grids + (self._h_grids // 2) * (self._w_grids - 1)
        back_steps = (min(self._h_grids, self._w_grids) - 1) + np.abs(self._h_grids - self._w_grids)
        info['optimal_loops'] = (self._num_steps - 1) // (forward_steps + back_steps)

        info['achievement'] = np.round(info['num_loops'] / info['optimal_loops'], 4)

        # ---------------------- Done -----------------------
        if self._elp_steps >= self._num_steps:
            done = True

        # ---------------------- Debug -----------------------
        if self._conf_rl['mode'] == 'debug' or self._conf_rl['mode'] == 'test':
            print('\nThe action is: {}; the previous grid id is: {}; the previous fixing steps is: {}'
                  '\nThe current grid is: {}, the fixing steps is: {}, the current reward is: {}.'
                  '\nThe unexplored grids are: {}, the explored grids are: {}'
                  ''.
                  format(action - debug.DEBUG['dim_actions'] // 2, self._task.states['pre_grid_id'], self._task.ground_truth['pre_fixing_steps'],
                         grid_id, self._task.ground_truth['fixing_steps'], reward,
                         self._task.ground_truth['unexplored_grid_ids'], self._task.ground_truth['explored_grid_ids']))

        return obs, reward, done, info

    def _get_obs(self):
        # Implicit visual perceptions.
        offset_h = debug.DEBUG['obs_height'] / 2
        offset_w = debug.DEBUG['obs_width'] / 2
        obs_idx_h = [int(self._height / 2 - offset_h), int(self._height / 2 + offset_h)]
        obs_idx_w = [int(self._width / 2 - offset_w), int(self._width / 2 + offset_w)]
        rgb = self._rgb_buffer.copy()[obs_idx_h[0]:obs_idx_h[1], obs_idx_w[0]:obs_idx_w[1], :]
        rgb_norm = (rgb.astype(float) / 255.0) * 2.0 - 1.0
        rgb_norm_transposed = np.transpose(rgb_norm, (2, 0, 1))   # Transpose from H*W*C to C*H*W.

        # Explicit task ground truth observations.
        fixing_steps = np.clip(self._task.ground_truth['fixing_steps'], 0, self._num_steps-1)
        wasted_steps = np.clip(self._task.ground_truth['wasted_steps'], 0, self._num_steps-1)
        pre_focus = np.clip(self._task.states['pre_grid_id'], 0, self._num_grids - 1)
        focus = np.clip(self._task.states['grid_id'], 0, self._num_grids - 1)

        # Get the unexplored grids.
        unexplored_grid_ids = self._task.ground_truth['unexplored_grid_ids'].copy()
        explored_grid_ids = self._task.ground_truth['explored_grid_ids'].copy()
        grids_status = [0 if grid_id in unexplored_grid_ids else 1 if grid_id in explored_grid_ids else 2 for grid_id in self._ref_grids]

        obs = {
            'rgb': rgb_norm_transposed,
            'fix_waste_steps': [fixing_steps, wasted_steps],
            'pre_now_focus': [pre_focus, focus],
            # 'grids_status': grids_status,
        }

        return obs

    def _reward_function(self):
        g_t = self._task.ground_truth
        f_s = g_t['fixing_steps']
        w_s = g_t['wasted_steps']

        basic_reward = 1
        gain_fixing = 1
        gain_wasting = -1
        gain_finish_grid = debug.DEBUG['fix_steps_threshold'] + 2
        gain_finish_loop = self._num_grids + 2

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

        # Save rgb buffers for potential video writing.
        self._append_rgb_images()

        # Manage the buffers in the non-headless mode.
        if self._window_visible == 1:
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        else:
            pass

    def _append_rgb_images(self):
        """
        This internal method generates the rgb images buffer by appending rbg_buffer step by step (1 step 1 frame).
        """
        rl = self._conf_rl
        buffer_size = 30000
        if rl['mode'] == 'test':
            if rl['train']['num_steps'] <= buffer_size:
                self._rgb_images.append(np.flipud(self._rgb_buffer).copy())
            else:
                raise ValueError('Episode length - the number of steps has exceeded the rgb images buffer size of: {}.'.format(buffer_size))

    @property
    def num_steps(self):
        """
        This property gets the number of steps assigned in the environment.
        """
        return self._num_steps

    @property
    def rgb_images(self):
        return self._rgb_images

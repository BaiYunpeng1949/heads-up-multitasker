import math

import gym
import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context
from collections import deque

READ = 'reading'
BG = 'background'
RELOC = 'relocating'


class RelocationStackFrame(Env):

    def __init__(self):
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config['rl']['mode']

        # Open the mujoco model
        self._xml_path = os.path.join(directory, "reloc-stack-frame.xml")
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data = mujoco.MjData(self._model)
        # Forward pass to initialise the model, enable all variables
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Initialise thresholds and counters
        self._steps = None
        self._ep_len = 100
        self._trials = None
        self._bg_trials = None
        self._max_trials = 1
        self._read_steps = None
        self._bg_steps = None
        self._reloc_steps = None
        self._task_mode = None

        # Get the primitives idxs in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")
        self._bgp_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get targets (geoms that belong to "smart-glass-pane")
        # Inter-line-spacing-100
        self._ils100_read_idxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_idx)[0]

        # Define the reading target idxs
        self._read_target_idx = None
        # Define the default text grid size and rgba from a sample grid idx=0, define the hint text size and rgba
        sample_grid_idx = self._ils100_read_idxs[0].copy()
        self._DFLT_READ_CELL_SIZE = self._model.geom(sample_grid_idx).size[0:4].copy()
        self._DFLT_READ_CELL_RGBA = [0, 0, 0, 1]
        self._HINT_READ_CELL_SIZE = [self._DFLT_READ_CELL_SIZE[0] * 4 / 3, self._DFLT_READ_CELL_SIZE[1],
                                     self._DFLT_READ_CELL_SIZE[2] * 4 / 3]
        self._HINT_READ_CELL_RGBA = [1, 1, 0, 1]
        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._read_dwell_steps = int(2 * self._action_sample_freq)

        # Get the background (geoms that belong to "background-pane")
        # background_idxs = np.where(self._model.geom_bodyid == self._bgp_body_idx)[0]
        self._bg_target_idx = np.where(self._model.geom_bodyid == self._bgp_body_idx)[0][0].copy()
        # Define the default background grid size and rgba from a sample grid idx=0, define the event text size and rgba
        self._DFLT_BG_SIZE = self._model.geom(self._bg_target_idx).size[0:3].copy()
        self._DFLT_BG_RGBA = self._model.geom(self._bg_target_idx).rgba[0:4].copy()
        self._HINT_BG_RGBA = [1, 0, 0, 1]
        # Define the events on the background pane
        self._bg_dwell_steps = int(self._read_dwell_steps * 0.25)

        # Define the task switch from reading to background event - the interruption timestep
        if self._max_trials <= 1:
            self._itrpt_read_steps = int(self._read_dwell_steps * 0.5)
        else:
            NotImplementedError('Not implemented for max_reading_trials > 1')

        # Define the relocation dwell timesteps
        self._reloc_dwell_steps = 4

        # Define the frame stack
        self._vision_frames = None
        self._qpos_frames = None
        self._steps_since_last_frame = 0
        self._num_stacked_frames = 2

        # Define observation space
        self._width = self._config['mj_env']['width']
        self._height = self._config['mj_env']['height']
        # self.observation_space = Box(low=-1, high=1, shape=(3 * self._num_stacked_frames, self._width, self._height))  # C*W*H
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stacked_frames, self._width, self._height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stacked_frames * self._model.nq + self._model.nu,))}
        )

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    def _get_obs(self):
        # TODO try frame skip and stack frame together - more efficient and might reduce the ocsillation of the eyeball
        #  Ref https://www.reddit.com/r/reinforcementlearning/comments/fucovf/comment/fmc25er/?utm_source=share&utm_medium=web2x&context=3

        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        # Ref - Q-Net paper - https://docs.google.com/presentation/d/14Pjd4twtDLBOcFq6Eo0Veyz3KE5uqjyt770FSQj5Q3E/edit#slide=id.g21d3b6ec6ec_0_43
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :, :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)

        # Stack the frames only in the reading mode, because they are the most useful information
        if self._task_mode == READ:
            self._vision_frames.append(gray_normalize)
            self._qpos_frames.append(self._data.qpos.copy())
            # Replicate the newest frame if the stack is not full
            while len(self._vision_frames) < self._num_stacked_frames:
                self._vision_frames.append(self._vision_frames[-1])
            while len(self._qpos_frames) < self._num_stacked_frames:
                self._qpos_frames.append(self._qpos_frames[-1])

        # Update only the latest frame in background mode and relocation mode
        else:
            self._vision_frames[-1] = gray_normalize
            self._qpos_frames[-1] = self._data.qpos.copy()

        # Reshape to the observation space shape
        # TODO learn the most recognized method to play with stacked frames tmr
        #  Ref https://stats.stackexchange.com/questions/406213/dqn-how-to-feed-the-input-of-4-still-frames-from-a-game-as-one-single-state-in
        #  Ref https://discuss.pytorch.org/t/how-can-i-process-stack-of-frames/164473
        vision = np.stack(self._vision_frames, axis=0)
        vision = vision.reshape((-1, vision.shape[-2], vision.shape[-1]))
        qpos = np.stack(self._qpos_frames, axis=0)
        qpos = qpos.reshape((1, -1))
        ctrl = self._data.ctrl.reshape((1, -1))

        # Get joint values (qpos) and motor set points (ctrl) -- call them proprioception for now
        # Ref - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/envs/context_switch_replication/SwitchBackLSTM.py#L96
        proprioception = np.concatenate([qpos.flatten(), ctrl.flatten()], axis=0)

        return {"vision": vision, "proprioception": proprioception}

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0
        self._trials = 0
        self._bg_trials = 0
        self._read_steps = 0
        self._bg_steps = 0
        self._reloc_steps = 0
        self._task_mode = READ
        self._vision_frames = deque(maxlen=self._num_stacked_frames)
        self._qpos_frames = deque(maxlen=self._num_stacked_frames)

        # Initialize eye ball rotation angles
        self._data.qpos[self._eye_joint_x_idx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_idx] = np.random.uniform(-0.5, 0.5)

        # Initialize the reading targets
        reading_target_idxs = self._ils100_read_idxs.copy()

        # Reset the all reading cells - hide
        for idx in reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DFLT_READ_CELL_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DFLT_READ_CELL_SIZE.copy()
        # Reset the background scene
        self._model.geom(self._bg_target_idx).rgba[0:4] = self._DFLT_BG_RGBA.copy()

        # Randomize the target cell - randomly choose one of the reading targets
        self._read_target_idx = np.random.choice(reading_target_idxs)

        # Highlight the read target cell
        self._model.geom(self._read_target_idx).rgba[0:4] = self._HINT_READ_CELL_RGBA.copy()
        self._model.geom(self._read_target_idx).size[0:3] = self._HINT_READ_CELL_SIZE.copy()

        return self._get_obs()

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _get_focus(self, site_name):
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = site.xmat.reshape((3, 3))[:, 2]
        # Exclude the body that contains the site, like in the rangefinder sensor
        bodyexclude = self._model.site_bodyid[site.id]
        geomid_out = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            self._model, self._data, pnt, vec, geomgroup=None, flg_static=1,
            bodyexclude=bodyexclude, geomid=geomid_out)
        return distance, geomid_out[0]

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reward
        reward = 0

        if self._task_mode == READ:
            if geomid == self._read_target_idx:
                reward = 1
                self._read_steps += 1
            # Interrupt the reading task and flip to the background dwell task
            if self._read_steps >= self._itrpt_read_steps and self._bg_trials < self._max_trials:
                self._task_mode = BG
                # Hide the reading cell
                self._model.geom(self._read_target_idx).rgba[0:4] = self._DFLT_READ_CELL_RGBA.copy()
                self._model.geom(self._read_target_idx).size[0:3] = self._DFLT_READ_CELL_SIZE.copy()
                # Highlight the background target with the hint color
                self._model.geom(self._bg_target_idx).rgba[0:4] = self._HINT_BG_RGBA.copy()
            # Terminate the reading task if the reading task is done
            if self._read_steps >= self._read_dwell_steps:
                self._trials += 1

        elif self._task_mode == BG:
            if geomid == self._bg_target_idx:
                reward = 1
                self._bg_steps += 1
            # Flip to the relocation task
            if self._bg_steps >= self._bg_dwell_steps:
                self._task_mode = RELOC
                self._model.geom(self._bg_target_idx).rgba[0:4] = self._DFLT_BG_RGBA.copy()
                # Update the background event trial counter
                self._bg_trials += 1

        elif self._task_mode == RELOC:
            if geomid == self._read_target_idx:
                reward = 1
                self._reloc_steps += 1
            # Resume the reading task
            if self._reloc_steps >= self._reloc_dwell_steps:
                self._model.geom(self._read_target_idx).rgba[0:4] = self._HINT_READ_CELL_RGBA.copy()
                self._model.geom(self._read_target_idx).size[0:3] = self._HINT_READ_CELL_SIZE.copy()
                self._task_mode = READ
        else:
            NotImplementedError(f'Unknown task mode: {self._task_mode}')

        # Check whether to terminate the episode
        terminate = False
        if self._steps >= self._ep_len or self._trials >= self._max_trials:
            terminate = True

        return self._get_obs(), reward, terminate, {}

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


class ZReadBase(Env):

    def __init__(self):
        """ Model the Zigzag reading as a MDP - read as fast and accurate as possible """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "zread-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")

        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_idxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_idx)[0]

        # Define the target in this z-reading task
        self._toread_idxs = None
        self._target_idx = None         # The current cell - target to read
        self._HINT_RGBA = [1, 0, 1, 1]
        self._DFLT_RGBA = [1, 1, 1, 1]

        self._dwell_steps = int(2 * self._action_sample_freq)  # 2 seconds
        self._rgba_diff_ps = float((self._DFLT_RGBA[1] - self._HINT_RGBA[1]) / self._dwell_steps)

        # Define the observation space
        # Origin - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/c9ef14a91febfcb258c4990ebef2246c972e8aaa/huc/envs/locomotion/RelocationStackFrame.py#L111
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 3     # TODO adjust this later
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
            })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

        # Initialise thresholds and counters
        self.ep_len = 600
        self._steps = None
        self._on_target_steps = None
        self._off_target_steps = None
        self._num_read_cells = None     # Cells are already been read
        self._max_toread_cells = 5
        self._max_off_target_steps = int(2 * self._action_sample_freq)  # 2 seconds

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment TODO stack frame if necessary"""
        # Get the vision observation
        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess - H*W*C -> C*W*H
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)

        # Convert the rgb to grayscale - boost the training speed
        gray_normalize = rgb_normalize[0:1, :, :] * 0.299 + rgb_normalize[1:2, :, :] * 0.587 + rgb_normalize[2:3, :, :] * 0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1] - TODO adjust this later
        remaining_off_steps = self._off_target_steps / self._max_off_target_steps * 2 - 1
        remaining_dwell_steps = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        remaining_cells = (self._max_toread_cells - self._num_read_cells) / self._max_toread_cells * 2 - 1
        stateful_info = np.array([remaining_off_steps, remaining_dwell_steps, remaining_cells])
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0
        self._off_target_steps = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_idx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_idx] = np.random.uniform(-0.5, 0.5)

        # Initialize the targets
        # Randomly select a number of cells as the target cells
        self._toread_idxs = np.random.choice(self._ils100_cells_idxs.copy(), size=self._max_toread_cells, replace=False)
        self._target_idx = self._toread_idxs[0]

        self._num_read_cells = 0

        # Initialize and render all cells before the target cell as read, and the rest as unread
        for idx in self._ils100_cells_idxs.copy():
            if idx == self._target_idx:
                self._model.geom(idx).rgba[0:4] = self._HINT_RGBA.copy()
            else:
                self._model.geom(idx).rgba[0:4] = self._DFLT_RGBA.copy()

        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

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

    @staticmethod
    def angle_between(v1, v2):
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec / np.linalg.norm(vec)

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_target(self, site_name, target_idx):
        """
        Return the angle between the vector pointing from the site to the target and the vector pointing from the site to the front
        ranges from 0 to pi.
        """
        # Get vector pointing direction from site
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = pnt + site.xmat.reshape((3, 3))[:, 2]

        # Get vector pointing direction to target
        target_vec = self._data.geom(target_idx).xpos - pnt

        # Estimate distance as angle
        angle = self.angle_between(vec, target_vec)

        return angle

    def _get_next_target(self):
        # Reset the counter
        self._on_target_steps = 0
        self._off_target_steps = 0
        # Reset the rendering scene
        self._model.geom(self._target_idx).rgba[0:4] = self._DFLT_RGBA.copy()
        # Update the number of remaining unread cells
        self._num_read_cells += 1
        # Update the target
        # Get self._target_idx's index in self._toread_idxs
        idx = np.where(self._toread_idxs == self._target_idx)[0][0]
        self._target_idx = self._toread_idxs[idx + 1] if idx + 1 < self._max_toread_cells else self._toread_idxs[-1]  # TODO there is a buggy state here
        # Update the scene
        self._model.geom(self._target_idx).rgba[0:4] = self._HINT_RGBA.copy()

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

        # Apply the transition function - update the scene regarding the actions
        if geomid == self._target_idx:
            self._on_target_steps += 1
            self._model.geom(self._target_idx).rgba[1] += self._rgba_diff_ps
        else:
            self._off_target_steps += 1

        # Get rewards
        reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._target_idx)) - 1)

        # Update the transitions
        if self._on_target_steps >= self._dwell_steps:
            # Update the milestone bonus reward for finish reading a cell
            reward = 10
            # Get the next target
            self._get_next_target()

        # Early termination
        if self._off_target_steps >= self._max_off_target_steps:    # TODO I think the model was trapped in a local sub-optimal by ending the episode ASAP by using early termination this way - continuous negative rewards.
            # Get the next target
            self._get_next_target()

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        # Get termination condition
        terminate = False
        if self._num_read_cells >= self._max_toread_cells:
            terminate = True

        return self._get_obs(), reward, terminate, {}


# class ZReadTrain(ZReadBase)
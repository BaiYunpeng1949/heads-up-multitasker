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

READ = 'read'
BG = 'background'
RELOC = 'relocation'


class Read(Env):

    def __init__(self):
        """ Model the reading with belief model/function """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "attention-switch-v1.xml"))
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

        # Define the target idx probability distribution
        self._target_idx_prob = None           # The dynamic target idx probability distribution, should be updated with transition function
        self._sampled_target_idx = None     # The sampled target idx
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(2 * self._action_sample_freq)  # 2 seconds

        # Define the observation space
        # Origin - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/c9ef14a91febfcb258c4990ebef2246c972e8aaa/huc/envs/locomotion/RelocationStackFrame.py#L111
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 4
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
        self._steps = None
        self._on_target_steps = None
        self._num_trials = None     # Cells are already been read
        self._max_trials = 5     # Maximum number of cells to read - more trials in one episode will boost the convergence

        self.ep_len = int(self._max_trials * self._dwell_steps * 2)

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
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

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        sampled_target_idx_norm = self.normalise(self._sampled_target_idx, self._ils100_cells_idxs[0], self._ils100_cells_idxs[-1], -1, 1)
        stateful_info = np.array([remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, sampled_target_idx_norm])

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0
        self._num_trials = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_idx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_idx] = np.random.uniform(-0.5, 0.5)

        # Update the target idx probability distribution
        self._update_target_idx_prob(mj_target_idx=np.random.choice(self._ils100_cells_idxs.copy()))

        # Sample a target according to the target idx probability distribution
        self._sample_target()

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

    def _update_target_idx_prob(self, mj_target_idx):
        """TODO the memory and belief model will be built upon this later"""
        # Initialize the target idx probability distribution - make sure probability sum to 1
        self._target_idx_prob = np.zeros(len(self._ils100_cells_idxs.copy()))
        # Get the idx of mj_target_idx in self._ils100_cells_idxs
        idx = np.where(self._ils100_cells_idxs == mj_target_idx)[0][0]
        # Update the target idx probability distribution by assigning the probability of the target idx at the given mj_target_idx to be 1
        self._target_idx_prob[idx] = 1
        # Check whether the probability sum to 1
        if np.sum(self._target_idx_prob) != 1:
            raise ValueError("The target idx probability distribution does not sum to 1!")

    def _sample_target(self):
        # Sample a target from the cells according to the target idx probability distribution
        self._sampled_target_idx = np.random.choice(self._ils100_cells_idxs.copy(), p=self._target_idx_prob)

        # Reset the counter
        self._on_target_steps = 0

        # Update the number of remaining unread cells
        self._num_trials += 1

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

        # Reset the scene first
        for mj_idx in self._ils100_cells_idxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Apply the transition function - update the scene regarding the actions
        if geomid == self._sampled_target_idx:
            self._on_target_steps += 1
            self._model.geom(self._sampled_target_idx).rgba = self._VISUALIZE_RGBA

        # Update the transitions - get rewards and next state
        if self._on_target_steps >= self._dwell_steps:
            # Update the milestone bonus reward for finish reading a cell
            reward = 10
            # Update the target idx probability distribution - randomly choose a new target idx
            self._update_target_idx_prob(mj_target_idx=np.random.choice(self._ils100_cells_idxs.copy()))
            # Get the next target
            self._sample_target()
        else:
            reward = 0.1 * (np.exp(
                -10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._sampled_target_idx)) - 1)

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}


class AttentionSwitch(Env):

    def __init__(self):
        """ Model the reading, attention switch, switch back, relocation, resume reading with a belief model/function """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "attention-switch-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                        "smart-glass-pane-interline-spacing-100")
        self._bg_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get MuJoCo cell idxs (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]
        # Get MuJoCo background idx (geoms that belong to "background-pane")
        self._bg_pane_mjidxs = np.where(self._model.geom_bodyid == self._bg_body_mjidx)[0]
        self._bg_mjidx = self._bg_pane_mjidxs[0]
        # Concatenate cell idxs and background idx
        self._fixations_mjidxs = np.concatenate((self._ils100_cells_mjidxs, self._bg_mjidx))

        # Define the target idx probability distribution
        self._true_target_idx = None               # The true target MuJoCo idx
        self._sampled_target_mjidx = None          # The target MuJoCo idx, should be sampled from memory
        self._target_mjidx_belief = None           # 'Belief': The dynamic target MuJoCo idx probability distribution, should be updated at each step
        self._sampled_fixation_mjidx = None        # The fixation MuJoCo idx, should be sampled from belief

        self._VIS_CELL_RGBA = [1, 1, 0, 1]
        self._DFLT_CELL_RGBA = [0, 0, 0, 1]
        self._VIS_BG_RGBA = [1, 0, 0, 1]
        self._DFLT_BG_RGBA = [1, 1, 1, 0]

        self._dwell_cell_steps = int(2 * self._action_sample_freq)  # 2 seconds
        self._dwell_bg_steps = int(1 * self._action_sample_freq)  # 1 second
        self._dwell_reloc_steps = int(1 * self._action_sample_freq)  # 1 second     # TODO use Hick's law to determine this

        # Task mode
        self._task_mode = None

        # Attention switch flag on a certain trial
        self._attention_switch = None

        # Determine the radian of the visual spotlight for visual search, or 'neighbors'
        self._neighbour_radius = 0.0101     # Obtained empirically
        self._neighbors_mjidxs = None       # The MuJoCo idxs of the neighbors of the sampled target idx

        # Initialise thresholds and counters
        self._steps = None
        self._fixation_steps = None
        self._num_trials = None
        self._max_trials = 5
        self.ep_len = int(self._max_trials * self._dwell_cell_steps * 2)

        # Define the observation space
        width, height = self._config['mj_env']['width'], self._config['mj_env']['height']
        self._num_stk_frm = 1
        self._num_stateful_info = 4
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

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
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

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_cell_steps - self._fixation_steps) / self._dwell_cell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        sampled_target_mjidx_norm = self.normalise(self._sampled_target_mjidx, self._ils100_cells_mjidxs[0], self._ils100_cells_mjidxs[-1], -1, 1)
        stateful_info = np.array([remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, sampled_target_mjidx_norm])

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._num_trials = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_mjidx] = np.random.uniform(-0.5, 0.5)

        # Initialize whether this trial includes the attention switch
        # TODO to simplify, the attention switch always happen after a cell has been fixated for enough time
        self._attention_switch = np.random.choice([True, False])

        # Initialize the task mode
        self._task_mode = READ

        # Initialize the target mjidx belief - probability distribution
        self._target_mjidx_belief = np.zeros(len(self._fixations_mjidxs))

        # Update the target idx probability distribution - randomly select one target
        # TODO later sample from the memory function
        self._update_target_mjidx_belief(target_mjidx=np.random.choice(self._ils100_cells_mjidxs.copy()))

        # Sample a target according to the target idx probability distribution
        self._sample_fixation()

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

    def _update_target_mjidx_belief(self, target_mjidx, fixation_mjidx=None):
        """TODO update the target mjidx under different tasks"""
        # TODO use the fixation_mjidx to update the target mjidx belief
        if self._task_mode == READ or self._task_mode == BG:
            # Determinist belief
            if fixation_mjidx == target_mjidx:
                # Reset the target idx probability distribution to all 0 in the reading mode
                self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                # Allocate the probability of the sampled target mjidx to be 1
                idx = np.where(self._ils100_cells_mjidxs == target_mjidx)[0][0]
                self._target_mjidx_belief[idx] = 1
        elif self._task_mode == RELOC:
            # Initializations - Determine the neighbours by identifying all cells that are within the preset neighbour radius
            # TODO maybe move this to somewhere else
            if fixation_mjidx == None:
                self._neighbors_mjidxs = []
                center_xpos = self._data.geom(target_mjidx).xpos
                for mjidx in self._ils100_cells_mjidxs:
                    xpos = self._data.geom(mjidx).xpos
                    dist = np.linalg.norm(xpos - center_xpos)
                    if dist <= self._neighbour_radius:
                        self._neighbors_mjidxs.append(mjidx)
                # Leak probability to the neighbours
                # TODO make it a function of the elapsed time later
                center_prob = 0.5
                leak_prob_per_neighbour = float((1 - center_prob) / (len(self._neighbors_mjidxs) - 1))
                # Assign the probability to the neighbours and the center
                for mjidx in self._neighbors_mjidxs:
                    if mjidx == target_mjidx:
                        idx = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[idx] = center_prob
                    else:
                        idx = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[idx] = leak_prob_per_neighbour

            # Update the belief of according to actual fixations
            else:   # TODO fix the bug later.
                idx = np.where(self._fixations_mjidxs == fixation_mjidx)[0][0]
                if fixation_mjidx != target_mjidx:
                    prob = self._target_mjidx_belief[idx]
                    # Set 0 to the current fixation mjidx
                    self._target_mjidx_belief[idx] = 0
                    # Remove it from the neighbors_mjidx
                    self._neighbors_mjidxs.remove(fixation_mjidx)
                    # Reallocate the probability to assure they sum to 1
                    leak_prob_per_neighbour = float(prob / (len(self._neighbors_mjidxs)))
                    for mjidx in self._neighbors_mjidxs:
                        idx = np.where(self._fixations_mjidxs == mjidx)[0][0]
                        self._target_mjidx_belief[idx] += leak_prob_per_neighbour
                    # Make sure they sum to 1
                    self._target_mjidx_belief /= np.sum(self._target_mjidx_belief)
                else:
                    self._target_mjidx_belief = np.zeros(self._target_mjidx_belief.shape[0])
                    self._target_mjidx_belief[idx] = 1
                    # TODO find the sampled target mjidx and set the prob to 1

        else:
            raise ValueError("The task mode is not correct!")

    def _sample_target(self):
        if self._task_mode == READ:
            self._sampled_target_mjidx = np.random.choice(self._ils100_cells_mjidxs.copy())
        elif self._task_mode == BG:
            self._sampled_target_mjidx = self._bg_mjidx
        elif self._task_mode == RELOC:
            # Update the memory function - belief on which cell should be the target mjidx
            # TODO to start with a simplified version, the target mjidx are determinist - the cut off cell/word
            self._sampled_target_mjidx = self._true_target_idx

        return self._sampled_target_mjidx

    def _sample_fixation(self):
        # Sample a fixation mjidx from the cells according to the target mjidx probability distribution (belief)
        self._sampled_fixation_mjidx = np.random.choice(self._ils100_cells_mjidxs.copy(), p=self._target_mjidx_belief)
        # Reset the counter
        self._fixation_steps = 0

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

        # Reset the scene first - TODO optimize this part for the training speed
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_CELL_RGBA
        self._model.geom(self._bg_body_mjidx).rgba = self._DFLT_BG_RGBA

        # Step with different task modes
        if self._task_mode == READ:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_CELL_RGBA

            if self._fixation_steps >= self._dwell_cell_steps:
                # Update the milestone bonus reward for finish reading a cell
                reward = 10

                # Check whether to change the task mode
                if self._attention_switch == True:
                    self._task_mode = BG
                    self._true_target_idx = self._sampled_target_mjidx
                    # Update the target idx probability distribution - choose the background pane
                    self._update_target_mjidx_belief(target_mjidx=self._sample_target())
                    # Get the next target
                    self._sample_fixation()
                else:
                    # Update the target idx probability distribution - randomly choose a new target idx
                    self._update_target_mjidx_belief(target_mjidx=self._sample_target())
                    # Get the next target
                    self._sample_fixation()
                    # Update the number of trials
                    self._num_trials += 1
            else:
                reward = 0.1 * (np.exp(-10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._sampled_target_mjidx)) - 1)
        elif self._task_mode == BG:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
                self._model.geom(self._sampled_fixation_mjidx).rgba = self._VIS_BG_RGBA
            if self._fixation_steps >= self._dwell_bg_steps:
                reward = 10
                self._task_mode = RELOC
                # Update the target idx probability distribution - randomly choose a new target idx
                self._update_target_mjidx_belief(target_mjidx=self._sample_target())

                # Check whether the prob becomes deterministic by checking whether the max prob is 1
                if np.max(self._target_mjidx_belief) == 1:
                    self._task_mode = READ

                # Get the next target
                self._sample_fixation()

                # Update the number of trials
                self._num_trials += 1

        elif self._task_mode == RELOC:
            if geomid == self._sampled_fixation_mjidx:
                self._fixation_steps += 1
            if self._fixation_steps >= self._dwell_reloc_steps:
                self._update_target_mjidx_belief(target_mjidx=self._sampled_target_mjidx, fixation_mjidx=geomid)
                self._sample_fixation()     # TODO finish this later
        else:
            raise ValueError("The task mode is not correct!")

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True

        # Update the scene to reflect the transition function
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), reward, terminate, {}
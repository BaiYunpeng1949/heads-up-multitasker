import numpy as np
from collections import Counter, deque
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from collections import deque
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class MDPRead(Env):

    def __init__(self):
        """ Model sequential reading - read words in the layout one by one """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "mobile-read-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the joints idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")

        # Get the motors idx in MuJoCo
        self._eye_x_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-x-motor")
        self._eye_x_motor_translation_range = self._model.actuator_ctrlrange[self._eye_x_motor_mjidx]
        self._eye_y_motor_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye-y-motor")
        self._eye_y_motor_translation_range = self._model.actuator_ctrlrange[self._eye_y_motor_mjidx]

        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-interline-spacing-100")
        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]

        self._MEMORY_PAD_VALUE = -2
        # The pool of targets that can be attended to, -1 stands for attention switch away from reading
        self._attention_pool = np.array([self._MEMORY_PAD_VALUE, *self._ils100_cells_mjidxs.copy()])

        # Mental state related variables
        self._attention_deploy_target_mjidx = None
        self._mental_state = {   # The mental state of the agent, or it can be called the internal state
            'reading_memory': deque(maxlen=len(self._ils100_cells_mjidxs)),     # The memory of the words that have been read
            'attention': self._attention_deploy_target_mjidx,                          # The intended target mjidx
        }

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]     # TODO later use this to hint agents where to look at
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(0.05 * self._action_sample_freq)  # The number of steps to dwell on a target

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self.ep_len = 200

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._vision_frames = None
        self._qpos_frames = None
        self._num_stateful_info = 15

        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info,))

        # Define the action space
        # 1st: decision on attention distribution;
        # 2nd and 3rd: eyeball rotations;
        self._action_attention_deployment_idx = 0
        self._action_eye_rotate_x_idx = 1
        self._action_eye_rotate_y_idx = 2
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def _get_obs(self):
        """ Get the observation of the environment """
        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        attention_deploy_target_mjidx_norm = self.normalise(self._attention_deploy_target_mjidx, self._attention_pool[0], self._attention_pool[-1], -1, 1)

        reading_memory_norm = np.zeros((len(self._ils100_cells_mjidxs)))
        for i in range(len(self._ils100_cells_mjidxs)):
            if i <= len(self._mental_state['reading_memory']) - 1:
                reading_memory_norm[i] = self.normalise(self._mental_state['reading_memory'][i], self._ils100_cells_mjidxs[0], self._ils100_cells_mjidxs[-1], 0, 1)
            else:
                reading_memory_norm[i] = -1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, attention_deploy_target_mjidx_norm, *reading_memory_norm]
        )

        # Observation space check
        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information observation is not correct!")

        # TODO later when resuming the vision inputs for training agent how to look at deployed attention target,
        #  make the intended target yellow for faster eyeball rotation learning in the vision channel
        #  -  for attention deployment

        return stateful_info

    def reset(self, params=None):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_target_steps = 0

        # Mental state related variables
        # Randomly choose the intended target mjidx
        self._attention_deploy_target_mjidx = np.random.choice(self._ils100_cells_mjidxs)
        self._mental_state = {
            # The memory of the words that have been read
            'reading_memory': deque(maxlen=len(self._ils100_cells_mjidxs)),
            'attention': self._attention_deploy_target_mjidx,  # The intended target mjidx
            # TODO initialize the reading progress randomly later, but firstly just train simply, from end to end
        }

        # Set up the whole scene by confirming the initializations
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

    def step(self, action):
        # TODO short term memory can help inhibit revisits to the same cell, but we can set my own short term memory length free parameter tau
        # Action t+1 from state t - attention allocation - where should the agent look at - Reading is a sequential process (MDP)
        action[self._action_attention_deployment_idx] = self.normalise(action[self._action_attention_deployment_idx], -1, 1, 0, len(self._attention_pool))
        finish_reading = False
        # Only if the previous intended target is processed, then the agent can process the next intended target
        if self._on_target_steps >= self._dwell_steps:
            # Decode the attention allocation from continuous value to discrete attention targets
            # Clip the attention allocation to the attention pool
            attention_pool_idx = np.clip(int(np.floor(action[self._action_attention_deployment_idx])), 0, len(self._attention_pool) - 1)
            discrete_attention = self._attention_pool[attention_pool_idx]
            self._attention_deploy_target_mjidx = discrete_attention
            if discrete_attention == self._attention_pool[0]:
                finish_reading = True
            # Refresh the on target steps
            self._on_target_steps = 0
            # Update the mental state
            self._mental_state['attention'] = self._attention_deploy_target_mjidx.copy()

        # Eyeball movement
        action[self._action_eye_rotate_x_idx] = self.normalise(action[self._action_eye_rotate_x_idx], -1, 1, *self._model.actuator_ctrlrange[self._eye_x_motor_mjidx, :])
        action[self._action_eye_rotate_y_idx] = self.normalise(action[self._action_eye_rotate_y_idx], -1, 1, *self._model.actuator_ctrlrange[self._eye_y_motor_mjidx, :])
        # TODO uncomment later
        # self._data.ctrl[self._eye_x_motor_mjidx] = action[self._eye_x_action_idx]
        # self._data.ctrl[self._eye_y_motor_mjidx] = action[self._eye_y_action_idx]

        if not finish_reading:
            xpos = self._data.geom(self._attention_deploy_target_mjidx).xpos
            x, y, z = xpos[0], xpos[1], xpos[2]
            # TODO delete later
            self._data.ctrl[self._eye_x_motor_mjidx] = np.arctan(z/y)
            self._data.ctrl[self._eye_y_motor_mjidx] = np.arctan(-x/y)

        # Advance the simulation - State t+1
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State t+1
        reward = 0
        terminate = False
        info = {}

        # Eyeball movement driven by the intention / attention allocation
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")
        # Reset the scene first
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Count the number of steps that the agent fixates on the target
        if geomid == self._attention_deploy_target_mjidx:
            self._on_target_steps += 1
            self._model.geom(self._attention_deploy_target_mjidx).rgba = self._VISUALIZE_RGBA

        # Update the transitions - update the mental state
        if self._on_target_steps >= self._dwell_steps:
            self._mental_state['reading_memory'].append(self._attention_deploy_target_mjidx)

        # Estimate the reward
        reading_progress_seq = np.array(self._mental_state['reading_memory'])
        # Pad the reading progress sequence with -2 to the length of the ILS-100 cells
        reading_progress_seq = np.pad(reading_progress_seq, (0, len(self._ils100_cells_mjidxs) - len(reading_progress_seq)), 'constant', constant_values=self._MEMORY_PAD_VALUE)
        distance = self.euclidean_distance(reading_progress_seq, self._ils100_cells_mjidxs)

        reward += 0.1 * (np.exp(-0.1 * distance) - 1)

        # TODO reward shaping or modeling adjustment
        # According to Adaptive feature guidance: Modelling visual search with graphical layouts
        # The visual STM (VSTM) tries to inhibit visits to already visited locations.
        # So I should encourage the agent to visit the locations that have not been visited yet.
        # But the TODO here is: whether to hard code it or embed it in the reward function

        # If all materials are read, give a big bonus reward
        if finish_reading or self._steps >= self.ep_len:
            terminate = True
            # Successfully comprehend the text
            if np.array_equal(reading_progress_seq, self._ils100_cells_mjidxs):
                reward += 20
            else:
                reward += -20

        return self._get_obs(), reward, terminate, info

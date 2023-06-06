import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class Read(Env):

    def __init__(self):
        """ Model the reading under stationary and mobile (cause perturbations) conditions """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "mobile-read-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._perturbation_joint_z_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "perturbation-joint-z")
        self._perturbation_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT,
                                                             "perturbation-joint-x")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")

        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        self._ils100_cells_mjidxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_mjidx)[0]

        self._sampled_target_mjidx = None

        # Define the target idx probability distribution
        self._VISUALIZE_RGBA = [1, 1, 0, 1]
        self._DFLT_RGBA = [0, 0, 0, 1]

        self._dwell_steps = int(1 * self._action_sample_freq)  # 1 seconds

        # Initialize task related parameters
        self._MODES = ["stationary", "mobile"]
        self._mode = None
        # Initialize the perturbation parameters
        # The exact amplitude is difficult to specify without direct measurement,
        # but some studies suggest a range of 2-5 cm for normal walking.
        # This is also dependent on a person's height, stride length, and other factors.
        self._perturbation_peak = 0.005
        self._perturbation_period_z = int(0.5 * self._action_sample_freq)   # 1 seconds for normal human gait cycle modeled as a sine wave - 2 feet
        self._perturbation_period_x = int(0.25 * self._action_sample_freq)  # 0.5 seconds for normal human gait cycle modeled as a sine wave
        self._perturbation_velocity = None
        self._perturbation_amplitude = None

        # Oculomotor control related parameters
        # Start from action noises - the integration of oculomotor noise and the drifts after fixations.
        # TODO the current assumption is that the saccades on reading are not ballistic but under active and continuous control,
        #  now the simplified version is the saccade is a stepwise movement, every time step contains a saccade.
        # The oculomotor noise is formalized as SDN, the zero-mean Gaussian noise with a standard deviation of
        # the signal proportional to the magnitude of the signal itself.
        self._rho_ocular_motor = None
        self._rho_drift = None

        # Eye movement bounds related parameters
        # Saccade speed in reading task is 7 to 9 degrees per 200 to 250ms,
        # here I use a representative value of 35 to 36 degrees per second.
        # This applies to both horizontal and vertical saccades
        self._saccade_stepwise_speed_bounds = np.array([-35, 35]) * np.pi / 180 / self._action_sample_freq

        # The fatigue related parameters
        self._ctrl_list = None

        # Initialise RL related thresholds and counters
        self._steps = None
        self._on_target_steps = None
        self._num_trials = None  # Cells are already been read
        self._max_trials = 5  # Maximum number of cells to read - more trials in one episode will boost the convergence
        if self._config["rl"]["mode"] == "test":
            self._max_trials = len(self._ils100_cells_mjidxs)
        self.ep_len = int(self._max_trials * self._dwell_steps * 5)

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._num_stateful_info = 5
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(self._model.nu,))

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
        gray_normalize = rgb_normalize[0:1, :, :]*0.299 + rgb_normalize[1:2, :, :]*0.587 + rgb_normalize[2:3, :, :]*0.114
        gray_normalize = np.squeeze(gray_normalize, axis=0)
        vision = gray_normalize.reshape((-1, gray_normalize.shape[-2], gray_normalize.shape[-1]))

        # Get the proprioception observation
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        # Get the stateful information observation - normalize to [-1, 1]
        remaining_ep_len_norm = (self.ep_len - self._steps) / self.ep_len * 2 - 1
        remaining_dwell_steps_norm = (self._dwell_steps - self._on_target_steps) / self._dwell_steps * 2 - 1
        remaining_trials_norm = (self._max_trials - self._num_trials) / self._max_trials * 2 - 1
        sampled_target_idx_norm = self.normalise(self._sampled_target_mjidx, self._ils100_cells_mjidxs[0],
                                                 self._ils100_cells_mjidxs[-1], -1, 1)
        mode_norm = -1 if self._mode == self._MODES[0] else 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_dwell_steps_norm, remaining_trials_norm, sampled_target_idx_norm,
             mode_norm]
        )

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

        self._ctrl_list = []

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = np.random.uniform(-0.5, 0.5)
        self._data.qpos[self._eye_joint_y_mjidx] = np.random.uniform(-0.5, 0.5)

        self._mode = np.random.choice(self._MODES)
        if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
            self._mode = self._MODES[0]
            print(f"NOTE, the current mode is: {self._mode}")

        # Initialize the ocularmotor noise proportion,
        # 0.08 for the stationary mode (prior work), 0.16 for the moving mode (TODO hyperparameter tuning)
        self._rho_ocular_motor = 0.08 if self._mode == self._MODES[0] else 0.16
        self._rho_drift = 0.1 if self._mode == self._MODES[0] else 0.2

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

    def _sample_target(self):
        # Sample a target from the cells according to the target idx probability distribution
        new_sampled_target_mjidx = np.random.choice(self._ils100_cells_mjidxs.copy())

        # Make sure the sampled target is different from the previous one
        while True:
            if new_sampled_target_mjidx != self._sampled_target_mjidx:
                break
            else:
                new_sampled_target_mjidx = np.random.choice(self._ils100_cells_mjidxs.copy())
        self._sampled_target_mjidx = new_sampled_target_mjidx

        if self._config["rl"]["mode"] == "test":
            if self._num_trials <= self._max_trials - 1:
                self._sampled_target_mjidx = self._ils100_cells_mjidxs[self._num_trials]

        # Reset the counter
        self._on_target_steps = 0

        # Update the number of remaining unread cells
        self._num_trials += 1

    def step(self, action):
        # Action at t
        # Normalise action from [-1, 1] to actuator control range
        # action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        # action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # # Set motor control
        # self._data.ctrl[:] = action

        action[0] = self.normalise(action[0], -1, 1, *self._saccade_stepwise_speed_bounds[:])
        action[1] = self.normalise(action[1], -1, 1, *self._saccade_stepwise_speed_bounds[:])
        # TODO the saccade speed is mainly a function of the target distance/amplitude. I can model this later.

        # # TODO try ocular motor acc as actions - Dr Yang's suggestion
        #
        # # Get the current fixation status. If is still on the process of saccading, apply the ocular motor noise
        # dist, geomid = self._get_focus(site_name="rangefinder-site")
        # # TODO a potential bug: what if the focus has moved outside but the drift has not been finished.
        # #  It will be dragged back to the target as a normal fixation.
        # if geomid == self._sampled_target_mjidx:
        #     # If already fixate on the target, apply the fixational eye movements, such as drift and jitters
        #     target_size = self._model.geom(geomid).size[0] * 2
        #     drift_noise_xpos_x = np.random.normal(0, np.abs(self._rho_drift * target_size))
        #     drift_noise_xpos_z = np.random.normal(0, np.abs(self._rho_drift * target_size))
        #     target_xpos = self._data.geom(geomid).xpos
        #     x, y, z = target_xpos[0], target_xpos[1], target_xpos[2]
        #     drift_xpos_x, drift_xpos_z = x + drift_noise_xpos_x, z + drift_noise_xpos_z
        #     drift_sample_location_vertical_radius = np.arctan(drift_xpos_z / y)
        #     drift_sample_location_horizontal_radius = np.arctan(-drift_xpos_x / y)
        #
        #     drift_samples_ctrl = np.array([drift_sample_location_vertical_radius, drift_sample_location_horizontal_radius])
        #
        #     self._data.ctrl[0:2] = np.clip(drift_samples_ctrl, *self._model.actuator_ctrlrange[0:2])
        # else:
        # Get the ocular motor noise
        ocular_motor_noise = np.random.normal(0, np.abs(self._rho_ocular_motor * action[0:2]))
        action[0:2] += ocular_motor_noise

        # Set motor control in MuJoCo simulation - moves saccades by saccades
        for idx, act_name in enumerate(["eye-x-motor", "eye-y-motor"]):
            act_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self._data.ctrl[act_mjidx] = np.clip(self._data.ctrl[act_mjidx] + action[idx],
                                                 *self._model.actuator_ctrlrange[idx])

        # Save the control for fatigue calculation
        self._ctrl_list.append(self._data.ctrl.copy())

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State at t+1 - transition function?
        # Eye-sight detection
        dist, geomid = self._get_focus(site_name="rangefinder-site")

        # Reset the scene first
        for mj_idx in self._ils100_cells_mjidxs:
            self._model.geom(mj_idx).rgba = self._DFLT_RGBA
        # Apply the transition function - update the scene regarding the actions
        if geomid == self._sampled_target_mjidx:
            self._on_target_steps += 1
            self._model.geom(self._sampled_target_mjidx).rgba = self._VISUALIZE_RGBA

        # Update the perturbation - firstly try the linear perturbation
        # TODO simple sinusoidal perturbation is to weak to convince others the necessity of the using RL,
        #  maybe later add some tough perturbations and let the agent learns to adapt to any perturbations
        # With the given perturbation period, the perturbation peak, apply a sinusoidal perturbation
        if self._mode == self._MODES[0]:
            amplitude_z = 0
            # amplitude_x = 0
        else:
            amplitude_z = self._perturbation_peak * np.sin(2 * self._steps / self._perturbation_period_z)
            # amplitude_x = self._perturbation_peak * np.sin(2 * self._steps / self._perturbation_period_x)
        self._data.qpos[self._perturbation_joint_z_mjidx] = amplitude_z
        # self._data.qpos[self._perturbation_joint_x_mjidx] = amplitude_x
        mujoco.mj_forward(self._model, self._data)

        # Update the transitions - get rewards and next state
        if self._on_target_steps >= self._dwell_steps:
            # Update the milestone bonus reward for finish reading a cell
            reward = 10
            # Get the next target
            self._sample_target()
        else:
            reward = 0.1 * (np.exp(
                -10 * self._angle_from_target(site_name="rangefinder-site", target_idx=self._sampled_target_mjidx)) - 1)

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._num_trials > self._max_trials:
            terminate = True

            if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
                print(f"The total time steps is: {self._steps}")
                weight = 1
                print(f"The fatigue cost is: {weight * sum(np.array(self._ctrl_list) ** 2)}, "
                      f"the total value is: {sum(weight * sum(np.array(self._ctrl_list) ** 2))}")

        return self._get_obs(), reward, terminate, {}

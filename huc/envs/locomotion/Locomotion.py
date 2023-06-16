import numpy as np
import mujoco
import os
import yaml

from gym import Env
from gym.spaces import Box, Dict


from huc.utils.rendering import Camera, Context


class StraightWalk(Env):

    def __init__(self):
        """
        Model walking straight forward to reach the target as an adaptation to efficiently obtain the visual information.
        The trajectory path will indicate where the destination is, the agents need to learn to utilize the minimal eye
        movements to reach the target as fast as possible.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "locomotion-v1.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._body_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "body-joint-y")
        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")

        self._straight_walk_path_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "straight-walk-path")

        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        path_geom_mjidx = np.where(self._model.geom_bodyid == self._straight_walk_path_body_mjidx)[0][0]
        self._destination_xpos = self._data.geom(path_geom_mjidx).xpos[1] + self._model.geom(path_geom_mjidx).size[1]
        self._init_xpos = None

        self._max_walking_speed_per_step = 0.1  # Maximum walking speed per step
        self._destination_proximity_threshold = 0.05  # The threshold to determine whether the agent reaches the target
        self._destination_timesteps_threshold = 10  # Wait for 10 steps to determine whether the agent reaches the target
        self._stop_timesteps_threshold = 10  # If the agent stays over 10 steps on the same position, then thinks it stops
        self._timesteps_on_destination = None  # Number of steps the agent stays on the target
        self._on_destination = None

        # Initialise RL related thresholds and counters
        self._steps = None
        self._num_trials = None  # Cells are already been read
        self._max_trials = 5  # Maximum number of cells to read - more trials in one episode will boost the convergence
        self.ep_len = 200  # Maximum number of steps in one episode

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._num_stateful_info = 2
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space: 1 dof eyeball rotations (up and down) and 1 dof body translation
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

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
        remaining_timesteps_on_destination_norm = (self._destination_timesteps_threshold - self._timesteps_on_destination) / self._destination_timesteps_threshold * 2 - 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_timesteps_on_destination_norm]
        )

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_destination = False
        self._timesteps_on_destination = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = 0
        self._data.qpos[self._eye_joint_y_mjidx] = 0
        self._init_xpos = np.random.uniform(0, self._destination_xpos)
        self._data.qpos[self._body_joint_y_mjidx] = self._init_xpos
        self._data.ctrl[1] = self._init_xpos

        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def step(self, action):
        # Action at t
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, 0, self._max_walking_speed_per_step)

        # Eyeball movement control - saccade to target positions
        self._data.ctrl[0] = action[0]
        # Locomotion control - cumulative forward
        self._data.ctrl[1] = np.clip(self._data.ctrl[1] + action[1], *self._model.actuator_ctrlrange[1])

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State at t+1 - Transit the state - the transition function is 1 for a deterministic environment
        # Estimate rewards
        qpos_body = self._data.qpos[self._body_joint_y_mjidx].copy()
        abs_distance = abs(qpos_body - self._destination_xpos)
        distance_penalty = -0.1 * abs(self.normalise(abs_distance, 0, self._destination_xpos, 0, 1))
        # distance_penalty = 0.1 * (np.exp(-10 * abs_distance) - 1)
        # distance_penalty = 0.1 * (np.exp(-0.5 * abs_distance) - 1)
        controls = self._data.ctrl[0].copy()
        # eye_movement_fatigue_penalty = - 0.1 * np.sum(controls**2)
        # reward = distance_penalty + eye_movement_fatigue_penalty
        reward = distance_penalty

        if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
            print(f"Step: {self._steps}, the qpos_body is: {qpos_body}, the destination_xpos is: {self._destination_xpos},"
                  f" The abs_distance is: {abs_distance}, distance_penalty: {distance_penalty}, "
                  f" The controls are: {controls}, "
                  # f"eye_movement_fatigue_penalty: {eye_movement_fatigue_penalty}"
                  f" \nThe current total reward is: {reward}, the forward moving speed is: {action[1]}, "
                  f" the forward moving control is: {self._data.ctrl[1]}")

        if abs_distance <= self._destination_proximity_threshold:
            self._timesteps_on_destination += 1
            if self._timesteps_on_destination >= self._destination_timesteps_threshold:
                self._on_destination = True
                reward = 100
        else:
            self._timesteps_on_destination = 0

        # if self._steps >= self.ep_len:
        #     if not self._on_destination:
        #         reward = -20

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._on_destination:
            terminate = True

        return self._get_obs(), reward, terminate, {}


class SignWalk(Env):

    def __init__(self):
        """
        Model walking straight forward to reach the target as an adaptation to efficiently obtain the visual information.
        The floating sign will indicate where the destination is, the agents need to learn to utilize the visual information
        by deploy eye movements to reach the target as fast as possible.
        """
        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "locomotion-v2.xml"))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)

        # Get the primitive idx in MuJoCo
        self._eye_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._body_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "body-joint-y")
        self._sign_joint_x_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "sign-joint-x")
        self._sign_joint_y_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "sign-joint-y")
        self._sign_joint_z_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "sign-joint-z")
        self._sign_joint_hinge_z_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "sign-joint-hinge-z")

        self._eye_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._straight_walk_path_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "straight-walk-path")
        # Get targets (geoms that belong to "smart-glass-pane-interline-spacing-100")
        path_geom_mjidx = np.where(self._model.geom_bodyid == self._straight_walk_path_body_mjidx)[0][0]
        self._movable_sign_body_mjidx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "movable-sign")
        self._movable_sign_mjidx = np.where(self._model.geom_bodyid == self._movable_sign_body_mjidx)[0][0]
        self._sign_qpos_x = None
        self._sign_qpos_y = None
        self._sign_qpos_z = None
        self._sign_qpos_hinge_z = None

        self._destination_xpos_y = None

        self._max_walking_speed_per_step = 0.1  # Maximum walking speed per step
        self._destination_proximity_threshold = 0.05  # The threshold to determine whether the agent reaches the target
        self._destination_timesteps_threshold = 10  # Wait for 10 steps to determine whether the agent reaches the target
        self._timesteps_on_destination = None  # Number of steps the agent stays on the target
        self._on_destination = None

        # Initialise RL related thresholds and counters
        self._steps = None
        self._num_trials = None  # Cells are already been read
        self._max_trials = 5  # Maximum number of cells to read - more trials in one episode will boost the convergence
        self.ep_len = 200  # Maximum number of steps in one episode

        # Define the observation space
        width, height = 80, 80
        self._num_stk_frm = 1
        self._num_stateful_info = 2
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(self._num_stk_frm, width, height)),
            "proprioception": Box(low=-1, high=1, shape=(self._num_stk_frm * self._model.nq + self._model.nu,)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info,)),
        })

        # Define the action space: 2 dof eyeball rotations (up and down, left and right) and 1 dof body translation
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Initialize the context and camera
        context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(context, self._model, self._data, camera_id="eye", resolution=[width, height],
                               maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

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
        remaining_timesteps_on_destination_norm = (self._destination_timesteps_threshold - self._timesteps_on_destination) / self._destination_timesteps_threshold * 2 - 1

        stateful_info = np.array(
            [remaining_ep_len_norm, remaining_timesteps_on_destination_norm]
        )

        if stateful_info.shape[0] != self._num_stateful_info:
            raise ValueError("The shape of stateful information is not correct!")

        return {"vision": vision, "proprioception": proprioception, "stateful information": stateful_info}

    def reset(self):

        # Reset MuJoCo sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the variables and counters
        self._steps = 0
        self._on_destination = False
        self._timesteps_on_destination = 0

        # Initialize eyeball rotation angles
        self._data.qpos[self._eye_joint_x_mjidx] = 0
        self._data.qpos[self._eye_joint_y_mjidx] = 0

        # Initialize a random sign position
        jnt_range_x_bottom, jnt_range_x_top = self._model.jnt_range[self._sign_joint_x_mjidx][0], self._model.jnt_range[self._sign_joint_x_mjidx][1]
        jnt_range_y = self._model.jnt_range[self._sign_joint_y_mjidx]
        jnt_range_z = self._model.jnt_range[self._sign_joint_z_mjidx]
        sign_qpos_x = np.random.choice([jnt_range_x_bottom, jnt_range_x_top])
        if sign_qpos_x <= jnt_range_x_bottom:
            self._data.qpos[self._sign_joint_hinge_z_mjidx] = 1
        else:
            self._data.qpos[self._sign_joint_hinge_z_mjidx] = -1
        sign_qpos_y = np.random.uniform(*jnt_range_y)
        sign_qpos_z = np.random.uniform(*jnt_range_z)
        self._data.qpos[self._sign_joint_x_mjidx] = sign_qpos_x
        self._data.qpos[self._sign_joint_y_mjidx] = sign_qpos_y
        self._data.qpos[self._sign_joint_z_mjidx] = sign_qpos_z

        self._data.ctrl[3] = sign_qpos_x
        self._data.ctrl[4] = sign_qpos_y
        self._data.ctrl[5] = sign_qpos_z
        self._data.ctrl[6] = self._data.qpos[self._sign_joint_hinge_z_mjidx]

        self._destination_xpos_y = sign_qpos_y

        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def step(self, action):
        # Action at t
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])
        action[2] = self.normalise(action[2], -1, 1, 0, self._max_walking_speed_per_step)

        # Eyeball movement control - saccade to target positions
        self._data.ctrl[0] = action[0]
        self._data.ctrl[1] = action[1]
        # Locomotion control - cumulative forward
        self._data.ctrl[2] = np.clip(self._data.ctrl[2] + action[2], *self._model.actuator_ctrlrange[2])

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # State at t+1 - Transit the state - the transition function is 1 for a deterministic environment
        # Estimate rewards
        qpos_body = self._data.qpos[self._body_joint_y_mjidx].copy()
        abs_distance = abs(qpos_body - self._destination_xpos_y)
        distance_penalty = -0.1 * abs(self.normalise(abs_distance, 0, self._destination_xpos_y, 0, 1))
        # distance_penalty = 0.1 * (np.exp(-10 * abs_distance) - 1)
        # distance_penalty = 0.1 * (np.exp(-0.5 * abs_distance) - 1)
        controls = self._data.ctrl[0:2].copy()
        # eye_movement_fatigue_penalty = - 0.1 * np.sum(controls**2)
        # reward = distance_penalty + eye_movement_fatigue_penalty
        reward = distance_penalty

        # if self._config["rl"]["mode"] == "debug" or self._config["rl"]["mode"] == "test":
        #     print(f"Step: {self._steps}, the qpos_body is: {qpos_body}, the destination_xpos is: {self._destination_xpos_y},"
        #           f" The abs_distance is: {abs_distance}, distance_penalty: {distance_penalty}, "
        #           f" The controls are: {controls}, "
        #           # f"eye_movement_fatigue_penalty: {eye_movement_fatigue_penalty}"
        #           f" \nThe current total reward is: {reward}, the forward moving speed is: {action[1]}, "
        #           f" the forward moving control is: {self._data.ctrl[1]}")

        if abs_distance <= self._destination_proximity_threshold:
            self._timesteps_on_destination += 1
            if self._timesteps_on_destination >= self._destination_timesteps_threshold:
                self._on_destination = True
                reward = 100
        else:
            self._timesteps_on_destination = 0

        # Get termination condition
        terminate = False
        if self._steps >= self.ep_len or self._on_destination:
            terminate = True

        return self._get_obs(), reward, terminate, {}

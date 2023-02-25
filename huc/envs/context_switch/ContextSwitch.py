import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

from huc.utils.rendering import Camera, Context


class ContextSwitch(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "context-switch.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._target_switch_interval = 2 * self._action_sample_freq
        self._steps = 0
        self._max_trials = 5
        self._trial_idx = 0

        # Get targets (geoms that belong to "smart-glass-pane")
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._targets = [self._model.geom(idx) for idx in self._target_idxs]
        self._target_idx = None
        self._target = None

        # Get the background (geoms that belong to "background-pane")
        bp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == bp_body)[0]
        self._background_idx0 = self._background_idxs[0]

        # The smartglass pane: Determine the idx of grids which needs to be traversed in some sequence
        self._sequence_target_idxs = []
        # The reading result buffer - should has the same length
        self._sequence_results_idxs = []
        self._default_idx = -1
        self._num_targets = 0
        self._ep_len = 400

        # The background grids:
        self._background_on = False
        self._steps_background_on = 0
        self._steps_background_off = 0
        self._background_on_interval = 1 * self._action_sample_freq
        self._background_off_interval = 2 * self._action_sample_freq

        # Define observation space
        self._width = 80
        self._height = 80
        self.observation_space = Box(low=0, high=255, shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
        self._rangefinder_cutoff = 0.1

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

    def _get_obs(self):

        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess
        rgb = np.transpose(rgb, [2, 0, 1])
        return self.normalise(rgb, 0, 255, -1, 1)

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    # TODO there are some changable variables in this protected method, change it later
    def _reset_scene(self):

        # Leave some randomness by using the number smaller than the total registered grid number 16
        self._num_targets = len(self._target_idxs)

        # First reset the scene.
        for idx in self._target_idxs:
            self._model.geom(idx).rgba[0:4] = [0.5, 0.5, 0.5, 0.85]
            self._model.geom(idx).size[0:3] = [0.0025, 0.0001, 0.0025]

        # Refresh the scene with randomly generated targets.
        # # TODO the background task: generate 1 by 1 randomly - this is for training and testing
        # targets_idxs = np.random.choice(range(self._target_idxs[0], self._target_idxs[-1] + 1), size=self._num_targets,
        #                                 replace=False)
        # targets_idxs.sort()
        # targets_idxs_list = targets_idxs.tolist()
        # self._sequence_target_idxs = targets_idxs_list

        # TODO the sequential reading task: generate 1 by 1 sequentially - this is only for testing.
        self._sequence_target_idxs = [2, 3, 4]
        self._num_targets = len(self._sequence_target_idxs)
        # ------------------------------------------------------------------------------------------

        self._sequence_results_idxs = [self._default_idx for _ in self._sequence_target_idxs]

        self._switch_target(idx=self._sequence_target_idxs[0])

    def _switch_target(self, idx):

        self._model.geom(idx).rgba[0:4] = [0.8, 0.8, 0, 1]
        self._model.geom(idx).size[0:3] = [0.0045, 0.0001, 0.0045]

        self._target_idx = idx

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):

        # Update the steps of background on
        if self._background_on:
            self._steps_background_on += 1
            # Close the background change
            if self._steps_background_on >= self._background_on_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = [0.5, 0.5, 0.5, 0.85]
                self._model.geom(self._background_idx0).size[0:3] = [0.0035, 0.0001, 0.0035]
                self._steps_background_off = 0
                self._background_on = False

        # Start deterministic background change
        if self._background_on is False:
            self._steps_background_off += 1
            # Start the background change
            if self._steps_background_off >= self._background_off_interval:
                self._model.geom(self._background_idx0).rgba[0:4] = [0.8, 0, 0, 1]
                self._model.geom(self._background_idx0).size[0:3] = [0.0065, 0.0001, 0.0065]
                self._steps_background_on = 0
                self._background_on = True

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Update fixate point based on rangefinder
        x = self._data.sensor("rangefinder").data
        x = x if x >= 0 else self._rangefinder_cutoff
        self._model.geom("fixate-point").pos[2] = -x

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

        # Check for collisions, estimate reward
        reward = 0
        if x != self._rangefinder_cutoff and len(self._data.contact.geom2) > 0:
            geom2 = self._data.contact.geom2[0]
            # If the geom2 is in the target idxs array, then the rewards are applied, the environment changes a little bit
            if self._background_on:
                if geom2 == self._background_idx0:
                    reward = 1
                    # Update the environment
                    acc = 0.8 / self._background_on_interval
                    self._model.geom(geom2).rgba[0:3] = [x + y for x, y in zip(self._model.geom(geom2).rgba[0:3], [0, 0, acc])]
                    # Do a forward so everything will be set
                    mujoco.mj_forward(self._model, self._data)
            else:
                if geom2 == self._target_idx:
                    reward = 1
                    # Update the environment
                    acc = 0.8 / self._target_switch_interval
                    self._model.geom(geom2).rgba[0:3] = [x + y for x, y in zip(self._model.geom(geom2).rgba[0:3], [0, 0, acc])]
                    # Do a forward so everything will be set
                    mujoco.mj_forward(self._model, self._data)

        # if self._steps >= self._ep_len or (self._sequence_results_idxs.count(self._default_idx) <= 0):
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False
            for idx in self._sequence_target_idxs:

                # print('The sequence results are: {}, the sequence targets are: {} \nthe grid-4 b is: {} \nthe action is: {}'.
                #       format(self._sequence_results_idxs,
                #              self._sequence_target_idxs,
                #              self._model.geom(4).rgba[2],
                #              action,
                #              ))    # TODO debug delete later

                # Check whether the grid has been fixated for enough time
                if (self._model.geom(idx).rgba[2] >= 0.8) and (idx not in self._sequence_results_idxs):
                    # Update the results
                    for i in range(self._num_targets):
                        if self._sequence_results_idxs[i] == self._default_idx:
                            self._sequence_results_idxs[i] = idx
                            break
                    # Update the scene - a sharp update
                    self._model.geom(idx).size[0:3] = [0.0025, 0.00001, 0.0025]
                    # Switch a new target grid
                    for i in range(self._num_targets):
                        if self._sequence_results_idxs[i] != self._sequence_target_idxs[i]:
                            self._switch_target(idx=self._sequence_target_idxs[i])
                            break

        # Renew the smartglass grids
        if self._sequence_results_idxs.count(self._default_idx) <= 0:
            self._reset_scene()

        return self._get_obs(), reward, terminate, {}
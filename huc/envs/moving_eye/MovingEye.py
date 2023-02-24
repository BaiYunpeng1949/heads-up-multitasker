import math

import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

from huc.utils.rendering import Camera, Context


class MovingEye(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory, "moving-eye.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._target_switch_interval = 1 * self._action_sample_freq
        self._steps = 0
        self._max_trials = 5
        self._trial_idx = 0

        # Get targets (geoms that belong to "smart-glass-pane")
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._targets = [self._model.geom(idx) for idx in self._target_idxs]
        self._target_idx = None
        self._target = None

        # Added parts
        # Determine the idx of grids which needs to be traversed in some sequence - this should be changed according to different layouts
        self._sequence_target_idxs = []
        # The reading result buffer - should has the same length
        self._sequence_results_idxs = []
        self._default_idx = -1
        self._num_targets = 0
        self._collide_threshold = 0
        self._x_max, self._x_min, self._z_max, self._z_min = 0, 0, 0, 0
        self._ep_len = 200  # TODO check if ep_len is greater than num_ trials.

        # Define observation space
        # TODO: add sequential information + higher resolution + move closer to the panes.
        #  Higher resolutions are needed for detecting smaller changes.
        self._width = 80
        self._height = 80
        self.observation_space = Box(low=0, high=255, shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))
        # Note: use the relative movement might be more close to the visual behaviors, such as saccades

        # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
        self._rangefinder_cutoff = 0.1

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)  # TODO maybe the central vision's resolution is also one aspect of model integration.
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)

    def _get_obs(self):
        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess
        rgb = np.transpose(rgb, [2, 0, 1])
        # TODO Try the stacking frame technique - no need now since the agent can see the whole girds
        # TODO If we are using the partial pixels as observations, the agent does not know the whole scene in advance, it is categorized as a POMDP?
        return self.normalise(rgb, 0, 255, -1, 1)

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0
        # self._trial_idx = 0

        # Choose one target at random
        # self._switch_target()

        # Initialize the number of the target grids.
        # TODO next time when the target number changes, the rewards need to be normalized.
        # self._num_targets = np.random.choice(range(1, len(self._target_idxs)+1))
        self._num_targets = 4

        # First reset the scene.
        for idx in self._target_idxs:
            self._model.geom(idx).rgba[0:4] = [0.5, 0.5, 0.5, 0.85]
            self._model.geom(idx).size[0:3] = [0.0025, 0.0001, 0.0025]

        # Refresh the scene with randomly generated targets.
        targets_idxs = np.random.choice(range(self._target_idxs[0], self._target_idxs[-1]+1), size=self._num_targets, replace=False)
        targets_idxs.sort()
        targets_idxs_list = targets_idxs.tolist()
        # self._sequence_target_idxs = targets_idxs_list    # TODO the random trials.
        self._sequence_target_idxs = [2, 5, 14, 17]         # TODO the deterministic trial.
        self._sequence_results_idxs = [self._default_idx for _ in self._sequence_target_idxs]
        # Color and resize all the selected grids
        for idx in self._sequence_target_idxs:
            self._model.geom(idx).rgba[0:4] = [0.8, 0.8, 0, 1]
            self._model.geom(idx).size[0:3] = [0.0045, 0.0001, 0.0045]
            # TODO The hints must be salient, especially when the resolution is low.

        # TODO preserve these non-informative gris to provide some potential layout information.
        # for idx in self._target_idxs:
        #     if idx not in self._sequence_target_idxs:
        #         self._model.geom(idx).rgba[3] = 0

        # Initialize the fixate-target cell distance threshold
        self._collide_threshold = min(self._model.geom("grid-0").size[0], self._model.geom("grid-0").size[2]) / 2

        # Set the target grid
        self._switch_target()

        return self._get_obs()

    def _switch_target(self):

        # # Sample a random target
        # idx = np.random.choice(len(self._target_idxs))
        # self._target_idx = self._target_idxs[idx]
        # self._target = self._targets[idx]
        #
        # # Set position of target (the yellow border)
        # self._model.body("target").pos = self._model.body("smart-glass-pane").pos + self._target.pos
        #
        # # Do a forward so everything will be set
        # mujoco.mj_forward(self._model, self._data)

        seen = set()
        for idx in self._sequence_target_idxs:
            if idx not in self._sequence_results_idxs and idx not in seen:
                self._target_idx = idx
                break
            seen.add(idx)

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def normalise(self, x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # if self._target_idx == 2:
        #     action = [0.28, 0.31]   # TODO debug line
        # elif self._target_idx == 5:
        #     action = [0.31, -0.27]
        # elif self._target_idx == 14:
        #     action = [-0.28, 0.31]
        # elif self._target_idx == 17:
        #     action = [-0.3, -0.27]

        # Set motor control
        self._data.ctrl[:] = action
        # TODO there are some inconsistency with the actions from the simulator demonstrations.
        #  Or we can set a very large plane in the very behind.

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Update fixate point based on rangefinder
        # x = self._data.sensor("rangefinder").data
        # x = x if x >= 0 else self._rangefinder_cutoff
        # self._model.geom("fixate-point").pos[2] = -x

        # Do a forward so everything will be set
        # mujoco.mj_forward(self._model, self._data)

        # Check for collisions, estimate reward
        dwell_indicator = 0

        # # TODO change this part from using the contact collisions to another method.
        # if x != self._rangefinder_cutoff and len(self._data.contact.geom2) > 0:
        #     geom2 = self._data.contact.geom2[0]
            # if self._target_idx == geom2:
            #   reward = 1
            # If the geom2 is in the target idxs array, then the rewards are applied, the environment changes a little bit
            # if geom2 in self._sequence_target_idxs and geom2 not in self._sequence_results_idxs:
            #     reward = 0.1
            #     # Update the environment
            #     acc = 0.8 / self._target_switch_interval
            #     self._model.geom(geom2).rgba[0:3] = [x + y for x, y in zip(self._model.geom(geom2).rgba[0:3], [0, 0, acc])]
            #     # Do a forward so everything will be set        # TODO check if this is redundant.
            #     mujoco.mj_forward(self._model, self._data)

        # The distance detection    TODO the MuJoCo ray collision might serve the same functionalities
        # Define the ray vector
        p = self._data.site("rangefinder-site").xpos
        direction_ = self._data.geom("fixate-point").xpos  # TODO this was not updated correctly
        fixate_ray_len = abs(self._model.geom("fixate-point").pos[2])
        projection_xy = abs(fixate_ray_len * math.cos(action[0]))
        x = - projection_xy * math.sin(action[1])
        y = projection_xy * math.cos(action[1])
        z = fixate_ray_len * math.sin(action[0])
        direction = np.array([x, y, z])
        # Define the x-z plane equation     TODO normalize this later, the plane can be dynamical
        a, b, c = 0, 1, 0   # Normalize the vector of the x-z plane
        dist_plane = - self._data.body("smart-glass-pane").xpos[1]               # Distance from origin to plane
        # Calculate the intersection point
        t = - (a * p[0] + b * p[1] + c * p[2] + dist_plane) / (a * direction[0] + b * direction[1] + c * direction[2])
        intersection_xpos = p + t * direction

        aa = self._target_idx
        bb = self._sequence_target_idxs
        rgb = self._model.geom(self._target_idx).rgba[0:3]
        target_grid_xpos = self._data.geom(self._target_idx).xpos
        target_grid_size = self._model.geom(self._target_idx).size
        x_min = target_grid_xpos[0] - target_grid_size[0] / 2
        x_max = target_grid_xpos[0] + target_grid_size[0] / 2
        z_min = target_grid_xpos[2] - target_grid_size[2] / 2
        z_max = target_grid_xpos[2] + target_grid_size[2] / 2

        # Check the distance between the target grid and the intersection projection xpos - TODO used for the reward shaping
        dist = math.sqrt(sum([(self._data.geom(self._target_idx).xpos[i]-intersection_xpos[i])**2 for i in range(3)]))

        if x_min <= intersection_xpos[0] <= x_max and z_min <= intersection_xpos[2] <= z_max:
        # if dist < self._collide_threshold:
            # reward = 0.1
            # Update the environment
            acc = 0.8 / self._target_switch_interval
            self._model.geom(self._target_idx).rgba[0:3] = [x + y for x, y in zip(self._model.geom(self._target_idx).rgba[0:3], [0, 0, acc])]
            print('The preliminary, the target_idx is: {}, the b is: {}, the intersection_xpos is: {}, the dist is: {}, actions are: {}, the steps are: {}'.format(self._target_idx, self._model.geom(self._target_idx).rgba[2], intersection_xpos, dist, action, self._steps))  # TODO debug
            mujoco.mj_forward(self._model, self._data)

        # Check whether we should terminate episode (if we have gone through enough trials)
        # if self._trial_idx >= self._max_trials:
        #   terminate = True
        # else:
        #   terminate = False
        #   # Check whether we should switch target
        #   if self._steps >= self._target_switch_interval:
        #     self._switch_target()
        #     self._trial_idx += 1
        #     self._steps = 0

        # The final status
        if self._steps >= self._ep_len or (self._sequence_results_idxs.count(self._default_idx) <= 0):
            terminate = True
            if self._sequence_results_idxs.count(self._default_idx) <= 0:
                print('Terminate early for the task completion')
        else:
            terminate = False
            # Update
            if self._model.geom(self._target_idx).rgba[2] >= 0.8:
                # Update the result buffer
                for i in range(len(self._sequence_results_idxs)):
                    if self._sequence_results_idxs[i] == self._default_idx:
                        self._sequence_results_idxs[i] = self._target_idx
                        break
                # Update the scene - a sharp size change
                self._model.geom(self._target_idx).size[0:3] = [0.0025, 0.00001, 0.0025]
                # Update the target idx
                self._switch_target()
                # Do a forward so everything will be set    # TODO check if it is redundant.
                mujoco.mj_forward(self._model, self._data)
                # Set the dwell time indicator
                dwell_indicator = 1
                print('The middle goal was achieved. The new target is: {}, the sequence results are: {}'.format(self._target_idx, self._sequence_results_idxs))

            # for idx in self._sequence_target_idxs:
            #     # Check whether the grid has been fixated for enough time
            #     if (self._model.geom(idx).rgba[2] >= 0.8) and (idx not in self._sequence_results_idxs):
            #         # Update the results
            #         for i in range(len(self._sequence_results_idxs)):
            #             if self._sequence_results_idxs[i] == self._default_idx:
            #                 self._sequence_results_idxs[i] = idx
            #                 break
            #         # Update the reward - TODO try the staged rewards since we are playing with the sequential task
            #         # TODO Maybe add the sequential requirements here - the reward function should be normalized or something to generalize between different tasks with randomized settnigs.
            #         reward = (self._num_targets // 2) * (len(self._sequence_results_idxs) - self._sequence_results_idxs.count(self._default_idx))
            #         # Update the scene - a sharp update
            #         self._model.geom(idx).size[0:3] = [0.0025, 0.00001, 0.0025]
            #         # Do a forward so everything will be set    # TODO check if it is redundant.
            #         mujoco.mj_forward(self._model, self._data)

        # TODO get the staged/gradient reward function.
        reward = self._reward_function(x=dist, f=dwell_indicator)

        return self._get_obs(), reward, terminate, {}

    def _reward_function(self, x, f, a=50, b=10, c=10):
        offset = self._collide_threshold
        shaping_reward = (math.exp(-a * (x - offset)) - 1) / b
        actual_reward = c * f
        reward = shaping_reward + actual_reward
        return reward

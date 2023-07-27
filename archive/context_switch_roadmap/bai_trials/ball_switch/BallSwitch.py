import os
import sys
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

import mujoco

import gym
from gym import Env
from gym import utils
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete


class BallSwitch(Env):
    # Reference: https://github.com/BaiYunpeng1949/user-in-the-box/blob/main/uitb/simulator.py
    def __init__(self, model_xml_path):
        ###################################### Initialization: model, attributes, and features. ######################################
        # Set up the configurations.
        self._run_parameters = {
            "rendering_context": None,
            "action_sample_freq": 20,
            "frame_skip": 4,
            "dt": 0  # The short form of delta_time, could be defined as: model.opt.timestep*frame_skip
        }

        # Load the xml model.
        self._model = mujoco.MjModel.from_xml_path(model_xml_path)
        # Initialize MjData.
        self._data = mujoco.MjData(self._model)

        # Add run_parameters.
        self._frame_skip = self._run_parameters["frame_skip"]
        self._run_parameters["dt"] = self._model.opt.timestep * self._frame_skip
        # self._run_parameters["rendering_context"] = Context(model=self._model, max_resolution=[1280, 960]) # TODO uncomment this when the rendering is setup.

        # Set the action space.
        # The concrete action_space reference: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/simulator.py#L306:7
        # , while I just have one actuator. To limit it whithin [-1, 1]. TODO standarlize it.
        num_dof = 1  # the number of degree of freedom, which is up to me. Could be for x, y, and z separately. idx=0 for x offset; idx=1 for y offset, and idx=2 for z offset.
        axis_offset_limits = np.ones((num_dof, 2)) * np.array([-1.0, 1.0])  # '2' was for the lower and upper bound separately.
        axis_offset_limits = np.array([[0, 0],
                                       [-0.5, 0.5],
                                       [0, 0]])  # TODO Comment this declaration for formal usage, this is just for the simplest y-moving scenario, where the action_space.sample() will output [0, y_offset, 0]
        self.action_space = Box(low=np.float32(axis_offset_limits[:, 0]), high=np.float32(axis_offset_limits[:, 1]))

        # Set the observation space. Define the observation space which is a continuous space as well.
        # Ref: https://mujoco.readthedocs.io/en/latest/APIreference.html#mjmodel, https://mujoco.readthedocs.io/en/latest/APIreference.html#mjdata
        # I am choosing the dimension 3 because it is just a task with a ball moving aroung without considering the rotations.
        # According to https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/simulator.py#L312
        #              ---- https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/perception/base.py#L184 # It is just unpacking keys.
        #              ---------- https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/perception/base.py#L107
        # , the observation_space should have the same dimension and form as get_observation.
        # Return limits for the observations. These limits aren't currently used for anything (AFAIK, not in gym or stable-baselines3;
        # , only to initialise the observation space required by gym.Env), so let's just use a default of
        # , -inf to inf. Overwrite this method to use different ranges. If they are arrays, then the shape must be consistent. The shape should be (3,).
        self.observation_space = Dict({
            "joint_dot_positions": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "joint_dot_velocities": Box(low=-np.inf, high=np.inf, shape=(3,))
        })

        ###################################### Initialization: scripts for logic control, such as counters and flags. ######################################
        # TODO: use structure or some data storage to easily synchronize with the reset method.
        # Zero the speed and acc.
        self._data.qvel[0:3] = np.array([0, 0, 0])
        self._data.qacc[0:3] = np.array([0, 0, 0])

        # The reference: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/tasks/pointing/Pointing.py#L24
        # Use early termination if target is not hit in time
        self._max_steps_without_touch_plane = int(4 * self._run_parameters["action_sample_freq"])
        self._steps_since_last_touch_plane1 = 0
        self._steps_since_last_touch_plane2 = 0

        # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
        self._max_plane_trials = 100
        # Plane1.
        self._trial_plane1_idx = 0
        # Plane2.
        self._trial_plane2_idx = 0

        # Set the log info dictionary. Initialize with a replicable initializer.
        self._info_init = {
            "num_plane1_touched": 0,
            "num_plane2_touched": 0,
            "num_switch": 0,
            "last_touched_plane": 'nothing',  # The buffer that stores the last touched plane.
            "num_osci_pl1": 0,
            "num_osci_pl2": 0,
            "is_switched": False,
            # The switching flag, should encourage the dot to touch two planes reciprocate alternately.
            "target_plane1_touch": False,  # The final/general (big) identify of the touch gesture.
            "target_plane2_touch": False,
            "inside_target_plane1": False,
            # The step (small) motion status. It will accumulates up and be evaluated whether or not the plane has been touched.
            "inside_target_plane2": False,
            "finished": False,
            "termination": False
        }
        self.info = self._info_init

        # Set the steps that is taken in the environment - keep the track of simulated steps.
        self._steps = 0

        # The dot's dwelling based touching - the dot needs to have connection with the plane.
        self._steps_inside_plane1 = 0
        self._steps_inside_plane2 = 0
        self._dwell_threshold = int(0.15 * self._run_parameters["action_sample_freq"])  # This part could simulate the behavior of fixation.

        # Define a space where the target dot will move. Only in the x - y plane. TODO modify this to my plane touch scenario later.
        # self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([0.55, -0.1, 0])
        # self._target_origin = self._data.xpos[1] # The id=0 is the body 'world' stays at [0, 0, 0], id=1 is the body 'dot' initialized at [0, 1, 1].

        # The targets were planes, so get the planes' positions.
        self._target_plane1_origin = self._model.geom('plane1').pos
        self._target_plane2_origin = self._model.geom('plane2').pos

        self._target_plane1_position = self._target_plane1_origin.copy()
        self._target_plane2_position = self._target_plane2_origin.copy()

        # self._target_limits_x = np.array([-5, 5])
        # self._target_limits_y = np.array([-5, 5])
        # self._target_limits_z = np.array([-5, 5])

        self._dist_plane1 = 0
        self._dist_plane2 = 0
        self._dist_pl1_previous = 100
        self._dist_pl2_previous = 100

        # Define the threshold of the dot is touching the plane. TODO define later.
        self._touch_threshold = 0.05  # The touching distance threshold of the dot's centroid to the plane's section line with the xy horizontal plane where the ball resides. Given the size of the ball was 0.1, I made the threhold 0.025.

        # Update the dot's geometry - currently no need of doint that. TODO later.

        # Update the dot's location - Pointing.py line 63. Notice: this is not necessary here.
        # self._model.body("dot").pos = self._target_origin

        ###################################### Initialization: dynamical state information, rendering, camera settings. ######################################
        # Update the camera pos - currently no need of doing that. TODO later.
        # Initialize the viewer.TODO try Camera(self._run_parameters["rendering_context"], self._model, self._data, camera_id='for_testing',dt=self._run_parameters["dt"])
        # self._viewer = Camera(context=self._run_parameters["rendering_context"],
        #                      model=self._model,
        #                      data=self._data,
        #                      camera_id='for_testing',
        #                      dt=self._run_parameters["dt"])

        # Collect the episode statistics.
        self._epidose_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    def step(self, action):
        # Advance the simulation
        # mujoco.mj_step(self._model, self._data, nstep=self._run_parameters["frame_skip"])
        mujoco.mj_step(self._model, self._data)

        # Update environment.
        reward, done, info = self._update(action=action)

        # Update the simulated steps.
        self._steps += 1

        # Get observation.
        obs = self._get_obs()

        return obs, reward, done, info

    def reset(self):
        # Reset sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset the model.
        # Modify the referenc - pointing: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/tasks/pointing/Pointing.py#L140
        # Reset the counters.
        # Plane1.
        self._trial_plane1_idx = 0
        self._target_plane1_touch = 0
        # Plane2.
        self._trial_plane2_idx = 0
        self._target_plane2_touch = 0

        # Set the log info dictionary.
        self.info = self._info_init

        # Set the steps that is taken in the environment - keep the track of simulated steps.
        self._steps = 0

        # The dot's dwelling based touching - the dot needs to have connection with the plane.
        self._steps_inside_plane1 = 0
        self._steps_inside_plane2 = 0

        # Reset distances.
        self._dist_plane1 = 0
        self._dist_plane2 = 0
        self._dist_pl1_previous = 100  # Randomly assign a very big number to prevent in the first trial there are many penalty.
        self._dist_pl2_previous = 100

        # TODO - imitate the spawn target stuff. Update the dot's moving.
        # Reset the initial velocity and acc to 0s. Or the ball can easily bounce out (it was due to the horizontal planes). These things were not identified in the xml file, then need to reset here.
        self._data.qvel[0:3] = np.array([0, 0, 0])
        self._data.qacc[0:3] = np.array([0, 0, 0])

        # Do a forward so everything will be set.
        mujoco.mj_forward(self._model, self._data)

        ob = self._get_obs()
        return ob

    def _update(self, action):
        """
          Args:
            action: Action values between [-1, 1], with the form of [x, y, z] indicating 2 directions' offsets.
        """
        # TODO modify/specify according to my needs from: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/tasks/pointing/Pointing.py#L76
        # ------------------ https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/bm_models/base.py#L133
        # , where the action value was ranged from -1 to 1.

        # TODO use data.ctrl to reset values - should I set the dot as an actuator? - Not yet, it could be controlled directly by data.qpos. Check Ak's email.
        # Ref: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/bm_models/base.py#L133
        # data.ctrl[self._motor_actuators] = np.clip(self._motor_smooth_avg + action[:self._nm], 0, 1)
        # This was only need when we are playing with actuators.

        self._data.qpos[0:3] += action  # Update the current dot's position.

        # Set some defaults.
        done = False

        # Get the dot's current position. TODO integrate with the action.
        # current_dot_pos = self._data.xpos[1] # Cartesian position of body frame.
        current_dot_pos = self._data.qpos[0:3]  # Just the position, I am using it instead of the Cartesian position to be consistent with the prior settings.

        # Start to calculate the tasks: plane-touching requirements.
        # Distance to the target plane. TODO modify later - calculate the distance between the centroid and the plane section line. Since we are in the simplest mode, I am just calculating the offset on the y-axis.
        self._dist_plane1 = np.linalg.norm(self._target_plane1_position[1] - current_dot_pos[1])  # Distance calculation with the abs value. TODO remove the [1]s afterwards. Now is the simplest mode.
        self._dist_plane2 = np.linalg.norm(self._target_plane2_position[1] - current_dot_pos[1])
        # print("dist_plane1: {}, dist_plane2: {}".format(dist_plane1, dist_plane2)) # TODO debug, can be deleted later.

        # Check if the dot touches the planes1.
        # Check the distance to the plane1.
        if self._dist_plane1 <= self._touch_threshold:
            self._steps_inside_plane1 += 1
            self.info["inside_target_plane1"] = True
            self.info["inside_target_plane2"] = False
        else:  # Consecutive steps where the object is inside the plane is needed.
            self._steps_inside_plane1 = 0
            self.info["inside_target_plane1"] = False
        # Check the distance to the plane2.
        if self._dist_plane2 <= self._touch_threshold:
            self._steps_inside_plane2 += 1
            self.info["inside_target_plane1"] = False
            self.info["inside_target_plane2"] = True
        else:
            self._steps_inside_plane2 = 0
            self.info["inside_target_plane2"] = False

        # If the dot touches the plane, then check its dwelling time. For my scenario, we don't need the dwelling time. This could be an alternatieve to mimic visual fixation behavior.
        if self.info["inside_target_plane1"] and self._steps_inside_plane1 >= self._dwell_threshold:  # The _dwell_threshold was identified by the steps.
            # Update the counts
            if self.info["last_touched_plane"] == 'plane2':
                self.info["is_switched"] = True
                self.info["num_switch"] += 1
            elif self.info["last_touched_plane"] == 'nothing' or self.info["last_touched_plane"] == 'plane1':
                self.info["is_switched"] = False

            self.info["target_plane1_touch"] = True
            self.info["target_plane2_touch"] = False
            self.info["last_touched_plane"] = 'plane1'

            self.info["num_plane1_touched"] += 1
            self._trial_plane1_idx += 1

            # Refresh / update the steps.
            self._steps_since_last_touch_plane1 = 0
            self._steps_inside_plane1 = 0

        else:  # Not enough dwelling time.
            self.info["target_plane1_touch"] = False
            self.info["num_osci_pl1"] += 1
            # Check if the time limit has been reached.
            self._steps_since_last_touch_plane1 += 1
            if self._steps_since_last_touch_plane1 >= self._max_steps_without_touch_plane:
                self._steps_since_last_touch_plane1 = 0
                self._trial_plane1_idx += 1

        if self.info["inside_target_plane2"] and self._steps_inside_plane2 >= self._dwell_threshold:
            # Update the counts
            if self.info["last_touched_plane"] == 'plane1':
                self.info["is_switched"] = True
                self.info["num_switch"] += 1
            elif self.info["last_touched_plane"] == 'nothing' or self.info["last_touched_plane"] == 'plane2':
                self.info["is_switched"] = False

            self.info["target_plane1_touch"] = False
            self.info["target_plane2_touch"] = True
            self.info["last_touched_plane"] = 'plane2'

            self.info["num_plane2_touched"] += 1
            self._trial_plane2_idx += 1
            self._steps_since_last_touch_plane2 = 0
            self._steps_inside_plane2 = 0
        else:  # Not enough dwelling time.
            self.info["target_plane2_touch"] = False
            self.info["num_osci_pl2"] += 1
            # Check if the time limit has been reached.
            self._steps_since_last_touch_plane2 += 1
            if self._steps_since_last_touch_plane2 >= self._max_steps_without_touch_plane:
                self._steps_since_last_touch_plane2 = 0
                self._trial_plane2_idx += 1

        # Check if max number trials reached
        if self._trial_plane1_idx >= self._max_plane_trials or self._trial_plane2_idx >= self._max_plane_trials:
            done = True
            self.info["termination"] = "max_plane1_trials_reached"

        # Calculate the reward - TODO finalize this later.
        reward = self._reward_function()

        # Update iterative parameters.
        self._dist_pl1_previous = self._dist_plane1
        self._dist_pl2_previous = self._dist_plane2

        # Do a forward calculation.
        mujoco.mj_forward(self._model,
                          self._data)  # mj_forward doesn't change values. It is the mj_step that is changing values.

        return reward, done, self.info

    def _reward_function(self):
        # Cusomize + Reference: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/tasks/pointing/reward_functions.py#L76
        # Maybe need to add more constraints later. TODO add the switching flag later.

        # Initialize the reward, if it is doing nothing, then gets punished.
        # Penalties are given if the dot / ball is just lingering around / moving further away to both planes.
        # But I encourage the ball to move closer to 1 or 2. It is like an directional instruction.
        if self._dist_plane1 < self._dist_pl1_previous or self._dist_plane2 < self._dist_pl2_previous:
            reward = 1
        else:
            reward = -5

        # The plane 1 hard constraints - we still encourage the dot/ball moves around the planes, so all the rewards are positive.
        if self.info["target_plane1_touch"]:
            if self.info["is_switched"]:
                reward = 500
            elif self.info["is_switched"] == False:
                reward = -1
        elif self.info["target_plane1_touch"] == False and self.info["inside_target_plane1"]:  # Just jitters on or shallowly touches the plane.
            reward = -0.5
        # The plane 2 hard constraints.
        if self.info["target_plane2_touch"]:
            if self.info["is_switched"]:
                reward = 500
            elif self.info["is_switched"] == False:
                reward = -1
        elif self.info["target_plane2_touch"] == False and self.info["inside_target_plane2"]:
            reward = -0.5

        return reward

    def _get_obs(self):
        # TODO specify according to the perception observation update https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/perception/vision/fixed_eye/FixedEye.py#L97
        # and task state information: https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/tasks/pointing/Pointing.py#L131 this is mainly providing time relavant info.
        # https://github.com/BaiYunpeng1949/user-in-the-box/blob/1ca6e96d00c0603c2b03403bab9a13d9cd813a56/uitb/perception/proprioception/basic_with_end_effector_position/BasicWithEndEffectorPosition.py#L31
        # Update according to my experimental settings - the dot position, vel, acc
        # Get the current position, velocities, and accelerations in terms of x, y, z (dof=3). The shape consistency: 3.
        obs = {
            "joint_dot_positions": self._data.qpos[0:3].copy(),
            "joint_dot_velocities": self._data.qvel[0:3].copy()
        }
        return obs

    def render(self, mode="human"):
        print("This method will be developped in the future.")

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
    self._frame_skip = int((1/self._action_sample_freq) / self._model.opt.timestep)   # 0.05/0.002=25

    # Initialise thresholds and counters
    self._target_switch_interval = 2*self._action_sample_freq
    self._steps = 0
    self._max_trials = 8  # Cater to different layouts.
    self._trial_idx = 0

    # Get targets (geoms that belong to "smart-glass-pane")
    sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
    self._target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
    self._targets = [self._model.geom(idx) for idx in self._target_idxs]
    self._target_idx = None
    self._target = None

    # Added parts
    # Determine the idx of grids which needs to be traversed in some sequence - this should be changed according to different layouts
    self._sequence_target_idxs = [3, 6, 8, 15, 18]
    # The reading result buffer - should has the same length
    self._sequence_results_idxs = [0, 0, 0, 0, 0]
    self._ep_len = 2500

    # Define observation space
    self._width = 80
    self._height = 80
    self.observation_space = Box(low=0, high=255, shape=(3, self._width, self._height)) # width, height correctly set?

    # Define action space
    self.action_space = Box(low=-1, high=1, shape=(2,))   # Note: use the relative movement might be more close to the visual behaviors, such as saccades

    # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
    self._rangefinder_cutoff = 0.1

    # Initialise context, cameras
    self._context = Context(self._model, max_resolution=[1280, 960])
    self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                           resolution=[self._width, self._height], maxgeom=100, dt=1/self._action_sample_freq)
    self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                           dt=1/self._action_sample_freq)

  def _get_obs(self):
    # Render the image
    rgb, _ = self._eye_cam.render()
    # Preprocess
    rgb = np.transpose(rgb, [2, 0, 1])    # TODO Try the stacking frame technique
    return self.normalise(rgb, 0, 255, -1, 1)

  def reset(self):

    # Reset mujoco sim
    mujoco.mj_resetData(self._model, self._data)

    # Reset counters
    self._steps = 0
    self._trial_idx = 0

    # Choose one target at random
    # self._switch_target()

    # Color all the selected grids
    for idx in self._sequence_target_idxs:
      self._model.geom(idx).rgba[0:3] = [0.8, 0, 0]

    return self._get_obs()

  def _switch_target(self):

    # Sample a random target
    idx = np.random.choice(len(self._target_idxs))
    self._target_idx = self._target_idxs[idx]
    self._target = self._targets[idx]

    # Set position of target (the yellow border)
    self._model.body("target").pos = self._model.body("smart-glass-pane").pos + self._target.pos

    # Do a forward so everything will be set
    mujoco.mj_forward(self._model, self._data)


  def render(self, mode="rgb_array"):
    rgb, _ = self._env_cam.render()
    rgb_eye, _ = self._eye_cam.render()   # TODO debug
    return rgb, rgb_eye

  def normalise(self, x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
    return (b - a) * ((x - x_min) / (x_max - x_min)) + a

  def step(self, action):
    # Normalise action from [-1, 1] to actuator control range
    action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
    action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

    # Set motor control
    self._data.ctrl[:] = action

    # Advance the simulation
    mujoco.mj_step(self._model, self._data, self._frame_skip)
    self._steps += 1

    # Update fixate point based on rangefinder
    x = self._data.sensor("rangefinder").data
    x = x if x >= 0 else self._rangefinder_cutoff
    self._model.geom("fixate-point").pos[2] = -x

    # Check for collisions, estimate reward
    reward = 0
    if x != self._rangefinder_cutoff and len(self._data.contact.geom2) > 0:   # Cater to <exclude body1="eye" body2="target"/>
      geom2 = self._data.contact.geom2[0]
      # if self._target_idx == geom2:
      #   reward = 1
      # If the geom2 is in the target idxs array, then the rewards are applied, the environment changes a little bit
      if geom2 in self._sequence_target_idxs and geom2 not in self._sequence_results_idxs:
        reward = 0.05
        # Update the environment
        acc_g, acc_b = 0.8/self._target_switch_interval, 0.8/self._target_switch_interval
        self._model.geom(geom2).rgba[0:3] += [0, acc_g, acc_b]

    # Do a forward so everything will be set
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
    if self._steps >= self._ep_len or self._sequence_results_idxs.count(0) <= 0:
      terminate = True
      corr = np.corrcoef((self._sequence_target_idxs, self._sequence_results_idxs))[0, 1]
      if corr >= 0.85:
        reward = 5
    else:
      terminate = False
      for idx in self._sequence_target_idxs:
        # Check whether the grid has been fixated for enough time - the pane has changed to grey [0.8, 0.8, 0.8]
        if (sum(self._model.geom(idx).rgba[0:3]) >= 3*0.8) and (idx not in self._sequence_results_idxs):
          # Update the results
          for i in range(len(self._sequence_results_idxs)):
            if self._sequence_results_idxs[i] == 0:
              self._sequence_results_idxs[i] = idx
              break
          # Update the reward
          reward = 1

    # Note: Now we have a generalized eye tracking policy that can cater to various scenarios.
    #       But it might not be closely related to the information processing procedure - the sequential processing.
    #       To be more specific, the agent needs to learn to traverse information in a left2right and top2bottom fashion.
    #       This might be more intuitive.

    # Reward function:
    #   1. Time penalty. R=0 or -0.01 per step.
    #   2. Midterm rewards: get one grid's information received 1 in the end.
    #   3. Final rewards: R=5 all grids are done and the order is correct. Could be evaluated by similarity analysis.

    return self._get_obs(), reward, terminate, {}

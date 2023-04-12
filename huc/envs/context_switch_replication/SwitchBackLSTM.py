import math

import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box, Dict

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class SwitchBackLSTM(Env):

    def __init__(self):

        # RNG
        self.rng = np.random.default_rng()

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Open the mujoco model
        self._xml_path = os.path.join(directory, "context-switch-12-inter-line-spacing-50-v3.xml")
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._steps = 0
        self._steps_without_read = 0
        self._max_steps_without_read = int(self._action_sample_freq*4)
        self._dwell_time = 0
        self._dwell_time_threshold = int(self._action_sample_freq*0.5)

        # Traverse order of targets
        self._target_idx = 0
        self._traverse_names = ["grid-7", "grid-8", "grid-9", "grid-10",
                                "grid-25", "grid-26", "grid-27", "grid-28",
                                "grid-43", "grid-44", "grid-45", "grid-46"]
        self._traverse_order = self._get_traverse_order(self._traverse_names)
        self._target = self._traverse_order[self._target_idx]

        # Flash the geom that needs to be read
        self._flashing = False
        self._flash_duration_steps = 5
        self._flash_stop = -1
        self._flash_id = -1
        if self._flash_duration_steps == 0:
            raise RuntimeWarning("Flash has been set to zero steps, so the targets will not flash")
        if self._flash_duration_steps > self._dwell_time_threshold:
            raise NotImplementedError("Flash duration must be shorter than dwell time threshold")

        # Define observation space
        self._width = 40
        self._height = 40
        self.observation_space = Dict({
            "vision": Box(low=0, high=255, shape=(3, self._width, self._height)),
            "proprioception": Box(low=-1, high=1, shape=(self._model.nq+self._model.nu,))})
        # TODO set "proprioception" low and high according to joint/control limits, or make sure to output normalized
        #  joint/control values as observations

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Define a cutoff for rangefinder (in meters, could be something like 3 instead of 0.1)
        self._rangefinder_cutoff = 1.0

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._cam_eye_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    def _get_traverse_order(self, names):
        return [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]

    def _get_obs(self):

        # Render the image
        rgb, _ = self._eye_cam.render()
        # Preprocess
        rgb_foveated = rgb#self._foveate(img=rgb)
        rgb_foveated = np.transpose(rgb_foveated, [2, 0, 1])
        rgb_normalize = self.normalise(rgb_foveated, 0, 255, -1, 1)

        # Get joint values (qpos) and motor set points (ctrl) -- call them proprioception for now
        proprioception = np.concatenate([self._data.qpos, self._data.ctrl])

        return {"vision": rgb_normalize, "proprioception": proprioception}

    def _sample_initial_state(self):

        # Randomly sample joint values for eye-joint-x and eye-joint-y
        # joints = ["eye-joint-x", "eye-joint-y"]
        # for joint in joints:
        #     joint_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        #     self._data.qpos[joint_idx] = np.random.uniform(*self._model.jnt_range[joint_idx])

        # Randomly sample motor values for eye-x-motor and eye-y-motor
        motors = ["eye-x-motor", "eye-y-motor"]
        for motor in motors:
            motor_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)
            self._data.ctrl[motor_idx] = np.random.uniform(*self._model.actuator_ctrlrange[motor_idx])

    def reset(self, scramble=True):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Random initial state
        # self._sample_initial_state()

        # Reset counters
        self._steps = 0
        self._steps_without_read = 0
        self._dwell_time = 0

        # Reset other stuff
        self._flash_stop = -1
        self._flash_id = -1
        self._flashing = False

        # Scramble traverse order if necessary
        if scramble:
            self._traverse_order = self._get_traverse_order(self.rng.permutation(self._traverse_names))
        else:
            self._traverse_order = self._get_traverse_order(self._traverse_names)
        self._traverse_order = [self._traverse_order[0]]
        self._target_idx = 0
        self._target = self._traverse_order[self._target_idx]

        # Set flashing
        self._start_flash()

        mujoco.mj_forward(self._model, self._data)

        return self._get_obs()

    def _ray_from_site(self, site_name):
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = site.xmat.reshape((3, 3))[:, 2]
        # Exclude the body that contains the site, like in the rangefinder sensor
        siteid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        bodyexclude = self._model.site_bodyid[siteid]
        geomid_out = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            self._model, self._data, pnt, vec, geomgroup=None, flg_static=1,
            bodyexclude=bodyexclude, geomid=geomid_out)
        return distance, geomid_out[0]

    def angle_between(self, v1, v2):
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        def unit_vector(vec):
            return vec/np.linalg.norm(vec)
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _angle_from_target(self, site_name):

        # Get vector pointing direction from site
        site = self._data.site(site_name)
        pnt = site.xpos
        vec = pnt+site.xmat.reshape((3, 3))[:, 2]

        # Get vector pointing direction to target
        target_vec = self._data.geom(self._target).xpos - pnt

        # Estimate distance as angle
        angle = self.angle_between(vec, target_vec)

        return angle

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _foveate(self, img):

        # Define the blurring level
        sigma = 1

        # Define the foveal region
        fov = self._cam_eye_fovy
        foveal_size = 30
        foveal_pixels = int(foveal_size / 2 * img.shape[0] / fov)
        foveal_center = (img.shape[0] // 2, img.shape[1] // 2)

        # Define the blur kernel
        kernel_size = foveal_pixels * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))

        # Create a linear ramp for the blur kernel
        ramp = np.linspace(0, 1, kernel_size)
        kernel[:, foveal_pixels] = ramp
        kernel[foveal_pixels, :] = ramp

        # Create a circular mask for the foveal region
        y, x = np.ogrid[-foveal_center[0]:img.shape[0] - foveal_center[0], -foveal_center[1]:img.shape[1] - foveal_center[1]]
        mask = x ** 2 + y ** 2 <= (foveal_pixels ** 2)

        # Apply a Gaussian blur to each color channel separately
        blurred = np.zeros_like(img)
        for c in range(3):
            blurred_channel = gaussian_filter(img[:, :, c], sigma=sigma)
            blurred[:, :, c][~mask] = blurred_channel[~mask]

        # Combine the original image and the blurred image
        foveated = img.copy()
        foveated[~mask] = blurred[~mask]

        return foveated

    def _start_flash(self):

        self._flashing = True

        # Flash (= change color) for given number of steps
        self._flash_stop = self._steps + self._flash_duration_steps

        # Get id of target geom
        self._flash_id = self._target

        # Change color
        self._model.geom(self._flash_id).rgba = np.array([1, 0.8, 0.0, 1.0])

    def _stop_flash(self):

        # Change color back
        self._model.geom(self._flash_id).rgba = np.array([0, 0, 0, 1])

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):

        terminate = False

        # Normalise action from [-1, 1] to actuator control range
        # action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        # action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # self._data.ctrl += action
        for idx, act_name in enumerate(["eye-x-motor", "eye-y-motor"]):
            act_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self._data.ctrl[act_idx] = np.clip(self._data.ctrl[act_idx]+0.05*action[idx], *self._model.actuator_ctrlrange[idx])

        # Set motor control
        # self._data.ctrl[:] = action

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Check if we should stop flashing
        if self._flashing and self._steps >= self._flash_stop:
            self._stop_flash()

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        if geomid == self._target:
            self._dwell_time += 1
        else:
            self._dwell_time = 0

        # Check if current target has been read
        if self._dwell_time >= self._dwell_time_threshold:

            # Give big reward for reading target
            reward = 10

            # Advance to next target
            terminate = self._next_target()

        else:

            # Calculate reward based on angle difference
            reward = 0.1 * (np.exp(-10*self._angle_from_target(site_name="rangefinder-site"))-0)

            # Increase counter
            self._steps_without_read += 1

        # Check if we should switch target (early termination)
        if self._steps_without_read >= self._max_steps_without_read:
            terminate = self._next_target()

        return self._get_obs(), reward, terminate, {}

    def _next_target(self):

        self._steps_without_read = 0
        self._dwell_time = 0
        self._target_idx += 1
        self._target = self._traverse_order[self._target_idx] if self._target_idx < len(self._traverse_order) else None

        # Flash next target (if we haven't reached end of traverse list)
        if self._target is not None:
            self._start_flash()
            return False
        else:
            return True

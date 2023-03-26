import math

import numpy as np
import mujoco
import os

from gym import Env
from gym.spaces import Box

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context


class SwitchBack(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = config['rl']['mode']

        # Open the mujoco model
        self._model = mujoco.MjModel.from_xml_path(os.path.join(directory,
                                                                "context-switch-12-inter-line-spacing-50-v2.xml"))
        self._data = mujoco.MjData(self._model)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Define the length of training and testing episodes
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            self._ep_len = 80
        else:
            self._ep_len = 2000

        # Initialise thresholds and counters
        self._steps = 0
        self._max_trials = 1
        self._trials = 0
        self._b_change = 0.2

        # Get targets (geoms that belong to "smart-glass-pane")  TODO read how to handle nodes under nodes.
        sgp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane")
        self._reading_target_idxs = np.where(self._model.geom_bodyid == sgp_body)[0]
        self._reading_targets = [self._model.geom(idx) for idx in self._reading_target_idxs]
        self._reading_target_idx = None
        self._reading_target = None

        # Get the background (geoms that belong to "background-pane")
        bp_body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == bp_body)[0]
        self._background_idx0 = self._background_idxs[0]

        # Define the default text grid size and rgba from a sample grid idx=0
        self._DEFAULT_TEXT_SIZE = self._model.geom(self._reading_target_idxs[0]).size[0:4].copy()
        self._DEFAULT_TEXT_RGBA = self._model.geom(self._reading_target_idxs[0]).rgba[0:4].copy()
        self._RUNTIME_TEXT_RGBA = None
        self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0] * 4 / 3, self._DEFAULT_TEXT_SIZE[1],
                           self._DEFAULT_TEXT_SIZE[2] * 4 / 3]
        self._HINT_RGBA = [0.8, 0.8, 0, 1]
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:4].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [0.8, 0, 0, self._model.geom(self._background_idx0).rgba[3].copy()]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._reading_target_dwell_interval = 2 * self._action_sample_freq
        self._sequence_target_idxs = []
        # The reading result buffer - should has the same length
        self._sequence_results_idxs = []
        self._default_idx = -1
        self._num_targets = 0
        self._acc_reading_target = self._b_change / self._reading_target_dwell_interval

        # Define the grids on the background pane
        self._background_on = None
        self._background_dwell_interval = self._reading_target_dwell_interval
        self._background_show_flag = None
        self._background_show_timestep = None
        self._background_trials = None
        if self._mode == 'train' or self._mode == 'continual_train':
            self._background_max_trials = 1
        else:
            self._background_max_trials = 4
        self._acc_background = self._b_change / self._background_dwell_interval
        self._background_show_interval = 2.3 * self._reading_target_dwell_interval
        self._background_last_on_steps = None

        # Define the relocation distraction relevant variables
        self._relocating_dwell_interval = 0.5 * self._reading_target_dwell_interval
        self._relocating_steps = None
        self._relocating_neighbors = None
        self._relocating_incorrect_steps = None
        # Neighbor distance threshold
        self._relocating_dist_neighbor = 0.01

        # Define observation space
        self._width = 40
        self._height = 40
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
        rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)
        rgb_foveated = self._foveate(img=rgb_normalize)
        return rgb_foveated

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0
        self._trials = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Reset the background scene and variables
        self._background_trials = 0
        self._background_on = False
        self._background_last_on_steps = 0
        self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()

        # Relocating / pick-up issues
        self._relocating_steps = 0
        self._relocating_neighbors = []

        # TODO change according to the settings
        # Specify according to the training or non-trainings:
        #  training simple but generalizable abilities, non-training actual tasks
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            # Initialize eye ball rotation angles
            eye_x_motor_init_range = [-0.5, 0.5]
            eye_y_motor_init_range = [-0.25, 0.25]
            action = [np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1]),
                      np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])]
            # TODO try to use data.xmat directly set orientations. The model.quat should not be changed.
            for i in range(10):
                # Set motor control
                self._data.ctrl[:] = action
                # Advance the simulation
                mujoco.mj_step(self._model, self._data, self._frame_skip)

            # Define the target reading grids
            self._sequence_target_idxs = np.random.choice(self._reading_target_idxs.tolist(), 3, False)
            self._num_targets = len(self._sequence_target_idxs)

            # Define the background events
            # self._background_show = True
            # # self._background_show_timestep = int(0.5 * self._reading_target_dwell_interval)
            self._background_show_flag = np.random.choice([True, False])
            # self._background_show_timestep = np.random.randint(0, self._reading_target_dwell_interval)

            # Decide whether or not show the reading grids
            if self._background_show_flag == False:
                self._switch_target(idx=self._sequence_target_idxs[0])

        # Non-trainings
        else:
            # Reading grids
            self._sequence_target_idxs = self._reading_target_idxs.tolist()
            self._sequence_results_idxs = [self._default_idx for _ in self._sequence_target_idxs]
            self._switch_target(idx=self._sequence_target_idxs[0])

            # Relocating
            self._relocating_incorrect_steps = 0

    def _switch_target(self, idx):

        for _idx in self._reading_target_idxs:
            if _idx != idx:
                self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        self._model.geom(idx).rgba[0:4] = self._HINT_RGBA.copy()
        self._model.geom(idx).size[0:3] = self._HINT_SIZE.copy()

        # Update the target id
        self._reading_target_idx = idx

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):

        # Trainings
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            if self._background_on == False:
                if self._background_trials < self._background_max_trials:
                    # Show the background event by changing to a brighter color
                    if self._steps >= 0:
                        self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                        self._background_on = True
            else:
                # Identify whether should stop the background event
                if self._model.geom(self._background_idx0).rgba[2] >= self._b_change:
                    self._background_trials += 1
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    self._background_on = False

        # Non-trainings
        else:       # TODO change it later.
            if self._background_on == False:
                if (self._steps - self._background_last_on_steps) >= self._background_show_interval and self._background_trials < self._background_max_trials:
                    # Background
                    self._background_on = True
                    self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                    # Reading grids
                    self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                    self._model.geom(self._reading_target_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                    self._model.geom(self._reading_target_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            else:
                if self._model.geom(self._background_idx0).rgba[2] >= self._b_change:
                    # Background
                    self._background_trials += 1
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    self._background_on = False
                    # Update the step buffer
                    self._background_last_on_steps = self._steps
                    # Reading grids distractions
                    self._find_neighbors()    # TODO baseline vs issues on relocating
                    # self._model.geom(self._reading_target_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()
                    # self._model.geom(self._reading_target_idx).size[0:3] = self._HINT_SIZE.copy()

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):

        # TODO maybe later the memory mechanism can be added here.

        # TODO generalize this using the self._relocating_neighbor_dist later
        target_grid_idx = self._reading_target_idx
        target_xpos = self._data.geom(target_grid_idx).xpos

        neighbors = []

        for grid_idx in self._sequence_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - target_xpos)
            if dist <= self._relocating_dist_neighbor:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()
                self._model.geom(grid_idx).size[0:3] = self._HINT_SIZE.copy()

        self._relocating_neighbors = neighbors.copy()

    def _ray_from_site(self, site_name):
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

    def _dist_from_target(self, dist_plane, ray_site_name, target_id):
        ray_site = self._data.site(ray_site_name)
        pnt = ray_site.xpos
        vec = ray_site.xmat.reshape((3, 3))[:, 2]

        # Define the x-z plane equation
        a, b, c = 0, 1, 0  # Normalize the vector of the x-z plane
        # dist_plane = - self._data.body("smart-glass-pane").xpos[1]  # Distance from origin to plane
        # Calculate the intersection point
        t = - (a * pnt[0] + b * pnt[1] + c * pnt[2] + dist_plane) / (a * vec[0] + b * vec[1] + c * vec[2])
        itsct_pnt = pnt + t * vec
        # Get the target point
        target_pnt = self._data.geom(target_id).xpos
        # Calculate the distance
        dist = math.sqrt((itsct_pnt[0]-target_pnt[0])**2 + (itsct_pnt[2]-target_pnt[2])**2)
        return dist

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _foveate(self, img):

        # Define the foveal region
        fov = 90      # TODO get the fov value from mujoco model
        foveal_size = 20
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
            blurred_channel = gaussian_filter(img[:, :, c], sigma=5)
            blurred[:, :, c][~mask] = blurred_channel[~mask]

        # Combine the original image and the blurred image
        foveated = img.copy()
        foveated[~mask] = blurred[~mask]

        return foveated

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes, trainings
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            if self._background_show_flag == True:
                self._update_background()
        # Non-trainings
        else:
            self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        # Trainings
        if (self._mode == 'train') or (self._mode == 'continual_train'):
            if self._background_show_flag == True:
                target_idx = self._background_idx0
                acc = self._acc_background
            else:
                target_idx = self._reading_target_idx
                acc = self._acc_reading_target
        # Non-trainings
        else:
            if self._background_on == True:
                target_idx = self._background_idx0
                acc = self._acc_background
            else:
                target_idx = self._reading_target_idx
                acc = self._acc_reading_target

        # Focus on targets detection
        if geomid == target_idx:
            reward = 1
            # Update the environment
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, acc])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # print(geomid, target_idx, self._steps, reward, self._background_on, self._model.geom(self._background_idx0).rgba[0:4])      # TODO debug delete later

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Trainings
            if (self._mode == 'train') or (self._mode == 'continual_train'):
                if self._background_show_flag == True:
                    if self._background_trials >= self._background_max_trials:
                        terminate = True
                else:
                    if self._model.geom(self._reading_target_idx).rgba[2] >= self._b_change:
                        self._trials += 1
                        if self._trials >= self._max_trials:
                            terminate = True
            # Non-trainings
            else:
                # Check the relocating / pick-up issues
                if self._relocating_steps >= self._relocating_dwell_interval:
                    for idx in self._relocating_neighbors:
                        if idx != self._reading_target_idx:
                            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
                            self._relocating_steps = 0
                else:
                    if geomid in self._relocating_neighbors:
                        self._relocating_steps += 1
                        if geomid != self._reading_target_idx:
                            self._relocating_incorrect_steps += 1

                # Check whether the grid has been fixated for enough time
                if (self._model.geom(self._reading_target_idx).rgba[2] >= self._b_change) and (self._reading_target_idx not in self._sequence_results_idxs):

                    # Update the results for complex testing mode
                    for i in range(self._num_targets):
                        if self._sequence_results_idxs[i] == self._default_idx:
                            self._sequence_results_idxs[i] = self._reading_target_idx
                            break
                    # Terminate the loop if all grids are traversed, or switch to the next grid
                    if self._reading_target_idx >= self._sequence_target_idxs[-1]:
                        terminate = True
                    else:
                        self._switch_target(idx=self._reading_target_idx + 1)

                if terminate:
                    print('The total timesteps is: {}. \n'
                          'The incorrect relocating timesteps / The switch back duration is: {}. \n'
                          'The reading goodput is: {} (grids per timestep). \n'
                          'The switch back error rate is: {}%.'.
                          format(self._steps, self._relocating_incorrect_steps, round(len(self._sequence_target_idxs) / self._steps, 5),
                                 round(100*self._relocating_incorrect_steps/(self._background_max_trials*self._relocating_dwell_interval), 2)))

        return self._get_obs(), reward, terminate, {}

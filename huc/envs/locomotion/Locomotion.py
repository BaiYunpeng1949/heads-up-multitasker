import math

import numpy as np
from collections import Counter
import mujoco
import os

from gym import Env
from gym.spaces import Box

import yaml
from scipy.ndimage import gaussian_filter

from huc.utils.rendering import Camera, Context

READING_MODE = 'reading'
BACKGROUND_MODE = 'background'
RELOCATING_MODE = 'relocating'


class LocomotionBase(Env):

    def __init__(self):

        # Get directory of this file
        directory = os.path.dirname(os.path.realpath(__file__))

        # Read the configurations from the YAML file.
        root_dir = os.path.dirname(os.path.dirname(directory))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config['rl']['mode']

        # Open the mujoco model
        self._xml_path = os.path.join(directory, self._config['mj_env']['xml'])
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data = mujoco.MjData(self._model)
        # Forward pass to initialise the model, enable all variables
        mujoco.mj_forward(self._model, self._data)

        # Define how often policy is queried
        self._action_sample_freq = 20
        self._frame_skip = int((1 / self._action_sample_freq) / self._model.opt.timestep)  # 0.05/0.002=25

        # Initialise thresholds and counters
        self._steps = None
        # The rgba cumulative change for finishing a fixation
        self._rgba_delta = 0.2

        # Get the primitives idxs in MuJoCo
        self._eye_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-x")
        self._eye_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "eye-joint-y")
        self._head_joint_y_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-y")
        self._head_joint_x_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "head-joint-x")
        self._head_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._eye_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self._sgp_ils100_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                      "smart-glass-pane-interline-spacing-100")
        self._sgp_bc_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                                  "smart-glass-pane-bottom-center")
        self._sgp_mr_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "smart-glass-pane-mid-right")
        self._bgp_body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "background-pane")

        # Get targets (geoms that belong to "smart-glass-pane")
        # Inter-line-spacing-100
        self._ils100_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_ils100_body_idx)[0]
        # Bottom-center
        self._bc_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_bc_body_idx)[0]
        # Middle-right
        self._mr_reading_target_idxs = np.where(self._model.geom_bodyid == self._sgp_mr_body_idx)[0]
        # All layouts
        self._all_layouts_reading_traget_idxs = np.concatenate((self._ils100_reading_target_idxs.copy(),
                                                                self._bc_reading_target_idxs.copy(),
                                                                self._mr_reading_target_idxs.copy()))
        # General reading target index
        self._reading_target_idxs = None  # The reading target idxs list
        self._reading_target_idx = None  # The exact reading target idx

        # Define the default text grid size and rgba from a sample grid idx=0, define the hint text size and rgba
        sample_grid_idx = self._ils100_reading_target_idxs[0].copy()
        self._DEFAULT_TEXT_SIZE = self._model.geom(sample_grid_idx).size[0:4].copy()
        self._DEFAULT_TEXT_RGBA = [0, 0, 0, 1]
        self._RUNTIME_TEXT_RGBA = None
        # self._HINT_SIZE = [self._DEFAULT_TEXT_SIZE[0] * 6 / 5, self._DEFAULT_TEXT_SIZE[1],
        #                    self._DEFAULT_TEXT_SIZE[2] * 6 / 5]
        self._HINT_SIZE = self._DEFAULT_TEXT_SIZE.copy()
        self._HINT_RGBA = [1, 1, 0, 1]

        # Get the background (geoms that belong to "background-pane")
        self._background_idxs = np.where(self._model.geom_bodyid == self._bgp_body_idx)[0]
        self._background_idx0 = self._background_idxs[0].copy()
        # Define the default background grid size and rgba from a sample grid idx=0, define the event text size and rgba
        self._DEFAULT_BACKGROUND_SIZE = self._model.geom(self._background_idx0).size[0:4].copy()
        self._DEFAULT_BACKGROUND_RGBA = self._model.geom(self._background_idx0).rgba[0:4].copy()
        self._EVENT_RGBA = [1, 0, 0, 1]

        # Define the idx of grids which needs to be traversed sequentially on the smart glass pane
        self._reading_target_dwell_timesteps = int(2 * self._action_sample_freq)
        self._reading_rgb_change_per_step = self._rgba_delta / self._reading_target_dwell_timesteps

        # Define the events on the background pane
        self._background_on = None
        self._background_trials = None
        self._background_dwell_timesteps = self._reading_target_dwell_timesteps
        self._background_rgba_change_per_step = self._rgba_delta / self._background_dwell_timesteps

        # Define the locomotion variables
        self._displacement_lower_bound = self._model.jnt_range[self._head_joint_y_idx][0].copy()
        self._displacement_upper_bound = self._model.jnt_range[self._head_joint_y_idx][1].copy()
        self._nearest_head_xpos_y = self._data.body(self._head_body_idx).xpos[1].copy() + self._displacement_lower_bound
        self._furthest_head_xpos_y = self._data.body(self._head_body_idx).xpos[
                                         1].copy() + self._displacement_upper_bound
        self._head_disp_per_timestep = (self._displacement_upper_bound - self._displacement_lower_bound) / 400

        # Define observation space
        self._width = self._config['mj_env']['width']
        self._height = self._config['mj_env']['height']
        self.observation_space = Box(low=0, high=255,
                                     shape=(3, self._width, self._height))  # width, height correctly set?

        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(2,))

        # Initialise context, cameras
        self._context = Context(self._model, max_resolution=[1280, 960])
        self._eye_cam = Camera(self._context, self._model, self._data, camera_id="eye",
                               resolution=[self._width, self._height], maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._env_cam = Camera(self._context, self._model, self._data, camera_id="env", maxgeom=100,
                               dt=1 / self._action_sample_freq)
        self._eye_cam_fovy = self._model.cam_fovy[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")]

    def _get_obs(self):

        # Render the image
        rgb, _ = self._eye_cam.render()

        # Preprocess
        # Foveated vision applied
        if 'foveate' in self._config['rl']['train']['checkpoints_folder_name']:
            rgb_foveated = self._foveate(img=rgb)
            rgb_foveated = np.transpose(rgb_foveated, [2, 0, 1])
            rgb_normalize = self.normalise(rgb_foveated, 0, 255, -1, 1)
            return rgb_normalize
        # Foveated vision not applied
        else:
            rgb = np.transpose(rgb, [2, 0, 1])
            rgb_normalize = self.normalise(rgb, 0, 255, -1, 1)
            return rgb_normalize

    def reset(self):

        # Reset mujoco sim
        mujoco.mj_resetData(self._model, self._data)

        # Reset counters
        self._steps = 0

        # Reset the scene
        self._reset_scene()

        return self._get_obs()

    def _reset_scene(self):

        # Reset the all reading grids - hide
        for idx in self._all_layouts_reading_traget_idxs:
            self._model.geom(idx).rgba[3] = 0

        # Reset the background scene
        self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()

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
        return

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
        dist = math.sqrt((itsct_pnt[0] - target_pnt[0]) ** 2 + (itsct_pnt[2] - target_pnt[2]) ** 2)
        return dist

    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a

    def _foveate(self, img):
        """
        Foveate the image, note that the shape of image has to be (height, width, 3)
        """

        # Define the blurring level
        sigma = 1

        # Define the foveal region
        fov = self._eye_cam_fovy
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
        y, x = np.ogrid[-foveal_center[0]:img.shape[0] - foveal_center[0],
               -foveal_center[1]:img.shape[1] - foveal_center[1]]
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

    def render(self, mode="rgb_array"):
        rgb, _ = self._env_cam.render()
        rgb_eye, _ = self._eye_cam.render()
        return rgb, rgb_eye

    def step(self, action):
        return


class LocomotionTrain(LocomotionBase):

    def __init__(self):
        super().__init__()

        # Initialize the episode length
        self._ep_len = 100

        # Initialize the steps on target: either reading or background
        self._steps_on_target = None

        # Initialize the counter, and the max number of trials for the reading task
        self._reading_trials = None
        self._reading_max_trials = 1

        # Initialize the max number of trials for the background task
        self._background_max_trials = 1

        # Define the flag for the background task - show or not
        self._background_show_flag = None

        # Define the initial displacement of the agent's head
        self._head_init_displacement_y = None

    def _reset_scene(self):
        super()._reset_scene()

        # Initialize eye ball rotation angles
        eye_x_motor_init_range = [-0.5, 0.5]
        eye_y_motor_init_range = [-0.5, 0.4]

        init_angle_x = np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1])
        init_angle_y = np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])

        self._data.qpos[self._eye_joint_x_idx] = init_angle_x
        self._data.qpos[self._eye_joint_y_idx] = init_angle_y

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        random_choice = np.random.choice([0, 1, 2], 1)
        if random_choice == 0:
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
        elif random_choice == 1:
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
        else:
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Define the target reading grids from the selected reading layouts
        self._reading_trials = 0
        self._reading_target_idxs = np.random.choice(self._reading_target_idxs.tolist(), 3, False)

        # Define the background events
        self._background_trials = 0
        self._background_on = False
        self._background_show_flag = np.random.choice([True, False])

        # Initialize the steps on target
        self._steps_on_target = 0

        # Initialize the locomotion slide displacement
        self._data.qpos[self._head_joint_y_idx] = 0

        # Define whether or not show the reading grids
        if self._background_show_flag == False:  # Show the reading grids
            self._switch_target(idx=self._reading_target_idxs[0])
            self._head_init_displacement_y = np.random.uniform(self._displacement_lower_bound,
                                                               self._displacement_upper_bound)
            self._data.qpos[self._head_joint_y_idx] = self._head_init_displacement_y
        else:
            self._data.qpos[self._head_joint_y_idx] = self._displacement_upper_bound

        mujoco.mj_forward(self._model, self._data)

    def _update_background(self):
        super()._update_background()

        # The background pane is showing events - the red color events show
        if self._background_show_flag == True:
            if self._background_on == False:
                if self._background_trials < self._background_max_trials:
                    # Show the background event by changing to a brighter color
                    self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                    self._background_on = True
        # The background pane is not showing events - the head is moving
        else:
            # Move the head if it has not getting close to the background enough
            if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_timestep

        mujoco.mj_forward(self._model, self._data)

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background events
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._background_show_flag == True:
            target_idx = self._background_idx0
            change_rgba = self._background_rgba_change_per_step
        else:
            target_idx = self._reading_target_idx
            change_rgba = self._reading_rgb_change_per_step

        # Focus on targets detection
        if geomid == target_idx:
            # Sparse reward
            reward = 1

            # Update the steps on target
            self._steps_on_target += 1

            # Update the environment
            self._model.geom(geomid).rgba[2] += change_rgba

            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Background events scenario
            if self._background_show_flag == True:
                if self._steps_on_target >= self._background_dwell_timesteps:
                    self._background_trials += 1
                    # For multiple trials - remember to initialize scene - flag and geom color
                    self._steps_on_target = 0
                    if self._background_trials >= self._background_max_trials:
                        terminate = True
            # Reading grids scenario
            else:
                if self._steps_on_target >= self._reading_target_dwell_timesteps:
                    self._reading_trials += 1
                    # For multiple trials - remember to initialize scene - flag and geom color
                    self._steps_on_target = 0
                    if self._reading_trials >= self._reading_max_trials:
                        terminate = True

        return self._get_obs(), reward, terminate, {}


class LocomotionRelocationTrain(LocomotionBase):
    def __init__(self):
        super().__init__()

        # Initialize the episode length and training trial thresholds
        self._ep_len = 400
        self._max_trials = 1
        self._trials = 0

        # Initialize the steps on target: either reading or background
        self._steps_on_target = None

        # Initialize the relocation relevant variables
        self._neighbor_dist_thres = None
        self._relocating_center_grid_idx = None
        self._relocating_dwell_steps_thres = self._reading_target_dwell_timesteps
        self._neighbors = None
        self._neighbors_steps = None

        # Color settings for relocation
        self._relocation_target_idx = None
        self._RELOCATION_HINT_RGBA = [1, 1, 0, 0.5]
        self._relocation_rgb_change_per_fixation = 1 / self._relocating_dwell_steps_thres
        self._relocation_alpha_change_per_fixation = (1 - self._RELOCATION_HINT_RGBA[
            3]) / self._relocating_dwell_steps_thres

        # Define the transition of 3 modes: 1-reading, 2-background, 3-relocation
        self._task_mode = None
        self._layout = None

        # Define the initial displacement of the agent's head
        self._head_init_displacement_y = None

    def _reset_scene(self):
        super()._reset_scene()

        # Initializations
        self._trials = 0
        self._steps_on_target = 0

        # Eyeball rotation initialization
        # Initialize eye ball rotation angles
        eye_x_motor_init_range = [-0.5, 0.5]
        eye_y_motor_init_range = [-0.5, 0.4]

        init_angle_x = np.random.uniform(eye_x_motor_init_range[0], eye_x_motor_init_range[1])
        init_angle_y = np.random.uniform(eye_y_motor_init_range[0], eye_y_motor_init_range[1])

        self._data.qpos[self._eye_joint_x_idx] = init_angle_x
        self._data.qpos[self._eye_joint_y_idx] = init_angle_y

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        self._layout = np.random.choice(['interline-spacing-100', 'bottom-center', 'middle-right'], 1)
        # self._layout = self._config['rl']['test']['layout_name']
        if self._layout == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101
        elif self._layout == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0121  # Bottom-center layout - not aligned with the visual search flow, more distractions will caused
        elif self._layout == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101
        else:
            raise NotImplementedError('Invalid layout name: {}'.format(self._layout))

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            mujoco.mj_forward(self._model, self._data)

        # Define the task mode: 1-reading, 2-background, 3-relocation
        self._task_mode = np.random.choice([1, 2, 3], 1).copy()
        # Reading task
        if self._task_mode == 1:
            random_reading_target_idxs = np.random.choice(self._reading_target_idxs.tolist(), 3, False)
            # Highlight the target reading grids
            self._switch_target(idx=random_reading_target_idxs[0])
        # Background task
        elif self._task_mode == 2:
            # Define the background events
            self._background_on = False
        # Relocation task
        elif self._task_mode == 3:
            # Randomize the center grid index for evoking neighbors
            self._center_grid_idx = np.random.choice(self._reading_target_idxs.tolist().copy(), 3, False)[0]
            self._neighbors, self._neighbors_steps = [], []
            # Find the neighbors and set up the scene
            self._find_neighbors()
        else:
            raise NotImplementedError('The task mode is not defined! Should be 1-reading, 2-background, 3-relocation!')

        # Initialize the locomotion head position
        self._data.qpos[self._head_joint_y_idx] = 0
        # Reading and Relocation task
        if self._task_mode == 1 or self._task_mode == 3:
            self._head_init_displacement_y = np.random.uniform(self._displacement_lower_bound,
                                                               self._displacement_upper_bound)
            self._data.qpos[self._head_joint_y_idx] = self._head_init_displacement_y
        # Background task
        else:
            self._data.qpos[self._head_joint_y_idx] = self._displacement_upper_bound

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):
        # Find the neighbors of the center grid
        center_xpos = self._data.geom(self._center_grid_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - center_xpos)
            if dist <= self._neighbor_dist_thres:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RELOCATION_HINT_RGBA.copy()

        # Randomly choose one grid in the neighbors list to be the target
        self._relocation_target_idx = np.random.choice(neighbors, 1)[0].copy()

        # Update the neighbors list
        self._neighbors = neighbors.copy()

        # Initialize the steps on each neighbor
        self._neighbors_steps = [0] * len(neighbors)

        if self._mode == "test":  # TODO debug delete later - get the own tests
            self._relocation_target_idx = self._center_grid_idx

    def _update_background(self):
        super()._update_background()

        # The background pane is showing events - the red color events show
        if self._task_mode == 2:
            if self._background_on == False:
                # Show the background event by changing to a brighter color
                self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                self._background_on = True
        # The background pane is not showing events - the head is moving
        else:
            # Move the head if it has not getting close to the background enough
            if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_timestep

        mujoco.mj_forward(self._model, self._data)

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background events
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._task_mode == 1:
            target_idx = self._reading_target_idx
            change_rgba = 0  # self._reading_rgb_change_per_step
        elif self._task_mode == 2:
            target_idx = self._background_idx0
            change_rgba = self._background_rgba_change_per_step
        else:
            target_idx = self._relocation_target_idx
            change_rgba = self._relocation_rgb_change_per_fixation

        # Single target scenarios - 1 reading and 2 background
        if self._task_mode == 1 or self._task_mode == 2:
            # Focus on targets detection
            if geomid == target_idx:
                # Sparse reward
                reward = 1
                # Update the steps on target
                self._steps_on_target += 1
                # Update the environment
                self._model.geom(geomid).rgba[2] += change_rgba
                # Check if the target has been fixated enough
                if self._steps_on_target >= self._background_dwell_timesteps and self._task_mode == 2:
                    self._trials += 1
                if self._steps_on_target >= self._reading_target_dwell_timesteps and self._task_mode == 1:
                    self._trials += 1

        # Multiple targets scenarios - 3 relocation
        else:
            if geomid in self._neighbors:
                # Update the steps on target
                self._neighbors_steps[self._neighbors.index(geomid)] += 1

                if geomid == target_idx:
                    # Sparse reward
                    reward = 1
                    # Update the steps on target
                    self._steps_on_target += 1
                    # Update the reading target - becomes more opaque
                    self._model.geom(geomid).rgba[3] += self._relocation_alpha_change_per_fixation
                else:
                    # Update the distractions - becomes dimmer
                    self._model.geom(geomid).rgba[2] += self._relocation_rgb_change_per_fixation

                # Update the environment
                # De-highlight the geom if it has been fixated enough
                if self._neighbors_steps[self._neighbors.index(geomid)] >= self._relocating_dwell_steps_thres:
                    # Update the neighbors list
                    self._neighbors[self._neighbors.index(geomid)] = -2

                    # Check for the reading target
                    if geomid == target_idx:
                        # Update the reading target
                        self._trials += 1

        # Do a forward so everything will be set
        mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len or self._trials >= self._max_trials:
            terminate = True
        else:
            terminate = False

        return self._get_obs(), reward, terminate, {}


class LocomotionTest(LocomotionBase):

    def __init__(self):
        super().__init__()

        # Initialize the length of the episode
        self._ep_len = 8000

        # Initialize the number of trials
        self._background_max_trials = 8

        # Define the buffer for storing the number of goodput grids
        self._num_read_grids = None

        # Define the relocation distraction relevant variables
        self._relocating_neighbors = None
        # Neighbor distance threshold
        self._relocating_dist_neighbor = 0.010 + 0.0001  # Actual inter-word distance + offset

        # Relocating - version 1: cumulative fixations determine the pick up grid
        self._relocating_pickup_dwell_steps = 10  # int(0.25 * self._reading_target_dwell_timesteps)
        self._relocating_pickup_records = None
        self._relocating_incorrect_num = None
        self._off_background_step = None
        self._switch_back_durations = None

    def _reset_scene(self):
        super()._reset_scene()

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        layout_name = self._config['rl']['test']['layout_name']
        if layout_name == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
        elif layout_name == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
        elif layout_name == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
        else:
            raise ValueError('Invalid layout name.')

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Reading grids
        self._reading_target_idxs = self._reading_target_idxs.tolist()
        self._switch_target(idx=self._reading_target_idxs[0])

        # The counter of read grids
        self._num_read_grids = 0

        # Background flag
        self._background_trials = 0
        self._background_on = False

        # Relocating / pick-up issues
        self._relocating_neighbors = []
        self._relocating_pickup_records = []
        self._relocating_incorrect_num = 0
        self._off_background_step = 0
        self._switch_back_durations = []

        # Locomotion
        self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

    def _update_background(self):
        super()._update_background()

        # If the head is not on the furthest position, it will move towards the background pane
        if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
            # Move the head towards the background pane
            self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_timestep.copy()

        # If the head is on the furthest position, i.e., near the background, starts the red color event
        else:
            # Prevent the head from moving further - stop at the furthest position
            if self._data.body(self._head_body_idx).xpos[1] > self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] = self._displacement_upper_bound.copy()

            # Start the red color event
            if self._background_on == False:
                if self._background_trials < self._background_max_trials:
                    self._background_on = True
                    # Set the red color
                    self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                    # De-highlight the previous job - reading
                    self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                    for idx in self._reading_target_idxs:
                        self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                        self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

            # Check when to stop the red color event when the red color is on
            else:
                if self._model.geom(self._background_idx0).rgba[2] >= self._rgba_delta:
                    self._background_trials += 1
                    # Reset the background variables: the color, status flag
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    self._background_on = False
                    # Reset the head and env-cam position
                    self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

                    # Switch back to reading and display distractions
                    self._find_neighbors()  # TODO baseline vs issues on relocating
                    # Reading without distractions
                    # self._model.geom(self._reading_target_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()
                    # self._model.geom(self._reading_target_idx).size[0:3] = self._HINT_SIZE.copy()
                    # self._relocating_neighbors = [self._reading_target_idx]

                    # Switch back off background timestamp
                    if self._off_background_step == 0:
                        # Only update when last pick up happened and this time stamp was reset to 0
                        self._off_background_step = self._steps

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):

        target_grid_idx = self._reading_target_idx
        target_xpos = self._data.geom(target_grid_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - target_xpos)
            if dist <= self._relocating_dist_neighbor:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()  # Continue to avoid infinite loop
                self._model.geom(grid_idx).size[0:3] = self._HINT_SIZE.copy()

        self._relocating_neighbors = neighbors.copy()

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes, non-trainings
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._background_on == True:
            target_idx = self._background_idx0
            change = self._background_rgba_change_per_step
        else:
            target_idx = self._reading_target_idx
            change = self._reading_rgb_change_per_step

        # Focus on targets detection
        if geomid == target_idx:
            reward = 1
            # Update the environment
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, change])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Check the relocating / pick-up status when the background event is off - version 0328
            if self._background_on == False:
                # Check the relocating / pick-up issues - version 0328
                if geomid in self._relocating_neighbors:
                    self._relocating_pickup_records.append(geomid)
                    for idx in self._relocating_neighbors:
                        # The first grid with cumulative fixations has not been found - TODO do it
                        if self._relocating_pickup_records.count(idx) > self._relocating_pickup_dwell_steps:
                            # Update the switch back duration
                            current_step = self._steps
                            switch_back_duration = current_step - self._off_background_step
                            self._switch_back_durations.append(switch_back_duration)

                            # Update the counters
                            if idx == self._reading_target_idx:
                                pass
                            else:
                                self._relocating_incorrect_num += 1

                            # Draw all distractions back to normal - version continue with the current grid
                            for _idx in self._relocating_neighbors:
                                if _idx != idx:
                                    self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                                    self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
                            # Refresh the current target grid and the sequence results idxs if idx is not the previous target
                            if idx != self._reading_target_idx:
                                self._reading_target_idx = idx

                            # Clear the buffer
                            self._relocating_pickup_records = []
                            self._relocating_neighbors = []
                            self._off_background_step = 0
                            break
            # Check at the start of the next background event, whether the pervious relocating trial was successful
            else:
                # The relocating did not pick up anything at the previous trial
                if self._off_background_step != 0:
                    self._relocating_incorrect_num += 1
                    self._switch_back_durations.append(0)  # TODO handle this later
                    print(
                        'One relocating trial was unable to pick up any grid at the target grid {}. '
                        'The relocating pickup records is: {}. '
                        'The switch back duration is: {}. '
                        'The background trial number is: {}.'
                            .format(self._reading_target_idx, 0,
                                    self._relocating_pickup_records, self._background_trials)
                    )
                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

            # Check whether the grid (background or reading) has been fixated for enough time
            if self._model.geom(self._reading_target_idx).rgba[2] >= self._rgba_delta:

                # Update the number of read grids
                self._num_read_grids += 1

                # Update the intervened relocating where relocating dwell was smaller than the remaining reading time
                if not self._relocating_neighbors:
                    pass
                else:  # When the relocating_neighbor is not empty - the relocation was not over that the relocating dwell was bigger than the remaining reading time
                    # Add new switch back durations
                    current_step = self._steps
                    switch_back_duration = current_step - self._off_background_step
                    self._switch_back_durations.append(switch_back_duration)

                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

                    print(
                        'One relocating trial was finished earlier at target grid {}. The switch back duration is: {}. '
                        'The background trial is: {}'.
                            format(self._reading_target_idx, switch_back_duration, self._background_trials)
                    )

                # Terminate the loop if all moving background trials are done
                if self._background_trials >= self._background_max_trials:
                    terminate = True
                else:
                    # Traverse all girds idx in the target sequence, if reaches the end, start from the beginning
                    if self._reading_target_idx >= self._reading_target_idxs[-1]:
                        self._reading_target_idx = self._reading_target_idxs[0] - 1
                    self._switch_target(idx=self._reading_target_idx + 1)

            if terminate:
                print('The total timesteps is: {}. \n'
                      'The switch back duration is: {}. The durations are: {} \n'
                      'The reading goodput is: {} (grids per timestep). \n'
                      'The switch back error rate is: {}%.'.
                      format(self._steps, np.sum(self._switch_back_durations), self._switch_back_durations,
                             round(self._num_read_grids / self._steps, 5),
                             round(100 * self._relocating_incorrect_num / self._background_max_trials, 2)))

        return self._get_obs(), reward, terminate, {}


class LocomotionTrickyTest(LocomotionBase):

    def __init__(self):
        super().__init__()

        # Initialize the length of the episode
        self._ep_len = 8000

        # Initialize the number of trials
        self._background_max_trials = 8

        # Define the buffer for storing the number of goodput grids
        self._num_read_grids = None

        # Define the relocation distraction relevant variables
        self._relocating_neighbors = None
        # Neighbor distance threshold
        self._relocating_dist_neighbor = 0.010 + 0.0001  # Actual inter-word distance + offset

        # Relocating - version 2: within a limited fixation duration, the grid with the most fixations is picked up
        self._relocating_pickup_dwell_steps = 10
        self._relocating_pickup_records = None
        self._relocating_incorrect_num_method1 = None
        self._relocating_incorrect_num_method2 = None
        self._off_background_step = None
        self._switch_back_durations = None

    def _reset_scene(self):
        super()._reset_scene()

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        layout_name = self._config['rl']['test']['layout_name']
        if layout_name == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
        elif layout_name == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
        elif layout_name == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
        else:
            raise ValueError('Invalid layout name.')

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        # Reading grids
        self._reading_target_idxs = self._reading_target_idxs.tolist()
        self._switch_target(idx=self._reading_target_idxs[0])

        # The counter of read grids
        self._num_read_grids = 0

        # Background flag
        self._background_trials = 0
        self._background_on = False

        # Relocating / pick-up issues
        self._relocating_neighbors = []
        self._relocating_pickup_records = []
        self._relocating_incorrect_num_method1 = 0
        self._relocating_incorrect_num_method2 = 0
        self._off_background_step = 0
        self._switch_back_durations = []

        # Locomotion
        self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

    def _update_background(self):
        super()._update_background()

        # If the head is not on the furthest position, it will move towards the background pane
        if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
            # Move the head towards the background pane
            self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_timestep.copy()

        # If the head is on the furthest position, i.e., near the background, starts the red color event
        else:
            # Prevent the head from moving further - stop at the furthest position
            if self._data.body(self._head_body_idx).xpos[1] > self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] = self._displacement_upper_bound.copy()

            # Start the red color event
            if self._background_on == False:
                if self._background_trials < self._background_max_trials:
                    self._background_on = True
                    # Set the red color
                    self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                    # De-highlight the previous job - reading
                    self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                    for idx in self._reading_target_idxs:
                        self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                        self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

            # Check when to stop the red color event when the red color is on
            else:
                if self._model.geom(self._background_idx0).rgba[2] >= self._rgba_delta:
                    self._background_trials += 1
                    # Reset the background variables: the color, status flag
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    self._background_on = False
                    # Reset the head and env-cam position
                    self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

                    # Switch back to reading and display distractions
                    self._find_neighbors()  # TODO baseline vs issues on relocating
                    # Reading without distractions
                    # self._model.geom(self._reading_target_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()
                    # self._model.geom(self._reading_target_idx).size[0:3] = self._HINT_SIZE.copy()
                    # self._relocating_neighbors = [self._reading_target_idx]

                    # Switch back off background timestamp
                    if self._off_background_step == 0:
                        # Only update when last pick up happened and this time stamp was reset to 0
                        self._off_background_step = self._steps

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):

        target_grid_idx = self._reading_target_idx
        target_xpos = self._data.geom(target_grid_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - target_xpos)
            if dist <= self._relocating_dist_neighbor:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()  # Continue to avoid infinite loop
                self._model.geom(grid_idx).size[0:3] = self._HINT_SIZE.copy()

        self._relocating_neighbors = neighbors.copy()

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes, non-trainings
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Specify the targets on different conditions
        if self._background_on == True:
            target_idx = self._background_idx0
            change = self._background_rgba_change_per_step
        else:
            target_idx = self._reading_target_idx
            change = self._reading_rgb_change_per_step

        # Focus on targets detection
        if geomid == target_idx:
            reward = 1
            # Update the environment
            self._model.geom(geomid).rgba[0:3] = [x + y for x, y in
                                                  zip(self._model.geom(geomid).rgba[0:3], [0, 0, change])]
            # Do a forward so everything will be set
            mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len:
            terminate = True
        else:
            terminate = False

            # Check the relocating / pick-up status when the background event is off - version 0328
            if self._background_on == False:
                # Check the relocating / pick-up issues - version 0328
                if geomid in self._relocating_neighbors:
                    self._relocating_pickup_records.append(geomid)
                    # Relocating detection - version 2
                    if len(self._relocating_pickup_records) >= self._relocating_pickup_dwell_steps:
                        # Update the switch back duration
                        current_step = self._steps
                        switch_back_duration = current_step - self._off_background_step
                        self._switch_back_durations.append(switch_back_duration)

                        # Determine the pick-up grid by its dwell time
                        counts = Counter(self._relocating_pickup_records)
                        # Find the element(s) with the highest count
                        highest_count = max(counts.values())
                        most_common = [k for k, v in counts.items() if v == highest_count]
                        # If there is only one element with the highest count, return it
                        if len(most_common) == 1:
                            relocate_idx = most_common[0]
                        # If there are multiple elements with the highest count, choose one randomly
                        else:
                            relocate_idx = np.random.choice(most_common)

                        # Update the switch back error rate
                        # Method 1 - exact pick up grid
                        if relocate_idx != self._reading_target_idx:
                            self._relocating_incorrect_num_method1 += 1
                        # Method 2 - ratio of incorrect pick up grids over total pick up grids
                        num_incorrect_relocations = len(
                            self._relocating_pickup_records) - self._relocating_pickup_records.count(
                            self._reading_target_idx)
                        self._relocating_incorrect_num_method2 += (
                                    num_incorrect_relocations / len(self._relocating_pickup_records))

                        # Draw all distractions back to normal - version continue with the current grid
                        for _idx in self._relocating_neighbors:
                            if _idx != relocate_idx:
                                self._model.geom(_idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                                self._model.geom(_idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
                        # Refresh the current target grid and the sequence results idxs if idx is not the previous target
                        if relocate_idx != self._reading_target_idx:
                            self._reading_target_idx = relocate_idx.copy()

                        # Clear the buffer
                        self._relocating_pickup_records = []
                        self._relocating_neighbors = []
                        self._off_background_step = 0

            # Check at the start of the next background event, whether the pervious relocating trial was successful
            else:
                # The relocating did not pick up anything at the previous trial
                if self._off_background_step != 0:
                    # self._relocating_incorrect_num += 1
                    # self._switch_back_durations.append(0)  # TODO handle this later
                    print(
                        'One relocating trial was unable to pick up any grid at the target grid {}. '
                        'The relocating pickup records is: {}. '
                        'The switch back duration is: {}. '
                        'The background trial number is: {}.'
                            .format(self._reading_target_idx, 0,
                                    self._relocating_pickup_records, self._background_trials)
                    )
                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

            # Check whether the grid (background or reading) has been fixated for enough time
            if self._model.geom(self._reading_target_idx).rgba[2] >= self._rgba_delta:

                # Update the number of read grids
                self._num_read_grids += 1

                # Update the intervened relocating where relocating dwell was smaller than the remaining reading time
                if not self._relocating_neighbors:
                    pass
                else:  # When the relocating_neighbor is not empty - the relocation was not over that the relocating dwell was bigger than the remaining reading time
                    # Add new switch back durations
                    current_step = self._steps
                    switch_back_duration = current_step - self._off_background_step
                    self._switch_back_durations.append(switch_back_duration)

                    # Clear the buffer
                    self._relocating_pickup_records = []
                    self._relocating_neighbors = []
                    self._off_background_step = 0

                    print(
                        'One relocating trial was finished earlier at target grid {}. The switch back duration is: {}. '
                        'The background trial is: {}'.
                            format(self._reading_target_idx, switch_back_duration, self._background_trials)
                    )

                # Terminate the loop if all moving background trials are done
                if self._background_trials >= self._background_max_trials:
                    terminate = True
                else:
                    # Traverse all girds idx in the target sequence, if reaches the end, start from the beginning
                    if self._reading_target_idx >= self._reading_target_idxs[-1]:
                        self._reading_target_idx = self._reading_target_idxs[0] - 1
                    self._switch_target(idx=self._reading_target_idx + 1)

            if terminate:
                print('The total timesteps is: {}. \n'
                      'The switch back duration is: {}. The durations are: {} \n'
                      'The reading goodput is: {} (grids per timestep). \n'
                      'The switch back error rate (method 1) is: {}% \n'
                      'The switch back error rate (method 2) is: {}%.'.
                      format(self._steps, np.sum(self._switch_back_durations),
                             self._switch_back_durations,
                             round(self._num_read_grids / self._steps, 5),
                             round(100 * self._relocating_incorrect_num_method1 / self._background_max_trials, 2),
                             round(100 * self._relocating_incorrect_num_method2 / self._background_max_trials, 2)))

        return self._get_obs(), reward, terminate, {}


class LocomotionRelocationTest(LocomotionBase):

    def __init__(self):
        super().__init__()

        # Initialize the length of the episode
        self._ep_len = 8000

        # Initialize the number of trials
        self._background_max_trials = 8
        self._background_trials = None

        self._steps_on_reading_target = None
        self._steps_on_relocation_target = None
        self._steps_on_background_target = None

        # Define the buffer for storing the number of goodput grids
        self._num_read_grids = None

        # Initialize the relocation relevant variables
        self._neighbor_dist_thres = None
        self._neighbors = None
        self._neighbors_steps = None

        self._relocation_target_idx = None
        self._relocating_dwell_steps_thres = self._reading_target_dwell_timesteps
        self._relocating_limit_steps_thres = self._relocating_dwell_steps_thres

        self._RELOCATION_HINT_RGBA = [1, 1, 0, 0.5]
        self._relocation_rgb_change_per_step = 1 / self._relocating_dwell_steps_thres
        self._relocation_alpha_change_per_step = (1 - self._RELOCATION_HINT_RGBA[
            3]) / self._relocating_dwell_steps_thres

        # Initialize the context switch relevant variables
        self._off_background_step = None
        self._switch_back_durations = None
        self._switch_back_error_rates = None

        # Initialize the task mode and layout name
        self._task_mode = None
        self._layout_name = None

    def _reset_scene(self):
        super()._reset_scene()

        # Reset the permanent and temporary counters
        self._background_trials = 0
        self._num_read_grids = 0
        self._switch_back_durations = []
        self._switch_back_error_rates = []

        self._steps_on_reading_target = 0
        self._steps_on_relocation_target = 0
        self._steps_on_background_target = 0
        self._off_background_step = 0
        self._neighbors, self._neighbors_steps = [], []
        self._relocation_target_idx = -2

        # Define the target reading layouts, randomly choose one list to copy from self._ils100_reading_target_idxs, self._bc_reading_target_idxs, self._mr_reading_target_idxs
        self._layout_name = self._config['rl']['test']['layout_name']
        if self._layout_name == 'interline-spacing-100':
            self._reading_target_idxs = self._ils100_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101
        elif self._layout_name == 'bottom-center':
            self._reading_target_idxs = self._bc_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0121
        elif self._layout_name == 'middle-right':
            self._reading_target_idxs = self._mr_reading_target_idxs.copy()
            self._neighbor_dist_thres = 0.0101
        else:
            raise ValueError('Invalid layout name.')

        # Reset the smart glass pane scene and variables
        for idx in self._reading_target_idxs:
            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
            self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()
            # Set everything so the we can find neighbors if needed
            mujoco.mj_forward(self._model, self._data)

        # Reading grids
        self._reading_target_idxs = self._reading_target_idxs.tolist()
        self._switch_target(idx=self._reading_target_idxs[0])
        self._task_mode = READING_MODE

        # Locomotion
        self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

    def _update_background(self):
        super()._update_background()

        # Locomotion
        # If the head is not on the furthest position, it will move towards the background pane
        if self._data.body(self._head_body_idx).xpos[1] < self._furthest_head_xpos_y:
            # Move the head towards the background pane
            self._data.qpos[self._head_joint_y_idx] += self._head_disp_per_timestep.copy()

        # If the head is on the furthest position, i.e., near the background, starts the red color event
        else:
            # Prevent the head from moving further - stop at the furthest position
            if self._data.body(self._head_body_idx).xpos[1] > self._furthest_head_xpos_y:
                self._data.qpos[self._head_joint_y_idx] = self._displacement_upper_bound.copy()

            # Start the red color event
            if self._task_mode != BACKGROUND_MODE:
                self._task_mode = BACKGROUND_MODE
                # Set the red color
                self._model.geom(self._background_idx0).rgba[0:4] = self._EVENT_RGBA.copy()
                # De-highlight the previous job - reading
                self._RUNTIME_TEXT_RGBA = self._model.geom(self._reading_target_idx).rgba[0:4].copy()
                for idx in self._reading_target_idxs:
                    self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
                    self._model.geom(idx).size[0:3] = self._DEFAULT_TEXT_SIZE.copy()

        mujoco.mj_forward(self._model, self._data)

    def _find_neighbors(self):
        # Find the neighbors of the center grid
        center_xpos = self._data.geom(self._reading_target_idx).xpos

        neighbors = []

        for grid_idx in self._reading_target_idxs:
            grid_xpos = self._data.geom(grid_idx).xpos
            dist = np.linalg.norm(grid_xpos - center_xpos)
            if dist <= self._neighbor_dist_thres:
                neighbors.append(grid_idx)
                self._model.geom(grid_idx).rgba[0:4] = self._RELOCATION_HINT_RGBA.copy()

        # Randomly choose one grid in the neighbors list to be the target
        self._relocation_target_idx = self._reading_target_idx

        # Update the neighbors list
        self._neighbors = neighbors.copy()

        # Initialize the steps on each neighbor
        self._neighbors_steps = [0] * len(neighbors)

    def step(self, action):
        super().step(action)

        # Normalise action from [-1, 1] to actuator control range
        action[0] = self.normalise(action[0], -1, 1, *self._model.actuator_ctrlrange[0, :])
        action[1] = self.normalise(action[1], -1, 1, *self._model.actuator_ctrlrange[1, :])

        # Set motor control
        self._data.ctrl[:] = action

        # Update the background changes, non-trainings
        self._update_background()

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        self._steps += 1

        # Eye-sight detection
        dist, geomid = self._ray_from_site(site_name="rangefinder-site")

        # Estimate reward for each step
        reward = 0

        # Check the tasks and update accordingly
        if self._task_mode == READING_MODE:
            if geomid == self._reading_target_idx:
                # Sparse reward
                reward = 1
                # Update the reading target counter
                self._steps_on_reading_target += 1
                # Update the reading target color
                self._model.geom(geomid).rgba[2] += self._reading_rgb_change_per_step
                # Check the termination condition
                if self._steps_on_reading_target >= self._reading_target_dwell_timesteps:
                    # Update the number of reading target
                    self._num_read_grids += 1

                    # Reset the grid color
                    if self._reading_target_idx >= self._reading_target_idxs[-1]:
                        self._reading_target_idx = self._reading_target_idxs[0] - 1
                    self._switch_target(idx=self._reading_target_idx + 1)
                    self._task_mode = READING_MODE

                    # Reset the reading target counter
                    self._steps_on_reading_target = 0

        elif self._task_mode == BACKGROUND_MODE:
            if geomid == self._background_idx0:
                # Sparse reward
                reward = 1
                # Update the background target counter
                self._steps_on_background_target += 1
                # Update the background target color
                self._model.geom(geomid).rgba[2] += self._background_rgba_change_per_step
                # Check the termination condition
                if self._steps_on_background_target >= self._background_dwell_timesteps:
                    # Update the number of background trials
                    self._background_trials += 1
                    # Update the off background step
                    self._off_background_step = self._steps

                    # Reset the background target counter
                    self._steps_on_background_target = 0
                    # Reset the background variables: the color, status flag, the counter, and the head position
                    self._model.geom(self._background_idx0).rgba[0:4] = self._DEFAULT_BACKGROUND_RGBA.copy()
                    # Reset the locomotion position
                    self._data.qpos[self._head_joint_y_idx] = self._displacement_lower_bound.copy()

                    # Jump into the relocation task
                    self._find_neighbors()
                    self._task_mode = RELOCATING_MODE

        elif self._task_mode == RELOCATING_MODE:
            if geomid in self._neighbors:
                # Update the steps on target
                self._neighbors_steps[self._neighbors.index(geomid)] += 1

                if geomid == self._relocation_target_idx:
                    # Sparse reward
                    reward = 1
                    # Update the relocation target counter
                    self._steps_on_relocation_target += 1
                    # Update the relocation target color
                    self._model.geom(geomid).rgba[3] += self._relocation_alpha_change_per_step
                else:
                    # Update the distractions - becomes dimmer
                    self._model.geom(geomid).rgba[2] += self._relocation_rgb_change_per_step

                # Check the termination condition
                if np.sum(self._neighbors_steps) >= self._relocating_limit_steps_thres:
                    # Update the switch back duration
                    current_step = self._steps
                    switch_back_duration = current_step - self._off_background_step
                    self._switch_back_durations.append(switch_back_duration)

                    # Update the switch back error rate
                    # Get the total number of steps in self._neighbors_steps that does not corresponding to the relocation target
                    num_switch_back_errors = np.sum(self._neighbors_steps) - self._neighbors_steps[self._neighbors.index(self._relocation_target_idx)]
                    # Get the error rate
                    switch_back_error_rate = num_switch_back_errors / np.sum(self._neighbors_steps)
                    # Update the switch back error rate list
                    self._switch_back_error_rates.append(switch_back_error_rate)

                    # Reset the relocation target counter
                    self._steps_on_relocation_target = 0

                    # Get back to the reading mode
                    self._task_mode = READING_MODE
                    # Reset the grid color
                    for idx in self._reading_target_idxs:
                        if idx == self._reading_target_idx:
                            self._model.geom(self._relocation_target_idx).rgba[0:4] = self._RUNTIME_TEXT_RGBA.copy()

                        else:
                            self._model.geom(idx).rgba[0:4] = self._DEFAULT_TEXT_RGBA.copy()
        else:
            raise ValueError(f'Unknown task mode: {self._task_mode}')

        mujoco.mj_forward(self._model, self._data)

        # Check termination conditions
        if self._steps >= self._ep_len or self._background_trials > self._background_max_trials:
            terminate = True
        else:
            terminate = False

        if terminate:
            print(f'The total timesteps is: {self._steps}. \n'
                  f'The switch back duration is: {np.sum(self._switch_back_durations)}. The durations are: {self._switch_back_durations} \n'
                  f'The reading goodput is: {round(self._num_read_grids / self._steps, 5)} (grids per timestep). \n'
                  f'The switch back error rate (method 1) is: {round(100 * np.mean(self._switch_back_error_rates), 2)} % \n')

        return self._get_obs(), reward, terminate, {}

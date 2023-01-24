import cv2
import numpy as np
import os
import math
from typing import Callable, NamedTuple, Optional, Union, List
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # This package.class could be used to add XML elements dynamically.

import mujoco
import yaml
from mujoco.glfw import glfw

import gym
from gym import Env
from gym import utils
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from torch import nn
import torch

from Task import Task


class GlassSwitch(Env):

    def _config(self, configs):
        """
        This method configures all static settings from the YAML file, including flags and counters.
        The 'static' here means configurations that will not change in the runtime.
        And this is the only changable part of the simulation by the user.

        Args:
            configs: the configuration content from YAML file.
        """
        # -------------------------------------------------------------------------------------------------------------
        # Config the mujoco environment.
        # Get strings. Get the names that were declared in XML. # TODO might use the etree to programmatically access to the string names.
        # The fixed abstract camera name:
        self._camera_name = 'fixed-eye'
        # The geom names.
        self._geom_name_ambient_env = 'ambient-env'
        self._geom_name_smart_glass_lenses = 'smart-glass-lenses'
        self._geom_names_glass_display_ids = {
            0: 'smart-glass-display-0',
            1: 'smart-glass-display-1',
            2: 'smart-glass-display-2'
        }

        # Config the OpenGL rendering related configurations.
        self._configs = configs  # TODO for debugging, refine later.
        self._config_mj_env = configs['mujoco_env']
        # Resolution in pixels: width * height
        # Reference: glfw.GLFWwindow * glfwCreateWindow
        # https://www.glfw.org/docs/3.3/group__window.html#ga3555a418df92ad53f917597fe2f64aeb
        # TODO might need to scale the resolution down to increment the computing and rendering speed.
        self._width = self._config_mj_env['render']['width']
        self._height = self._config_mj_env['render']['height']

        # Config Flags.
        # Internal buffer configuration.
        self._is_rgb = self._config_mj_env['render']['rgb']  # Set to True to read rgb.
        self._is_depth = self._config_mj_env['render']['depth']  # Set to True to read depth.
        # The camera to be fixed or not.
        self._is_camera_fixed = False  # self._config_mj_env['render']['is_camera_fixed']   # Set to True to disable the camera moving functions.
        # The display window, rendering buffer, and the operating system vs. OpenGL.
        # Ref https://mujoco.readthedocs.io/en/latest/programming/visualization.html#buffers-for-rendering
        # From the perspective of OpenGL, there are important differences between the window framebuffer and offscreen framebuffer,
        #  and these differences affect how the MuJoCo user interacts with the renderer.
        #  The window framebuffer is created and managed by the operating system and not by OpenGL.
        #  As a result, properties such as resolution, double-buffering, quad-buffered stereo, mutli-samples, v-sync
        #  are set outside OpenGL; this is done by GLFW calls in our code samples.
        #  All OpenGL can do is detect these properties; we do this in mjr_makeContext and record the results in the
        #  various window capabilities fields of mjrContext. This is why such properties are not part of the MuJoCo model;
        #  they are session/software-specific and not model-specific. In contrast, the offscreen framebuffer is managed
        #  entirely by OpenGL, and so we can create that buffer with whatever properties we want,
        #  namely with the resolution and multi-sample properties specified in mjModel.vis.
        self._is_window_visible = self._config_mj_env['render'][
            'is_window_visible']  # Set to 1 to enable the window being visible. 0 to hide the window.
        if self._is_window_visible == 1:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_WINDOW.value
            # The key and mouse interaction to be enabled or not.
            self._is_key_mouse_interaction = self._config_mj_env['utils']['is_key_mouse_interaction']
        else:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value
            # The keyboard interaction setting.
            self._is_key_mouse_interaction = False
        # Raise some value errors when the settings are having conflicts.
        if self._is_key_mouse_interaction and (self._is_camera_fixed or self._is_window_visible == 0):
            raise ValueError(
                'Invalid keyboard and mouse settings: a visible window and a free camera are both necessary.')
        # The printings.
        self._is_print_cam_config = self._config_mj_env['utils'][
            'is_print_cam_config']  # Set to True to print camera configuration.
        self._is_print_cam_rgb_depth = self._config_mj_env['utils'][
            'is_print_cam_rgb_depth']  # Set to True to print camera read pixel rgb and depth info.

        # Config constants.
        # The display modes. Must select from the predefined ones.
        self._viewer_modes = ['through_glass', 'overhead']
        self._viewer_mode = self._config_mj_env['render']['viewer_mode']
        if self._viewer_mode not in self._viewer_modes:
            raise ValueError(
                'Invalid viewer mode: it must be selected from the predefined options [through_glass, overhead].')
        # The task modes. Must select from the predefined modes.
        # 'demo' corresponds to the demonstration, a visible window is required.
        # The others are tasks used in RL, they can be operated in an off-screen manner.
        self._task_modes = ['demo', 'color_switch']
        self._task_mode = self._config_mj_env['render']['task_mode']
        if self._task_mode not in self._task_modes:
            raise ValueError('Invalid task mode: it must be selected from the predefined options [demo, color_switch].')
        if self._task_mode == self._task_modes[0] and (self._is_window_visible == 0 or self._is_camera_fixed is True):
            raise ValueError(
                'Invalid configuration: the window must be visible and the camera must be free in the demo task mode.')

        # -------------------------------------------------------------------------------------------------------------
        # Config the RL pipeline related stuff.
        # The length of the episode. The timesteps that it takes.
        self._num_steps = configs['rl_pipeline']['num_steps']

        # -------------------------------------------------------------------------------------------------------------
        # Config the task game related stuff.
        # Create an instance of task.
        # self._task = Task(configs=configs)        # TODO separate this later.

        # Focal point moving distance configurations.
        self._demo_dist_configs = {
            'min_dist': 0,
            'max_dist': 1,  # TODO This could be a tunable parameter in the design space.
        }
        self._demo_mapping_range = self._demo_dist_configs['max_dist'] - self._demo_dist_configs['min_dist']
        # The task environment settings configuration.
        self._task_scripts = configs['task_spec']['scripts']
        offset = 2  # TODO normalize this later.
        self._task_scripts['x_sample'] = int(self._height / 2 - offset)
        self._task_scripts['y_sample'] = int(self._width / 2 - offset)

        # The demo mode's simulation time.
        self._demo_sim_time = configs['task_spec']['demo_sim_time']

    def _init_rl_data(self):
        """
        This method initializes all the reinforcement related data, i.e., parameters.
        """
        # ------------------------------------------------------------------------------------------------------------
        # The agent's internal state: a finite status machine. Will be used on evaluating results TODO to be updated.
        self._states = {
            'optimal_score': 0,
            'total_time_on_glass_B': 0,
            'total_time_on_env_red': 0,
            'total_time_on_glass_X': 0,
            'total_time_miss_glass_B': 0,
            'total_time_miss_env_red': 0,
            'total_time_miss_glass_X': 0,
            'total_time_glass_B': 0,
            'total_time_glass_X': 0,
            'total_time_env_red': 0,
            'total_time_intermediate': 0,
            'num_on_glass': 0,
            'num_on_env': 0,
            'current_on_level': 0,
            # 0 for getting nothing, 1 for X, 2 for red, 3 for B, -1 for losing X, -2 for red, -3 for
        }

        # ------------------------------------------------------------------------------------------------------------
        # The RL task related stuff, reset the task states.
        # self._task.reset()        # TODO enable this later.
        # The task's finite state machine.
        self._task_states = {  # task finite state machine
            'current_glass_display_id': 0,
            'current_env_color_id': 0,
            'start_step_glass_display_timestamp': 0,  # the current glass display started frame's timestamp, in seconds
            'start_step_env_color_timestamp': 0,  # the current env class started frame's timestamp, in seconds
            'previous_glass_display_id': 0,
            'previous_env_color_id': 0,
            'previous_step_timestamp': 0,
        }

        # The global simulation step.
        self._steps = 0

        # The rgb images that will be used in writing to a video.
        self._rgb_images = []

    def _init_cam_data(self):
        """
        This method is developed for making parameters initialization replicable.
         1. The GUI interactions, including button events and mouse cursor positions.
         2. The camera configurations / input conditions. To be noted that cam_azimuth, cam_elevation, cam_distance, and cam_lookat
            can decide where and how much the camera can see.
         3. What the camera can see are encoded as rgb and depth, and the current frame's information are stored
            in the rgb and depth buffer.
        """

        # GUI interactions.
        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_x = 0
        self._last_y = 0

        # Initialize the abstract camera's pose specifications.
        # Variability: free or fixed.
        self._set_cam_free()
        # mjvCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        # Set the camera position and pose according to the environment setting, e.g., the planes' positions.
        if self._viewer_mode == self._viewer_modes[0]:  # through the glass
            self._static_cam_height = 5.5
            self._initial_cam_pos_y = -7.5
            pass
        elif self._viewer_mode == self._viewer_modes[1]:  # overhead
            # TODO firstly I just do a 1-D change on the y axis. Complete later.
            self._static_cam_height = 5.5
            self._initial_cam_pos_y = -7.5

        self._geom_pos_y_smart_glass_lenses = self._model.geom(self._geom_name_smart_glass_lenses).pos[1]
        self._geom_pos_y_ambient_env = self._model.geom(self._geom_name_ambient_env).pos[1]

        self._init_cam_pose = {
            'cam_lookat': np.array([0, self._geom_pos_y_smart_glass_lenses, 1.5]),  # lookat point
            'cam_distance': 5,  # Distance to lookat point or tracked body.
            'cam_azimuth': 90.0,
            # Camera azimuth (deg). The concept reference: https://www.photopills.com/articles/understanding-azimuth-and-elevation
            'cam_elevation': 0.0  # Camera elevation (deg).
        }

        # Assign the initial values.
        self._cam.lookat = self._init_cam_pose['cam_lookat']
        self._cam.distance = self._init_cam_pose['cam_distance']
        self._cam.azimuth = self._init_cam_pose['cam_azimuth']
        self._cam.elevation = self._init_cam_pose['cam_elevation']

        # Internal buffers: rbg buffer and depth buffer.
        self._rgb_buffer = np.empty((self._height, self._width, 3),
                                    dtype=np.uint8) if self._is_rgb else None
        self._depth_buffer = np.empty((self._height, self._width),
                                      dtype=np.float32) if self._is_depth else None

    def _set_cam_free(self):
        """
        This method sets the camera's variability: free or fixed, according to the configuration.
        """
        # Identify whether to make the abstract camera's position fixed.
        # Reference: Aleksi's rendering.py, class Camera. https://github.com/BaiYunpeng1949/user-in-the-box/blob/main/uitb/utils/rendering.py
        # Reference: mjvCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        # Reference: mjtCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjtcamera
        # Get the camera ID. camid=0; ncam=1
        if self._is_camera_fixed:
            if isinstance(self._camera_name, str):
                self._camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
            if self._camera_id < -1:
                raise ValueError(
                    'camera_id cannot be smaller than -1. The abstract camera must be specified in the XML model.')
            if self._camera_id >= self._model.ncam:
                raise ValueError(
                    'model has {} fixed cameras. camera_id={} is invalid.'.format(self._model.ncam, self._camera_id))
            # Fix the camera.
            self._cam.fixedcamid = self._camera_id
            if self._camera_id == -1:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
                # camera explicitly defined in the model.
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def _set_interactions(self):
        """
        This method installs the keyboard and mouse actions by setting up the correspoding callback functions
        in the glfw.
        """
        if self._is_key_mouse_interaction:
            glfw.set_key_callback(self._window, self._keyboard)
            glfw.set_cursor_pos_callback(self._window, self._mouse_move)
            glfw.set_mouse_button_callback(self._window, self._mouse_button)
            glfw.set_scroll_callback(self._window, self._scroll)

    def __init__(self):
        """
        This class defines the mujoco environment where simulates human's visual perception behaviors when
        they are shifting/switching focal point between the smart glasses and the environment behind it.
        The interest of this project starts from investigating using RL to evaluate information loss, and
        other effects available.

        I implemented mujoco abstract camera to simulate human's visual perceptions,
        I prioritize from the Camera sensor developed by Aleksk and MuJoCoPy Bootcamp Lec 13:
        https://pab47.github.io/mujocopy.html

        Concepts:
            Window: a window is a top-level graphical object that represents a window on the screen.
                It is created using the glfwCreateWindow function, and it can be shown, hidden, resized,
                and moved by calling various GLFW functions.
            Scene: a scene is a collection of 3D objects that are rendered in a window.
                It is typically created by defining a set of geometric models, lighting, and materials,
                and rendering these objects in a loop using OpenGL commands.
            Viewport: the viewport is a rectangular region of a window where the 3D scene is drawn.
                It is defined by the x, y, width, and height arguments to the glViewport function,
                which specify the position and size of the viewport within the window.
                The viewport transformation maps the 3D coordinates of the scene to the 2D coordinates of the window.
                It is defined by the projection matrix and the view matrix,
                which specify the position and orientation of the camera in the 3D scene.
                The viewport is important because it determines how the 3D scene is projected onto the 2D window.
        """
        # Initialize the configurations, including camera settings.
        # Ref: MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html
        # Read the configurations from the YAML file.
        with open('config.yaml') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self._config(configs=configs)
        # --------------------------------------- RL initialization -------------------------------------------------------------
        # Load the xml MjModel.
        self._model = mujoco.MjModel.from_xml_path(self._config_mj_env['model_path'])

        # Initialize MjData.
        self._data = mujoco.MjData(self._model)

        # Initialize necessary properties for RL: action_space.
        # The action will be either 1 or 0.    # TODO the simple version: look at the glass or environment.
        #  0 for looking at the smart-glass and 1 for looking at the ambient-env.
        self.action_space = Discrete(2)

        # The observation_space - the simple version: the perception pixels. TODO try only a small central portion of the pixels.
        offset = int(self._width / 4)  # TODO normalize it later
        self._obs_idx_h = [int(self._height / 2 - offset), int(self._height / 2 + offset)]
        self._obs_idx_w = [int(self._width / 2 - offset), int(self._width / 2 + offset)]
        self.observation_space = Box(low=np.uint8(0), high=np.uint8(255), shape=(offset * 2, offset * 2, 3))
        # self.observation_space = Dict({
        #     'rgb': Box(low=np.uint8(0), high=np.uint8(255), shape=(offset * 2, offset * 2, 3))
        # })
        # self.observation_space = Dict({
        #     'rgb': Box(low=np.uint8(0), high=np.uint8(255), shape=(self._height, self._width, 3))   # TODO the rgb was between 0-255
        # })
        # TODO debug delete later
        # print('dududu', self.observation_space['rgb'].sample())
        # print('The observation space sample looks like: {}, the shape is: {}'.format(self.observation_space.sample(), self.observation_space['rgb'].shape))

        # Initiate the rl related data, i.e., parameters.
        self._init_rl_data()
        # --------------------------------------- Visual perception camera and rendering initialization -------------------------------------------------------------
        # TODO the rendering structure can be further enhanced following the
        #  dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/viewer/renderer.py
        # Initializations.
        # The abstract camera.
        self._cam = mujoco.MjvCamera()  # Abstract camera.
        # Concepts: see the
        # "OpenGL® camera" is the name given to the virtual position of a viewer within an Open Graphics Library®
        #  (OpenGL®) scene. It is defined by the position of the viewer within the scene, and then the location or
        #  direction in which the viewer is looking. The position of the camera in an OpenGL® scene will determine
        #  what portion of a scene will be rendered to the display device and at what angle.
        #  Reference: https://www.easytechjunkie.com/what-is-an-opengl-camera.htm#:~:text=%22OpenGL%C2%AE%20camera%22%20is%20the,which%20the%20viewer%20is%20looking.
        # Set the default settings.
        mujoco.mjv_defaultCamera(self._cam)
        # Update the dynamic parameters.
        self._init_cam_data()

        # Initialize the options.
        self._opt = mujoco.MjvOption()  # Visualization options.
        mujoco.mjv_defaultOption(self._opt)

        # Initialize the OpenGL and rendering functionality.
        # Init GLFW, create window, make OpenGL context current, request v-sync.
        #  This process has been encapsulated in the glfw.GLContex, I configure it explicitly here for controlling
        #  the visibility of the OpenGL window.
        # Reference: https://mujoco.readthedocs.io/en/latest/programming/visualization.html#context-and-gpu-resources
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, self._is_window_visible)
        # Create an OpenGL context using GLFW.
        self._window = glfw.create_window(width=self._width, height=self._height, title="Bai Yunpeng's Window",
                                          monitor=None, share=None)
        glfw.make_context_current(self._window)
        # An OpenGL context is what enables the application to talk to the video driver and send rendering commands.
        #  It must exist and must be current in the calling thread before mjr_makeContext is called.
        #  GLFW and related libraries provide the necessary functions as shown above.
        # glfw.swap_interval(1)
        # This function sets the swap interval for the current OpenGL or OpenGL ES context, i.e. the number of screen
        # updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning.
        # This is sometimes called vertical synchronization, vertical retrace synchronization or just vsync.
        # Parameter interval: The minimum number of screen updates to wait for until the buffers are swapped by glfwSwapBuffers.
        # Ref: https://www.glfw.org/docs/3.3/group__context.html#ga6d4e0cdf151b5e579bd67f13202994ed

        # Initialize visualization data structures.
        self._scene = mujoco.MjvScene(self._model, maxgeom=10000)
        # TODO in the headless rendering, try offwidth and offheight. And disable the viewport get windows resolution.
        #  Ref: https://github.com/deepmind/mujoco/issues/517#issuecomment-1272187780
        # Make the model-specific mjrContext. Ref https://mujoco.readthedocs.io/en/latest/programming/visualization.html#opengl-rendering
        # MjrContext contains the function mjr_makeContext in the light of Cpp.
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # After creation, it contains references (called “names” in OpenGL) to all the resources
        #  that were uploaded to the GPU by mjr_makeContext.
        # Set the context's frame buffer type: a visible window buffer or an offscreen buffer.
        mujoco.mjr_setBuffer(self._framebuffer_type, self._context)

        # Viewport.
        self._viewport = mujoco.MjrRect(0, 0, self._width, self._height)

        # --------------------------------------- Utilities -------------------------------------------------------------
        # Initialize the mouse and keyboard interactions if permitted, these are also related to rendering.
        self._set_interactions()

        # --------------------------------------- Confirmation -------------------------------------------------------------
        # Finally confirm all the settings.
        mujoco.mj_forward(m=self._model, d=self._data)

    def reset(self):
        """
        The override method: reset all data.

        Returns:
            obs: the reset observations.
        """
        # Reset the simulated environment. TODO check whether the dynamic values are changed to their defaults.
        mujoco.mj_resetData(m=self._model, d=self._data)

        # Reset the abstract camera related data.
        self._init_cam_data()

        # Reset the rl related data, including the global simulation steps.
        self._init_rl_data()

        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        The override method: generates a specific step.

        Args:
            action: one sample from the action space.

        Returns:
            obs: observations.
            reward: reward for the current step.
            done: the flag shows whether the simulation is finished.
            info: the detailed information.
        """
        # Advance the simulation
        mujoco.mj_step(m=self._model, d=self._data, nstep=1)  # nstep=self._run_parameters["frame_skip"]

        # Update environment.
        obs, reward, done, info = self._update(action=action)  # TODO research on the pipeline and observation ~ states.

        # Update the simulated steps.
        self._steps += 1

        return obs, reward, done, info

    def _update(self, action):
        """
        The hierarchy game:
         B letter - higher priority: stay on the glass until the content changed.
          Only when the focal point is on the glass that the agent knows the content.
          The rewards are proportional to the time it stays on the smart glass.
         Red color - lower priority: stay on the env until the color changed.
          The rewards are proportional to the time it stays on the wall.
          The tricky part is: letting the agent learn, when he is looking at the wall, it must go back and check the smart
          glass content. What would be the best focal change frequency.
         Specifically, the action space will be a binary decision space: look at the glass or look at the environment.
          The observation space would be what the agent can see: the pixels with only rgb values captured by the abstract camera.
          And to simplify the task, only when the user is looking at the environment, the color will be disclosed. Or it will be the default grey.

        Args:
            action: action: one sample from the action space.

        Returns:
            obs: observations.
            reward: reward for the current step.
            done: the flag shows whether the simulation is finished.
            info: the detailed information.
        """

        # Do the task.
        obs, done = self._do_task(action=action)

        # Update the info. TODO current version: direct ask from states. Add more information about actual game states, such as env_red, env_green later.
        info = self._states

        # Calculate the reward
        reward = self._reward_function()

        return obs, reward, done, info

    def _get_obs(self):
        """
        Get the observations.

        Returns:
            obs: observations.
        """
        rgb = self._rgb_buffer.copy()[self._obs_idx_h[0]:self._obs_idx_h[1], self._obs_idx_w[0]:self._obs_idx_w[1], :]
        # Returns the read rgb pixels.
        # obs = {  # TODO the partial pixels - make it as an API
        #     'rgb': rgb
        #     # 'rgb': self._rgb_buffer.copy()
        # }
        obs = rgb
        # TODO delete later
        # print('The sample obs: {}, the shape is: {}'.format(obs, obs_rgb.shape))
        return obs

    def _reward_function(self):
        """
        Defines the reward function and gets the reward value.

        Returns:
            reward: the reward value for the current step.
        """
        # TODO outline
        #  1. considering using the sum-up manner, i.e., the environment and task has their own reward_functions,
        #   the sum of the scores will be optimized.
        #  2. the reward_functions should be high-level and generic enough to utilize the advantages of RL.
        #   Or there is no difference to using rule-based/hard coded algorithm.

        # Write the reward function based on the agent's states.
        states = self._states.copy()
        task_states = self._task_states.copy()

        # TODO unbalance the task by tuning the rewards.
        # Get the rewards. Write in this awkward form because it is easy to tune them.
        if states['current_on_level'] == 3:
            reward = 5
        elif states['current_on_level'] == 2:
            reward = 3
        elif states['current_on_level'] == 1:
            reward = 1
        # TODO -rewards for spending time or useless switches,
        #  which might enhance the too frequent changing behavior.
        elif states['current_on_level'] == -1:
            reward = -1
        elif states['current_on_level'] == -2:
            reward = -3
        elif states['current_on_level'] == -3:
            reward = -5
        else:
            reward = 0

        # Calculate the optimal rewards.
        if task_states['current_glass_display_id'] == 1:
            self._states['optimal_score'] += 5
        elif task_states['current_glass_display_id'] == 2:
            if task_states['current_env_color_id'] != 1:
                self._states['optimal_score'] += 1
            else:
                self._states['optimal_score'] += 3
        else:
            if task_states['current_env_color_id'] == 1:
                self._states['optimal_score'] += 3
            else:
                self._states['optimal_score'] += 0

        return reward

    def render(self, mode="human"):
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        Here I referenced from MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html,
        to build my camera and GL rendering.
        """
        # Render the demo in the 'demo' task mode.
        if self._task_mode == self._task_modes[0]:
            while not glfw.window_should_close(
                    self._window):  # Decides when the window will be closed, such as a mouse click event.
                time_prev = self._data.time

                # Skip some frames, restrict the render fps to be around 60Hz.
                while self._data.time - time_prev < 1.0 / 60.0:
                    mujoco.mj_step(self._model, self._data)
                # The smoother and natural render update.
                # mujoco.mj_step(self._model, self._data)

                if self._data.time >= self._demo_sim_time:
                    break

                # Do the given task.
                self._do_task(action=None)

                # Print logs.
                self._print_logs()
        # The task mode is 'color_switch'.
        elif self._task_mode == self._task_modes[1]:
            # TODO do the frame-wise updates in the override step() --> _update().
            pass

    def close(self):
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        It terminates the glfw, end rendering and display.
        """
        # Close the OpenGL context created by the GLFW. Referenced from the GLContext.
        if self._window:
            if glfw.get_current_context() == self._window:
                glfw.make_context_current(None)
            # Destroy the window.
            glfw.destroy_window(self._window)
            self._window = None

        # Terminate the glfw.
        glfw.terminate()

    def _do_task(self, action):
        """
        Important: this method determines how the tasks are performed at each step / frame.
        It specifies how the actions are taken, and how the observation space would be.

        P.S., before I made a separate task class, this method is the temporal place to hold all task logics.
        TODO change this into the task class later.
        Returns:
            obs: observations.
            done: the flag shows whether the simulation is finished.
        """
        if self._task_mode == self._task_modes[0]:  # the demo mode
            const_animation = 0.05
            abs_dx = 0.05
            abs_da = 0.45
            timestep_length = 1  # int(const_animation / abs_dx)
            num_timesteps = 100
            for i in range(num_timesteps):
                if self._data.time <= i * timestep_length:
                    if i % 2 != 0:
                        dx = abs_dx
                        da = abs_da
                    else:
                        dx = - abs_dx
                        da = - abs_da
                    break

            self._cam.lookat[1] += dx

            # Transparency manipulation.
            alpha_change_coefficient = 0.5
            alpha_change_gain = 1.0  # The gain could be used later. In a non-linear control.
            abs_y_dist = abs(self._cam.lookat[1] - self._geom_pos_y_smart_glass_lenses)

            if abs_y_dist >= self._demo_dist_configs['max_dist']:
                alpha = 0
            else:
                alpha = alpha_change_coefficient * (
                            self._demo_dist_configs['max_dist'] - abs_y_dist) / self._demo_mapping_range

            # Update the alpha from rgba's transparency value.
            self._model.geom(self._geom_names_glass_display_ids[1]).rgba[3] = alpha

            # Sync up the visualized focal point - ball's movement.
            self._data.qpos[0:3] = self._cam.lookat
            # print('elevation: {}, qpos: {}'.format(self._cam.elevation, self._data.qpos[0:3]))

            # Get the observations.
            obs = self._get_obs()

            # Update the scene and render everything.
            self._update_scene_render()

            done = False

        elif self._task_mode == self._task_modes[1]:  # the rl task: color switch
            # Update the environment changes according to the task/game rules.  # TODO maybe write into an independent class later.
            # Get the current timestamp and the elapsed time since last step.
            current_step_timestep = self._data.time  # TODO assume all the operations does not take time, improve it later - maybe use the step as the counter instead of the timer.
            elapsed_time = current_step_timestep - self._task_states['previous_step_timestamp']
            # Check if the glass display and env color has run out the duration.    # TODO apply some game settings that are less random and easier to learn.
            # The glass.
            current_glass_display_duration = current_step_timestep - self._task_states[
                'start_step_glass_display_timestamp']
            # If ran out of time, then allocate the new display and color.
            if current_glass_display_duration > self._task_scripts['glass_display_duration']:
                # Make sure the new display is not the same as the previous. TODO more generalized scenario.
                # while True:
                #     current_glass_display_id = Discrete(len(self._task_scripts['glass_display_choices'])).sample()
                #     if current_glass_display_id != self._task_states['previous_glass_display_id']:
                #         break
                #     else:
                #         pass
                # Generate fix-ordered displays. TODO easier scenario - the deterministic task with a fixed order.
                current_glass_display_id = int((self._task_states['current_glass_display_id'] + 1) %
                                               len(self._task_scripts['glass_display_choices']))

                # Update the task state machine.
                self._task_states['start_step_glass_display_timestamp'] = current_step_timestep
                self._task_states['previous_glass_display_id'] = self._task_states['current_glass_display_id']
                self._task_states['current_glass_display_id'] = current_glass_display_id
            # If not, update the time.
            elif current_glass_display_duration <= self._task_scripts['glass_display_duration']:
                # Get the exact glass display time:
                if self._task_states['current_glass_display_id'] == 1:
                    self._states['total_time_glass_B'] += elapsed_time
                elif self._task_states['current_glass_display_id'] == 2:
                    self._states['total_time_glass_X'] += elapsed_time
                else:
                    pass
            # The environment.
            current_env_color_duration = current_step_timestep - self._task_states['start_step_env_color_timestamp']
            if current_env_color_duration > self._task_scripts['env_color_duration']:
                # while True:
                #     current_env_color_id = Discrete(len(self._task_scripts['env_color_choices'])).sample()
                #     if current_env_color_id != self._task_states['previous_env_color_id']:
                #         break
                #     else:
                #         pass

                # A fix-ordered env color display. TODO easier mode
                current_env_color_id = int((self._task_states['current_env_color_id'] + 1) %
                                           len(self._task_scripts['env_color_choices']))

                # Update the task state machine.
                self._task_states['start_step_env_color_timestamp'] = current_step_timestep
                self._task_states['previous_env_color_id'] = self._task_states['current_env_color_id']
                self._task_states['current_env_color_id'] = current_env_color_id
            elif current_env_color_duration <= self._task_scripts['env_color_duration']:
                # Get the exact env color time:
                if self._task_states['current_env_color_id'] == 1:
                    self._states['total_time_env_red'] += elapsed_time
                else:
                    pass

            # Update stuff related to the action.
            # 1. The look at point's y position.
            # 2.And sync up the glass transparency - an easier version of demo. When look at the glass, the content shows.
            #  When look at the env, the content is hided.
            # 3. And the environment color - when look at the environment, the value will be disclosed,
            #  or it would always be the default grey color.
            if action == 0:  # look at the smart glass lenses
                self._cam.lookat[1] = self._geom_pos_y_smart_glass_lenses
                alpha = 0.5
                color_id = 0
                self._states['num_on_glass'] += 1
            elif action == 1:  # look at the env
                self._cam.lookat[1] = self._geom_pos_y_ambient_env
                alpha = 0
                color_id = self._task_states['current_env_color_id']
                self._states['num_on_env'] += 1
            else:
                alpha = 0
            # Sync up the visualized focal point - ball's movement.
            self._data.qpos[0:3] = self._cam.lookat
            # Sync up the correct smart glass's content transparency according to the above update.
            for i in range(len(self._geom_names_glass_display_ids)):
                if i == 0:  # the display 0 is always transparent - display nothing
                    self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = 0
                elif i == self._task_states['current_glass_display_id']:
                    self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = alpha
                else:
                    self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = 0
            # Sync up the correct env color according to the above update.
            self._model.geom(self._geom_name_ambient_env).rgba[0:3] = self._task_scripts['env_color_choices'][color_id]

            # Update the scene and render everything.
            self._update_scene_render()

            # Get the observations.
            obs = self._get_obs()

            # Decide based on the visual perceptions, and update the states.
            # Determine what the camera agent sees.
            sample_point_rgb = self._rgb_buffer[self._task_scripts['x_sample'], self._task_scripts['y_sample'], :]

            # TODO reconstruct this part later. Make it more generalizable and accurate.
            def identify_visual_content(sample_point, comparisons):
                states = comparisons.copy()
                dist_threshold = 6
                content = 'none'

                for comparison in states['on_env_grey']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        if self._task_states['current_glass_display_id'] != 0:
                            # TODO fix this ambiguity later: the overlapped identification for the env_grey and glass_nothing.
                            content = 'on_env_grey'
                        # elif (sample_point == states['on_env_grey'][0]).all():
                        #     content = 'on_env_grey'

                for comparison in states['on_env_green']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        content = 'on_env_green'

                for comparison in states['on_env_blue']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        content = 'on_env_blue'

                for comparison in states['on_glass_nothing']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        if self._task_states['current_env_color_id'] != 0:
                            # TODO fix this ambiguity later: the overlapped identification for the env_grey and glass_nothing.
                            content = 'on_glass_nothing'
                        # elif (sample_point == states['on_glass_nothing'][1]).all():
                        #     content = 'on_glass_nothing'

                for comparison in states['on_glass_X']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        content = 'on_glass_X'

                for comparison in states['on_env_red']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        content = 'on_env_red'

                for comparison in states['on_glass_B']:
                    if np.linalg.norm(sample_point - comparison) <= dist_threshold:
                        content = 'on_glass_B'

                return content

            # Get the current visual content.
            perceived_content = identify_visual_content(sample_point=sample_point_rgb, comparisons=self._task_scripts)

            if self._configs['rl_pipeline']['train'] is None:
                print('action: {}   rgb: {}     glass id: {}    env id: {}      perceived result: {}'
                      .format(action, sample_point_rgb, self._task_states['current_glass_display_id'],
                              self._task_states['current_env_color_id'], perceived_content))
            # TODO debug help promote the decision making section. TODO get this into the pipeline later.
            # TODO [89 89 42] is a very common transition status when the agent is changing his focal from the 2 planes.
            #  Maybe I can add some fixation + majority voting behavior to get rid of the transient [89 89 42] oscillations.

            # Update the internal agent states.
            if perceived_content == 'on_env_grey':
                # Check if lost the B glass display.
                if self._task_states['current_glass_display_id'] == 1:
                    self._states['total_time_miss_glass_B'] += elapsed_time
                    self._states['current_on_level'] = -3
                elif self._task_states['current_glass_display_id'] == 2:
                    self._states['total_time_miss_glass_X'] += elapsed_time
                    self._states['current_on_level'] = -1
                elif self._task_states['current_glass_display_id'] == 0:
                    self._states['current_on_level'] = 0
            elif perceived_content == 'on_env_red':
                self._states['total_time_on_env_red'] += elapsed_time
                # Check if lost the B glass display.
                if self._task_states['current_glass_display_id'] == 1:
                    self._states['total_time_miss_glass_B'] += elapsed_time
                    self._states['current_on_level'] = -3
                elif self._task_states['current_glass_display_id'] == (0 or 2):
                    self._states['current_on_level'] = 2
            elif perceived_content == 'on_env_green':
                # Check if lost the B glass display.
                if self._task_states['current_glass_display_id'] == 1:
                    self._states['total_time_miss_glass_B'] += elapsed_time
                    self._states['current_on_level'] = -3
                elif self._task_states['current_glass_display_id'] == 2:
                    self._states['total_time_miss_glass_X'] += elapsed_time
                    self._states['current_on_level'] = -1
                elif self._task_states['current_glass_display_id'] == 0:
                    self._states['current_on_level'] = 0
            elif perceived_content == 'on_env_blue':
                # Check if lost the B glass display.
                if self._task_states['current_glass_display_id'] == 1:
                    self._states['total_time_miss_glass_B'] += elapsed_time
                    self._states['current_on_level'] = -3
                elif self._task_states['current_glass_display_id'] == 2:
                    self._states['total_time_miss_glass_X'] += elapsed_time
                    self._states['current_on_level'] = -1
                elif self._task_states['current_glass_display_id'] == 0:
                    self._states['current_on_level'] = 0
            elif perceived_content == 'on_glass_nothing':
                # Check if lost the red env.
                if self._task_states['current_env_color_id'] == 1:
                    self._states['total_time_miss_env_red'] += elapsed_time
                    self._states['current_on_level'] = -2
                elif self._task_states['current_env_color_id'] == (0 or 2 or 3):
                    self._states['current_on_level'] = 0
            elif perceived_content == 'on_glass_B':
                self._states['total_time_on_glass_B'] += elapsed_time
                self._states['current_on_level'] = 3
            elif perceived_content == 'on_glass_X':
                self._states['total_time_on_glass_X'] += elapsed_time
                # Check if lost the red env.
                if self._task_states['current_env_color_id'] == 1:
                    self._states['total_time_miss_env_red'] += elapsed_time
                    self._states['current_on_level'] = -2
                elif self._task_states['current_env_color_id'] == (0 or 2 or 3):
                    self._states['current_on_level'] = 1
            else:
                self._states['total_time_intermediate'] += elapsed_time
                self._states['current_on_level'] = 0

            # Update the timer.
            self._task_states['previous_step_timestamp'] = current_step_timestep

            # Append the rgb frames.    # TODO add a warning.
            if not self._configs['rl_pipeline']['train'] and self._configs['rl_pipeline'][
                'total_timesteps'] <= 30000:  # TODO only enable this in the testing mode. Need to be reorganized later.
                self._rgb_images.append(np.flipud(self._rgb_buffer).copy())

            # Update the boundary.
            if self._steps >= (self._num_steps - 1):
                done = True
            else:
                done = False

        return obs, done

    def _update_cam_pos(self):
        """
        This method calculates and updates the abstract mjvCamera camera pose specifications with any given lookat point.
        The algorithm is based on the inverse kinematic, which makes the viewer feels like standing still while his view point is movable.
        The moving behavior is specified by the viewer mode.
        """
        # TODO I am starting from the very easy scenario: the 1-D focal point moving.
        #  Right now the only variable is lookat point, more features, such as azimuth, elevation, distance, will be added.
        if self._viewer_mode == self._viewer_modes[0]:
            self._cam.elevation = 0
            self._cam.distance = self._init_cam_pose['cam_distance'] + np.abs(
                self._init_cam_pose['cam_lookat'][1] - self._cam.lookat[1])
        elif self._viewer_mode == self._viewer_modes[1]:
            elevation_tan = self._static_cam_height / np.abs(self._initial_cam_pos_y - self._cam.lookat[1])
            elevation_degree = - 180 * math.atan(elevation_tan) / math.pi
            self._cam.elevation = elevation_degree
            self._cam.distance = self._static_cam_height / math.sin(np.abs(elevation_tan))

    def _update_scene_render(self):
        """
        This method gets the viewport, updates the abstract mjvScene and the render, and swap buffers, processing pending events.
        """
        # Update the abstract camera pose.
        self._update_cam_pos()

        # # Viewport.
        # viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        # self._viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        # Scene.
        # TODO be careful with this function, it updates everything (as the catmask sets) according to the mujoco model and data.
        #  The dynamical geoms created in the runtime programmatically was never stored.
        mujoco.mjv_updateScene(self._model, self._data, self._opt, None, self._cam, mujoco.mjtCatBit.mjCAT_ALL.value,
                               self._scene)

        # Render.
        # TODO the rendering will slow down 3 times together with read-pixels.
        mujoco.mjr_render(self._viewport, self._scene, self._context)

        # Read the current frame's pixels.
        #  Ref: mujoco mjr_readPixels: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjr-readpixels
        #  Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer;
        #  client buffer starts at (0,0).
        mujoco.mjr_readPixels(rgb=self._rgb_buffer, depth=self._depth_buffer,
                              viewport=self._viewport, con=self._context)

        # Swap OpenGL buffers (blocking call due to v-sync)
        # TODO this is the main cause for slowing the simulation down like 16 times. And it has nothing to do with the window buffer or offscreen buffer.
        # TODO modify these later into an interactive mode.
        if self._is_window_visible == 1:
            glfw.swap_buffers(self._window)

        # Process pending GUI events, call GLFW callbacks
        # glfw.poll_events()

    def _print_logs(self):
        """
        This method prints logs of simulation.
        """
        # print the camera pose specifications (help to initialize the view)
        if self._is_print_cam_config:
            print('cam.azimuth =', self._cam.azimuth, ';', 'cam.elevation =', self._cam.elevation, ';',
                  'cam.distance = ', self._cam.distance)
            print('cam.lookat =np.array([', self._cam.lookat[0], ',', self._cam.lookat[1], ',', self._cam.lookat[2],
                  '])')
        # print the camera rgb and depth info.
        if self._is_print_cam_rgb_depth:
            print('The size of rgb buffer: {}\nThe size of depth buffer: {}\n'.format(self._rgb_buffer.shape,
                                                                                      self._depth_buffer.shape))
            print('The rgb buffer:\n {}\nThe depth buffer:\n {}\n'.format(self._rgb_buffer, self._depth_buffer))

    def _keyboard(self, window, key, scancode, act, mods):
        """
        This method defines the action of pressing the BACKSPACE key and reset the model and data.

        Args:
            key: the key on the keyboard.
            act: the key action event defined in glfw, must be one of GLFW_PRESS, GLFW_REPEAT or GLFW_RELEASE.
                # Reference: https://www.glfw.org/docs/3.3/input_guide.html#input_key
        """
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)

    def _mouse_button(self, window, button, act, mods):
        """
        This method identifies the mouse button states and the mouse's cursor position.

        Args:
            window: the window rendered by OpenGL.
        """
        # update button state
        self._button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self._button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def _mouse_move(self, window, xpos, ypos):
        """
        This method identifies the mouse button states and the mouse's cursor position.

        Args:
            window: the window rendered by OpenGL.
            xpos: the current cursor's x position.
            ypos: the current cursor's y position.
        """
        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x = xpos
        self._last_y = ypos

        # no buttons down: nothing to do
        if (not self._button_left) and (not self._button_middle) and (not self._button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self._button_right:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # The camera can only be moved if it is not set as fixed.
        mujoco.mjv_moveCamera(self._model, action, dx / height, dy / height, self._scene, self._cam)

    def _scroll(self, window, xoffset, yoffset):
        """
        This method defines the mouse scrolling interaction with the GUI display.

        Args:
            yoffset: the offset in the y-axis.
        """
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self._model, action, 0.0, -0.05 * yoffset, self._scene, self._cam)

    @property
    def get_cam_pose(self):
        """
        This property method gets the current abstract mjvCamera camera's positions,
        encoded by lookat[3], distance, azimuth, and elevation.
        Ref: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        """
        return self._cam.lookat, self._cam.distance, self._cam.azimuth, self._cam.elevation

    def write_video(self, filepath):
        """
        Writes a video from images.
        Args:
          filepath: Path where the video will be saved.
        Raises:
          ValueError: If frames per second (fps) is not set (set_fps is not called)
        """

        # Make sure fps has been set.
        fps = 120  # TODO embed this into the configuration file later.

        if self._is_depth:
            depth_images = np.flipud(self._depth_buffer).copy()
            # TODO maybe add these depth details later.

        if self._is_rgb:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filepath, fourcc, fps, tuple([self._width, self._height]))
            for img in self._rgb_images:
                out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out.release()
            print('\nThe video has been made and released to: {}.'.format(filepath))
        else:
            pass

    def encoder(self):  # TODO implement a simple cnn to recognize pictures.
        """
        This method should encode the rgb data into something DL networks, such as CNN can interpret and classify.
        For example, the pure color (there must be rule-based algorithm to classify according to rgb values),
        the pixel numbers on the plane (ImageNet).
        """
        return small_cnn(observation_shape=self._observation_shape, out_features=256)


def small_cnn(observation_shape, out_features):
    """
    An encoder (typically a PyTorch neural network) that maps the observations from higher dimensional arrays into
    vectors.
    Referenced from Aleksi's repo.  TODO learn this later.

    Args:
        observation_shape: the shape of inputs: observation space.
        out_features: the output features.

    Returns:
        A cnn network.
    """
    cnn = nn.Sequential(nn.Conv2d(in_channels=observation_shape[0], out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
                        nn.LeakyReLU(),
                        nn.Flatten())

    # Compute shape by doing one forward pass.
    with torch.no_grad():
        n_flatten = cnn(torch.zeros(observation_shape)[None]).shape[1]

    return nn.Sequential(cnn,
                         nn.Linear(in_features=n_flatten, out_features=out_features),
                         nn.LeakyReLU())

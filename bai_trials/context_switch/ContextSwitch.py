import cv2
import numpy as np
import math

import mujoco
from mujoco.glfw import glfw


from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from torch import nn
import torch

from Task import Task


class GlassSwitch(Env):

    def _load_config(self, config):
        """
        This method configures all static settings from the YAML file, including flags and counters.
        The 'static' here means configurations that will not change in the runtime.
        All the changable parts are moved into the configuration file.

        Args:
            config: the configuration content from YAML file.
        """
        # Load the configurations.
        self._config = config
        try:
            self._config_mj_env = config['mj_env']
            self._config_rl = config['rl']
            self._config_task = config['task']
        except ValueError:
            print('Invalid configurations. Check the config.yaml file.')

        # ----------------------------------------- MuJoCo environment related initializations and configurations --------------------------------------------------------------------
        # TODO Get strings. Get the names that were declared in XML.
        #  Might use the etree to programmatically access to the string names.
        # The fixed abstract camera name:
        self._camera_name = 'single-eye'
        # The geom names.
        self._geom_name_ambient_env = 'env'
        self._geom_name_smart_glass_lenses = 'smart-glass-lenses'
        self._geom_names_glass_display_ids = {
            0: 'smart-glass-display-0',
            1: 'smart-glass-display-1',
            2: 'smart-glass-display-2'
        }
        # The viewport/GLFW window resolution: height*width.
        self._width = self._config_mj_env['render']['width']
        self._height = self._config_mj_env['render']['height']

        # The flags: whether or not read rgb and depth pixel values.
        self._is_rgb = self._config_mj_env['render']['rgb']  # Set to True to read rgb.
        self._is_depth = self._config_mj_env['render']['depth']  # Set to True to read depth.

        # The camera to be fixed or not.
        self._is_camera_fixed = False
        if self._is_camera_fixed is True:
            raise ValueError(
                'The camera must not be fixed because this is a visual simulation. The eye (camera) is moving.'
            )
        # The display window, rendering buffer, and the operating system vs. OpenGL.
        # Ref https://mujoco.readthedocs.io/en/latest/programming/visualization.html#buffers-for-rendering
        self._is_window_visible = self._config_mj_env['render']['is_window_visible']
        # Check the validity of the window's visible.
        if (self._config_rl['mode'] == ('train' or 'test')) and (self._is_window_visible == 1):
            self._is_window_visible = False
            raise Warning(
                'The training and testing modes does not allow visible rendering. The window has been reset invisible.'
            )

        # Force the offscreen rendering for the train and testings for higher speed.
        if self._is_window_visible == 1:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_WINDOW.value
            # The key and mouse interaction to be enabled or not.
            if self._config_rl['mode'] == 'interact':
                self._is_key_mouse_interaction = True
            else:
                self._is_key_mouse_interaction = False
        else:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value
            # The keyboard interaction setting.
            self._is_key_mouse_interaction = False

        # The printings.
        self._is_print_cam_config = self._config_mj_env['utils']['is_print_cam_config']
        self._is_print_cam_rgb_depth = self._config_mj_env['utils']['is_print_cam_rgb_depth']

        # ----------------------------------------- The RL simulation related variables --------------------------------------------------------------------
        # Config the RL pipeline related stuff.
        # The length of the episode. The timesteps that it takes.
        self._num_steps = self._config_rl['train']['num_steps']

        # ----------------------------------------- The task related variables --------------------------------------------------------------------
        # All done in the task class.

    def _init_data(self):
        """
        Initialize MuJoCo environment data.
        This method is developed for making parameters initialization replicable.
         1. The GUI interactions, including button events and mouse cursor positions.
         2. The camera configurations / input conditions. To be noted that cam_azimuth, cam_elevation, cam_distance, and cam_lookat
            can decide where and how much the camera can see.
         3. What the camera can see are encoded as rgb and depth, and the current frame's information are stored
            in the rgb and depth buffer.
        """
        # ----------------------------------------- The MuJoCo environment related variables --------------------------------------------------------------------
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
        self._static_cam_height = 5.5
        self._initial_cam_pos_y = -7.5

        self._geom_pos_y_smart_glass_lenses = self._model.geom(self._geom_name_smart_glass_lenses).pos[1]
        self._geom_pos_y_env = self._model.geom(self._geom_name_ambient_env).pos[1]

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
        # The rgb images that will be used in writing to a video.
        self._rgb_images = []

        # ----------------------------------------- The RL task related variables --------------------------------------------------------------------
        # The global simulation step.
        self._steps = 0

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
        else:
            pass

    def _set_interactions(self):
        """
        This method installs the keyboard and mouse actions by setting up the corresponding callback functions
        in the GLFW.
        """
        if self._is_key_mouse_interaction:
            glfw.set_key_callback(self._window, self._keyboard)
            glfw.set_cursor_pos_callback(self._window, self._mouse_move)
            glfw.set_mouse_button_callback(self._window, self._mouse_button)
            glfw.set_scroll_callback(self._window, self._scroll)
        else:
            print('\nThe key and mouse interactions are disabled.')

    def __init__(self, config):
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
        # Read the configurations from the YAML file.
        self._load_config(config=config)

        # --------------------------------------- Task initialization -------------------------------------------------------------
        self._task = Task(config=config)

        # --------------------------------------- MuJoCo environment initialization -------------------------------------------------------------
        # Load the xml MjModel.
        self._model = mujoco.MjModel.from_xml_path(self._config_mj_env['model_path'])

        # Initialize MjData.
        self._data = mujoco.MjData(self._model)

        # Initialize necessary properties for RL: action_space.
        # The action will be either 1 or 0.
        #  0 for looking at the smart-glass and 1 for looking at the ambient-env.
        self.action_space = Discrete(2)     # TODO the simple version: look at the either the glass or environment.

        # The observation_space - the simple version: the perception pixels.
        offset = int(self._width / 4)       # TODO generalize it later
        self._obs_idx_h = [int(self._height / 2 - offset), int(self._height / 2 + offset)]
        self._obs_idx_w = [int(self._width / 2 - offset), int(self._width / 2 + offset)]
        self.observation_space = Box(low=np.uint8(0), high=np.uint8(255), shape=(offset * 2, offset * 2, 3))

        # --------------------------------------- Visual perception camera and rendering initialization -------------------------------------------------------------
        # TODO the rendering structure can be further enhanced following the
        #  dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/viewer/renderer.py
        # Initialize the configurations, including camera settings.
        # Ref: MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html

        # Initializations.
        # The abstract camera.
        self._cam = mujoco.MjvCamera()  # Abstract camera.

        # Reference: https://www.easytechjunkie.com/what-is-an-opengl-camera.htm#:~:text=%22OpenGL%C2%AE%20camera%22%20is%20the,which%20the%20viewer%20is%20looking.
        # Set the default settings.
        mujoco.mjv_defaultCamera(self._cam)

        # Initialize the options.
        self._opt = mujoco.MjvOption()  # Visualization options.
        mujoco.mjv_defaultOption(self._opt)

        # Initialize the OpenGL and rendering functionality.
        # Init GLFW, create window, make OpenGL context current, request v-sync.
        # Reference: https://mujoco.readthedocs.io/en/latest/programming/visualization.html#context-and-gpu-resources
        glfw.init()     # TODO if on a Linux platform, use GLContext for offscreen rendering only.
        glfw.window_hint(glfw.VISIBLE, self._is_window_visible)
        # Create an OpenGL context using GLFW.
        self._window = glfw.create_window(
            width=self._width,
            height=self._height,
            title="Bai Yunpeng's Window",
            monitor=None,
            share=None
        )
        glfw.make_context_current(self._window)
        # Ref: https://www.glfw.org/docs/3.3/group__context.html#ga6d4e0cdf151b5e579bd67f13202994ed

        # Initialize visualization data structures.
        self._scene = mujoco.MjvScene(self._model, maxgeom=10000)
        # Ref https://mujoco.readthedocs.io/en/latest/programming/visualization.html#opengl-rendering
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # Set the context's frame buffer type: a visible window buffer or an offscreen buffer.
        mujoco.mjr_setBuffer(self._framebuffer_type, self._context)

        # Viewport.
        self._viewport = mujoco.MjrRect(
            left=0,
            bottom=0,
            width=self._width,
            height=self._height
        )

        # --------------------------------------- Data initializations -------------------------------------------------------------
        # Update the dynamic parameters.
        self._init_data()

        # --------------------------------------- Confirmation -------------------------------------------------------------
        # Finally confirm all the settings.
        mujoco.mj_forward(m=self._model, d=self._data)

    def reset(self):
        """
        The override method: reset all data.

        Returns:
            obs: the reset observations.
        """
        # Reset the task.
        self._task.reset()

        # Reset the simulated environment.
        mujoco.mj_resetData(m=self._model, d=self._data)

        # Reset the data.
        self._init_data()

        # Get the observations.
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
        obs, reward, done, info = self._update(action=action)

        # Update the simulated steps.
        self._steps += 1

        return obs, reward, done, info

    def _update(self, action):
        """
        The step-wise update.
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
        # --------------------------------------------- task jobs --------------------------------------------------
        # Update the task one step.
        alpha, color_id, task_status, elapsed_time = self._task.update(
            action=action,
            time=self._data.time
        )

        # --------------------------------------------- MuJoCo env --------------------------------------------------
        # Update MuJoCo environment stuff related to the action.
        if action == 0:  # look at the smart glass lenses
            self._cam.lookat[1] = self._geom_pos_y_smart_glass_lenses
        elif action == 1:  # look at the env
            self._cam.lookat[1] = self._geom_pos_y_env
        else:
            pass

        # Sync up the visualized focal point - ball's movement.
        self._data.qpos[0:3] = self._cam.lookat

        # Sync up the correct smart glass's content transparency according to the above update.
        for i in range(len(self._geom_names_glass_display_ids)):
            if i == 0:  # the display 0 is always transparent - display nothing
                self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = 0
            elif i == task_status['current_glass_display_id']:
                self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = alpha
            else:
                self._model.geom(self._geom_names_glass_display_ids[i]).rgba[3] = 0

        # Sync up the correct env color according to the above update.
        self._model.geom(self._geom_name_ambient_env).rgba[0:3] = self._config_task['env_color_choices'][color_id]

        # Update the scene and render everything.
        self._update_scene_render()

        # --------------------------------------------- task decisions --------------------------------------------------
        # Get the task's decisions.
        self._task.make_decision(
            action=action,
            rgb=self._rgb_buffer.copy(),
            elp_time=elapsed_time,
        )

        # --------------------------------------------- obs --------------------------------------------------
        # Get the observations.
        obs = self._get_obs()

        # Append the rgb frames.
        if (self._config_rl['mode'] == 'test') and (self._config_rl['train']['total_timesteps'] <= 30000):
            self._rgb_images.append(np.flipud(self._rgb_buffer).copy())

        # --------------------------------------------- done --------------------------------------------------
        # Update the boundary.
        if self._steps >= (self._num_steps - 1):
            done = True
        else:
            done = False

        # --------------------------------------------- info --------------------------------------------------
        # Update the info.
        info = self._task.states

        # --------------------------------------------- rewards --------------------------------------------------
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
        obs = rgb
        return obs

    def _reward_function(self):
        """
        Defines the reward function and gets the reward value.

        Returns:
            reward: the reward value for the current step.
        """
        # Write the reward function based on the agent's states.
        states = self._task.states.copy()

        # Get the rewards. Write in this awkward form because it is easy to tune them.
        if states['current_on_level'] == 3:
            reward = 5
        elif states['current_on_level'] == 2:
            reward = 3
        elif states['current_on_level'] == 1:
            reward = 1
        elif states['current_on_level'] == -1:
            reward = -1
        elif states['current_on_level'] == -2:
            reward = -3
        elif states['current_on_level'] == -3:
            reward = -5
        else:
            reward = 0

        return reward

    def render(self, mode="human"):
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        Here I referenced from MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html,
        to build my camera and GL rendering.
        """
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

    def _update_cam_pos(self):
        """
        This method calculates and updates the abstract mjvCamera camera pose specifications with any given lookat point.
        The algorithm is based on the inverse kinematic, which makes the viewer feels like standing still while his view point is movable.
        The moving behavior is specified by the viewer mode.
        """
        if self._config_rl['mode'] == 'interact':
            elevation_tan = self._static_cam_height / np.abs(self._initial_cam_pos_y - self._cam.lookat[1])
            elevation_degree = - 180 * math.atan(elevation_tan) / math.pi
            self._cam.elevation = elevation_degree
            self._cam.distance = self._static_cam_height / math.sin(np.abs(elevation_tan))
        else:
            self._cam.elevation = 0
            self._cam.distance = self._init_cam_pose['cam_distance'] + np.abs(self._init_cam_pose['cam_lookat'][1] - self._cam.lookat[1])

    def _update_scene_render(self):
        """
        This method gets the viewport, updates the abstract mjvScene and the render, and swap buffers, processing pending events.
        """
        # Update the abstract camera pose.
        self._update_cam_pos()

        # Scene.
        # TODO be careful with this function, it updates everything (as the catmask sets) according to the mujoco model and data.
        #  The dynamical geoms created in the runtime programmatically was never stored.
        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene
        )

        # Render.
        mujoco.mjr_render(self._viewport, self._scene, self._context)

        # Read the current frame's pixels.
        #  Ref: MuJoCo mjr_readPixels: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjr-readpixels
        mujoco.mjr_readPixels(
            rgb=self._rgb_buffer,
            depth=self._depth_buffer,
            viewport=self._viewport,
            con=self._context
        )

        # Visualize the renderings. Warning: largely slow down the simulation!
        if self._is_window_visible == 1:
            # Swap the buffer from back to front.
            glfw.swap_buffers(self._window)
            # Process pending GUI events, call GLFW callbacks
            glfw.poll_events()

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

    @property
    def num_steps(self):
        """
        This property gets the number of steps assigned in the environment.
        """
        return self._num_steps

    def encoder(self):  # TODO implement a simple cnn to recognize pictures.
        """
        This method should encode the rgb data into something DL networks, such as CNN can interpret and classify.
        For example, the pure color (there must be rule-based algorithm to classify according to rgb values),
        the pixel numbers on the plane (ImageNet).
        """
        # return small_cnn(observation_shape=self._observation_shape, out_features=256)
        pass


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

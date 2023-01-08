import numpy as np
import os
import math
from typing import Callable, NamedTuple, Optional, Union, List

import mujoco
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET   # This package.class could be used to add XML elements dynamically.

import gym
from gym import Env
from gym import utils
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete


class GlassSwitch(Env):

    def _config(self):
        """
        This method configures all static configurations, including flags and counters.
        The 'static' here means configurations that will not change in the runtime.
        And this is the only changable part of the simulation by the user.
        """
        # TODO this part will be connected with __init__ later.
        width = 400
        height = 320
        is_rgb = True
        is_depth = True
        is_camera_fixed = False
        is_window_visible = 1
        is_key_mouse_interaction = True
        is_print_cam_config = False
        is_print_cam_rgb_depth = False
        simend = 500
        viewer_mode = 'through_glass'
        task_mode = 'demo'
        # --------------------------------------------------------------------------------------------------------------

        # Get strings.
        # Get the names that were declared in XML.
        # TODO might use the etree to programmatically access to the string names.
        # The fixed abstract camera name:
        self._camera_name = 'fixed-eye'
        # The geom names.
        self._geom_name_ambient_env = 'ambient-env'
        self._geom_name_smart_glass_lenses = 'smart-glass-lenses'
        self._geom_name_smart_glass_display_1 = 'smart-glass-display-1'

        # OpenGL.
        # Resolution in pixels: width * height
        # Reference: glfw.GLFWwindow * glfwCreateWindow
        # https://www.glfw.org/docs/3.3/group__window.html#ga3555a418df92ad53f917597fe2f64aeb
        # TODO might need to scale the resolution down to increment the computing and rendering speed.
        self._width = width
        self._height = height

        # Flags.
        # Internal buffer configuration.
        self._is_rgb = is_rgb     # Set to True to read rgb.
        self._is_depth = is_depth   # Set to True to read depth.
        # The camera to be fixed or not.
        self._is_camera_fixed = is_camera_fixed   # Set to True to disable the camera moving functions.
        # The display window.
        self._is_window_visible = is_window_visible    # Set to 1 to enable the window being visible. 0 to hide the window.
        if self._is_window_visible == 1:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_WINDOW.value
        elif self._is_camera_fixed == 0:
            self._framebuffer_type = mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value
        # The key and mouse interaction to be enabled or not.
        self._is_key_mouse_interaction = is_key_mouse_interaction
        # Raise some value errors when the settings are having conflicts.
        if self._is_key_mouse_interaction and (self._is_camera_fixed or self._is_window_visible == 0):
            raise ValueError('Invalid keyboard and mouse settings: a visible window and a free camera are both necessary.')
        # The printings.
        self._is_print_cam_config = is_print_cam_config        # Set to True to print camera configuration.
        self._is_print_cam_rgb_depth = is_print_cam_rgb_depth     # Set to True to print camera read pixel rgb and depth info.

        # Constant configuration.
        # The simulation time.
        self._simend = simend  # Simulation time. TODO this can always be changed as I need.
        # The display modes. Must select from the predefined ones.
        self._viewer_modes = ['through_glass', 'overhead']
        self._viewer_mode = viewer_mode
        if self._viewer_mode not in self._viewer_modes:
            raise ValueError('Invalid viewer mode: it must be selected from the predefined options [through_glass, overhead].')
        # The task modes. Must select from the predefined modes.
        # 'demo' corresponds to the demonstration, a visible window is required.
        # The others are tasks used in RL, they can be operated in an off-screen manner.
        self._task_modes = ['demo', 'color_switch']
        self._task_mode = task_mode
        if self._task_mode not in self._task_modes:
            raise ValueError('Invalid task mode: it must be selected from the predefined options [demo, color_switch].')
        if self._task_mode == self._task_modes[0] and (self._is_window_visible == 0 or self._is_camera_fixed is True):
            raise ValueError('Invalid configuration: the window must be visible and the camera must be free in the demo task mode.')
        # Focal point moving distance configurations.
        self._dist_configs = {
            'min_dist': 0,
            'max_dist': 1.5,  # TODO This could be a tunable parameter in the design space.
        }
        self._mapping_range = self._dist_configs['max_dist'] - self._dist_configs['min_dist']

    def _init_cam_dyn_data(self):
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
        self._set_cam_variability()
        # mjvCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        # Set the camera position and pose according to the environment setting, e.g., the planes' positions.
        if self._viewer_mode == self._viewer_modes[0]:  # through the glass
            self._static_cam_height = 5.5
            self._initial_cam_pos_y = -7.5
            pass
        elif self._viewer_mode == self._viewer_modes[1]:    # overhead
            # TODO firstly I just do a 1-D change on the y axis. Complete later.
            self._static_cam_height = 5.5
            self._initial_cam_pos_y = -7.5

        self._init_cam_pose = {
            'cam_lookat': np.array([0, self._model.geom(self._geom_name_smart_glass_lenses).pos[1], 1.5]),  # lookat point
            'cam_distance': 5,  # Distance to lookat point or tracked body.
            'cam_azimuth': 90.0,    # Camera azimuth (deg). The concept reference: https://www.photopills.com/articles/understanding-azimuth-and-elevation
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

    def _set_cam_variability(self):
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

    def __init__(self, xml_path):
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

        Args:
            xml_path: the read-in mujoco model xml filepath.
        """
        # Initialize the configurations, including camera settings.
        # Ref: MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html
        # Configure the static parameters.
        self._config()
        # --------------------------------------- RL initialization -------------------------------------------------------------
        # Load the xml MjModel.
        self._model = mujoco.MjModel.from_xml_path(xml_path)

        # Initialize MjData.
        self._data = mujoco.MjData(self._model)

        # Initialize necessary properties for RL: action_space and observation_space.
        # TODO finish later, should specify where should the camera see, can be encoded into: arithma, elevation, distance, and focal point.
        self.action_space = None
        self.observation_space = None   # TODO finish later, should specify what the camera sees, the pixels.
        # TODO replace later.
        self._step = 0
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
        self._init_cam_dyn_data()

        # Initialize the options.
        self._opt = mujoco.MjvOption()  # Visualization options.
        mujoco.mjv_defaultOption(self._opt)

        # Initialize the OpenGL and rendering functionality.
        # Init GLFW, create window, make OpenGL context current, request v-sync.
        #  This process has been encapsulated in the glfw.GLContex, I configure it explicitly here for controling
        #  the visibility of the OpenGL window.
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, self._is_window_visible)
        self._window = glfw.create_window(self._width, self._height, "Bai Yunpeng's Window", None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        # Initialize visualization data structures.
        self._scene = mujoco.MjvScene(self._model, maxgeom=10000)
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # Set the context's frame buffer type: a visible window buffer or an offscreen buffer.
        mujoco.mjr_setBuffer(self._framebuffer_type, self._context)

        # --------------------------------------- XXX initialization -------------------------------------------------------------
        # Initialize the mouse and keyboard interactions if permitted.
        self._set_interactions()
        # Initializing the environment variation functions.
        # TODO in the design space part:
        #  The design_space method: Only one plane will be green, the other plane should be ***changed*** at some extend,
        #  e.g., the color or the alpha. This could be the insight of developing heads-up computing using RL.

        # --------------------------------------- Confirmation -------------------------------------------------------------
        # Finally confirm all the settings.
        mujoco.mj_forward(m=self._model, d=self._data)

    def reset(self):
        # Reset the simulated environment. TODO check whether the dynamic values are changed to their defaults.
        mujoco.mj_resetData(m=self._model, d=self._data)

        # Reset the abstract camera related data.
        self._init_cam_dyn_data()

        # Reset the RL related data. TODO to pack these variables later.
        self._step = 0

        ob = self._get_obs()
        return ob

    def step(self, action):
        # Advance the simulation
        mujoco.mj_step(m=self._model, d=self._data)  # nstep=self._run_parameters["frame_skip"]
        # TODO maybe try using mj_forward sometimes, it might be enough.
        # mujoco.mj_forward(m=self._model, d=self._data)

        # Update environment.
        reward, done, info = self._update(action=action)

        # Update the simulated steps.
        self._steps += 1

        # Get observation.
        obs = self._get_obs()

        return obs, reward, done, info

    def _update(self, action):
        # TODO outline
        #  1. action update.
        #  2. environment update, details are listed below.

        # TODO The hierarchy game
        #  B letter - higher priority: stay on the glass until the content changed.
        #   Only when the focal point is on the glass that the agent knows the content.
        #   The rewards are proportional to the time it stays on the smart glass.
        #  Red color - lower priority: stay on the env until the color changed.
        #   The rewards are proportional to the time it stays on the wall.
        #   The tricky part is: letting the agent learn, when he is looking at the wall, it must go back and check the smart
        #   glass content. What would be the best focal change frequency.

        # Setting the done flag.
        done = False

        # Update the info.
        self.info = {}  # TODO initialize this later in the __init__

        # Calculate the reward
        reward = self._reward_function()

        return reward, done, self.info

    def _get_obs(self):
        # TODO outline
        #  the next stage would be recognizing some text shown on the planes, but 2 more steps are needed:
        #  1. asset implementation, i.e., prepare a png photo or stl model, then introduce it into my env.
        #  2. picture recognition using CNN.
        #  3. Just like Aleksi's Perception-vision-base.py, the encoded pixel values could be directly fed into the
        #   observation space for RL to learn. Reference to get_observation method's comments.

        # TODO
        #  1. the perception port range factors: lookat point, azithum, elevation, distance.
        #  2. the perception content factors: encoded rgb + depth pixels.
        obs = {}
        return obs

    def _reward_function(self):
        # TODO outline
        #  1. considering using the sum-up manner, i.e., the environment and task has their own reward_functions,
        #   the sum of the scores will be optimized.
        #  2. the reward_functions should be high-level and generic enough to utilize the advantages of RL.
        #   Or there is no difference to using rule-based/hard coded algorithm.

        reward = 0

        return reward

    def render(self):  # , mode="human"
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        Here I referenced from MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html,
        to build my camera and GL rendering.
        """
        # Render the demo in the 'demo' task mode.
        if self._task_mode == self._task_modes[0]:
            while not glfw.window_should_close(self._window):   # Decides when the window will be closed, such as a mouse click event.
                time_prev = self._data.time

                # Skip some frames, restrict the render fps to be around 60Hz.
                while self._data.time - time_prev < 1.0 / 60.0:
                    mujoco.mj_step(self._model, self._data)
                # The smoother and natural render update.
                # mujoco.mj_step(self._model, self._data)

                if self._data.time >= self._simend:
                    break

                # Do the given task.
                self._do_task()

                # Update the scene and render everything.
                self._update_scene_render()

                # Print logs.
                self._print_logs()
        # The task mode is 'color_switch'.
        elif self._task_mode == self._task_modes[1]:
            # TODO do the framewise updates in the override step().
            pass

    def close(self):
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        It terminates the glfw, end rendering and display.
        """
        glfw.destroy_window(self._window)
        self._window = None
        glfw.terminate()

    def _do_task(self):
        """
        This method determines how the tasks are performed.
        The most importantly, it specifies how the actions are taken, and how the observation space should be changed.

        Before I made a separate task class, this method is the temporal place to hold all task logics.
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
            smart_glass_lenses_y = self._model.geom(self._geom_name_smart_glass_lenses).pos[1]
            alpha_change_gain = 1.0  # The gain could be used later. In a non-linear control.
            abs_y_dist = abs(self._cam.lookat[1] - smart_glass_lenses_y)

            if abs_y_dist >= self._dist_configs['max_dist']:
                alpha = 0
            else:
                alpha = alpha_change_coefficient * (self._dist_configs['max_dist'] - abs_y_dist) / self._mapping_range

            # Update the alpha from rgba's transparency value.
            self._model.geom(self._geom_name_smart_glass_display_1).rgba[3] = alpha

            # Sync the visualized focal point - ball's movement.
            self._data.qpos[0:3] = self._cam.lookat
            # print('elevation: {}, qpos: {}'.format(self._cam.elevation, self._data.qpos[0:3]))
        elif self._task_mode == self._task_modes[1]:    # the rl task: color switch
            # TODO the task specification: the viewer will be in the 'through_glass' viewer mode. Through the half-transparent smart-glass-lenses,
            #  the agent can perceive pixel values of the environment plane: 'ambient-env'. The agent's task is simply
            #  move his focal point to the ambient env whenever he detects any color change happened on the 'ambient-env'.
            #  And responds correctly to the correct color, i.e., the red color, when the focal point is fixed on the ambient-env.
            #  To be noted that when the fixed eye moves from the smart glass lenses (sgl), the sgl's transparency will increase and thus its masking effect
            #  over the environment will be diminished. This would be the first part of the task.

            pass

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
            self._cam.distance = self._init_cam_pose['cam_distance'] + np.abs(self._init_cam_pose['cam_lookat'][1] - self._cam.lookat[1])
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

        # Viewport.
        viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        self._viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        # Scene.
        # TODO be careful with this function, it updates everything (as the catmask sets) according to the mujoco model and data.
        #  The dynamical geoms created in the runtime programmatically was never stored.
        mujoco.mjv_updateScene(self._model, self._data, self._opt, None, self._cam, mujoco.mjtCatBit.mjCAT_ALL.value,
                               self._scene)

        # Render.
        mujoco.mjr_render(self._viewport, self._scene, self._context)

        # Read the current frame's pixels.
        #  Ref: mujoco mjr_readPixels: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjr-readpixels
        #  Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer;
        #  client buffer starts at (0,0).
        mujoco.mjr_readPixels(rgb=self._rgb_buffer, depth=self._depth_buffer,
                              viewport=self._viewport, con=self._context)

        # Swap OpenGL buffers (blocking call due to v-sync)
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

    def encoder(self):  # TODO implement a simple cnn to recognize pictures.
        """
        This method should encode the rgb data into something DL networks, such as CNN can interpret and classify.
        For example, the pure color (there must be rule-based algorithm to classify according to rgb values),
        the pixel numbers on the plane (ImageNet).
        """
        # return small_cnn(observation_shape=self._observation_shape, out_features=256)
        return

    @property
    def get_cam_pose(self):
        """
        This property method gets the current abstract mjvCamera camera's positions,
        encoded by lookat[3], distance, azimuth, and elevation.
        Ref: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        """
        return self._cam.lookat, self._cam.distance, self._cam.azimuth, self._cam.elevation

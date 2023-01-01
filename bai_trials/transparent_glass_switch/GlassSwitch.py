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
    @property
    def _config_cam_sta_params(self):
        """
        This method configures all camera related static values, including flags and counters.
        The 'static' here means configurations that will not change in one simulation.
        """
        # Names.
        # The fixed abstract camera name:
        self._camera_name = 'fixed-eye'
        # The geom names.
        self._geom_ambient_env_name = 'ambient-env'
        self._geom_smart_glass_name = 'smart-glass'

        # OpenGL.
        # Resolution in pixels: width * height
        # Reference: glfw.GLFWwindow * glfwCreateWindow
        # https://www.glfw.org/docs/3.3/group__window.html#ga3555a418df92ad53f917597fe2f64aeb
        self._width = 1200
        self._height = 960
        self._resolution = [self._width, self._height]

        # Flags.
        # Internal buffer configuration.
        self._is_rgb = True     # Set to True to read rgb.
        self._is_depth = True   # Set to True to read depth.
        # The camera to be fixed or not.
        self._is_camera_fixed = False   # Set to True to disable the camera moving functions.
        # The display window.
        self._is_window_visible = 1    # Set to 1 to enable the window being visible.
        # The key and mouse interaction to be enabled or not.
        if (self._is_window_visible == 1) and (self._is_camera_fixed is False):
            self._is_key_mouse_interaction = True   # Set to True to enable mouse and keyboard interaction on the display window.
        else:
            self._is_key_mouse_interaction = False
        # The printings.
        self._is_print_camera_config = False        # Set to True to print camera configuration.
        self._is_print_camera_rgb_depth = True     # Set to True to print camera read pixel rgb and depth info.

        # Constant configuration.
        self._simend = 500  # Simulation time. TODO this might be changed later.

    @property
    def _init_cam_dyn_params(self):
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
        self._lastx = 0
        self._lasty = 0

        # Initialize the abstract camera's pose specifications.
        # mjvCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        self._original_viewport_height = 5.5   # TODO firstly we just do a 1-D change on the y axis. Delete later.
        self._original_viewport_dist = 5.2
        self._original_viewport_y = -7.5

        self._init_cam_config = {
            'cam_lookat': np.array([0, -0.3, 1.5]),  # Lookat point
            'cam_distance': self._original_viewport_dist,  # Distance to lookat point or tracked body.
            'cam_azimuth': 90.0,    # Camera azimuth (deg). The concept reference: https://www.photopills.com/articles/understanding-azimuth-and-elevation
            'cam_elevation': 0  # Camera elevation (deg).
        }

        # Assign the initial values.
        self._cam.azimuth = self._init_cam_config['cam_azimuth']
        self._cam.elevation = self._init_cam_config['cam_elevation']
        self._cam.distance = self._init_cam_config['cam_distance']
        self._cam.lookat = self._init_cam_config['cam_lookat']

        # Internal buffers: rbg buffer and depth buffer.
        self._rgb_buffer = np.empty((self._resolution[1], self._resolution[0], 3),
                                    dtype=np.uint8) if self._is_rgb else None
        self._depth_buffer = np.empty((self._resolution[1], self._resolution[0]),
                                      dtype=np.float32) if self._is_depth else None

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
        # ----------------------------------------------------------------------------------------------------
        # Initialize the mujoco py-bindings.
        # Load the xml MjModel.
        self._model = mujoco.MjModel.from_xml_path(xml_path)

        # Initialize MjData.
        self._data = mujoco.MjData(self._model)

        # Initialize necessary properties for RL: action_space and observation_space.
        # TODO finish later, should specify where should the camera see, can be encoded into: arithma, elevation, distance, and focal point.
        # TODO but if the eye is fixed, then the only thing that could be changed should be the looking point.
        self.action_space = None
        self.observation_space = None   # TODO finish later, should specify what the camera sees, the pixels.

        # ----------------------------------------------------------------------------------------------------
        # Initialize the camera functionality.
        # Ref: MuJoCoPy Bootcamp Lec 13: https://pab47.github.io/mujocopy.html
        # Configure the camera related static parameters.
        self._config_cam_sta_params

        # Initializations.
        # The abstract camera.
        self._cam = mujoco.MjvCamera()  # Abstract camera.
        # Concepts:
        #  An "abstract camera" refers to a virtual camera that is implemented in software and is not directly
        #   related to the rendering of the scene using OpenGL.
        #   An abstract camera can be used to define a viewing volume or frustum,
        #   as well as to specify the position and orientation of the camera in 3D space.
        #
        #  On the other hand, an "OpenGL camera" refers to a camera that is implemented using OpenGL functions and is
        #   used to control the rendering of a scene using OpenGL.
        #   An OpenGL camera is typically defined by a projection matrix and a view matrix,
        #   which are used to transform the 3D coordinates of the objects in the scene into 2D coordinates for
        #   rendering on the screen.
        #
        # "OpenGL® camera" is the name given to the virtual position of a viewer within an Open Graphics Library®
        #  (OpenGL®) scene. It is defined by the position of the viewer within the scene, and then the location or
        #  direction in which the viewer is looking. The position of the camera in an OpenGL® scene will determine
        #  what portion of a scene will be rendered to the display device and at what angle.
        #  Reference: https://www.easytechjunkie.com/what-is-an-opengl-camera.htm#:~:text=%22OpenGL%C2%AE%20camera%22%20is%20the,which%20the%20viewer%20is%20looking.

        mujoco.mjv_defaultCamera(self._cam)
        # Update the camera related dynamic parameters.
        self._init_cam_dyn_params

        # Identify whether to make the abstract camera's position fixed.
        # Reference: Aleksi's rendering.py, class Camera. https://github.com/BaiYunpeng1949/user-in-the-box/blob/main/uitb/utils/rendering.py
        # Reference: mjvCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjvcamera
        # Reference: mjtCamera: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjtcamera
        # Get the camera ID. camid=0; ncam=1
        if self._is_camera_fixed:
            if isinstance(self._camera_name, str):
                self._camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
            if self._camera_id < -1:
                raise ValueError('camera_id cannot be smaller than -1. The abstract camera must be specified in the XML model.')
            if self._camera_id >= self._model.ncam:
                raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.format(self._model.ncam, self._camera_id))
            # Fix the camera.
            self._cam.fixedcamid = self._camera_id
            if self._camera_id == -1:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
                # camera explicitly defined in the model.
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # The options.
        self._opt = mujoco.MjvOption()  # Visualization options.
        mujoco.mjv_defaultOption(self._opt)

        # ----------------------------------------------------------------------------------------------------
        # Initialize the OpenGL and rendering functionality.
        # Init GLFW, create window, make OpenGL context current, request v-sync.
        #  This process has been encapsulated in the glfw.GLContex, I configure it explicitly here for controling
        #  the visibility of the OpenGL window.
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, self._is_window_visible)
        self._window = glfw.create_window(self._width, self._height, "Bai Yunpeng's Window", None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        # Initialize the mouse and keyboard interactions if permitted.
        if self._is_key_mouse_interaction:
            self._enable_key_mouse_actions

        # Initialize visualization data structures.
        self._scene = mujoco.MjvScene(self._model, maxgeom=10000)
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # # TODO debug
        # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._context)

        # ----------------------------------------------------------------------------------------------------
        # Initializing the environment variation functions.
        # TODO should include the functions: 1. the adaptive first plane transparency change function. 2. The planes' color changing function. 3. The ball (visualized focal point) moving function.
        #  correspondingly, if the camera could always track the green plane, then it would be rewards.
        #  When the glass is green, it just need to learn to quickly stare at the glass and recognize it as green (the color identification would be given by me).
        #  When the environment is green, the camera needs to learn to quickly fix its focal point onto the environment.

        # TODO in the design space part:
        #  The design_space method: Only one plane will be green, the other plane should be ***changed*** at some extend, e.g., the color or the alpha. This could be the insight of developing heads-up computing using RL.

        # TODO in the perceptual control part:
        #  1. The more realistic and accurate control: use a foveated vision filter.
        #  2. (Priority) The cheating alternative: the ball will represent the gaze point,
        #   when the gaze point touches different planes, different information will be disclosed.
        #   For example, when the gaze point is in the green plane, the plane will be less transparent,
        #   and the information would be clearer for a decision_making method (encoding + CNN) to tell.

        # TODO in the task part:
        #  The agent will be looking for the X letter on either the glass plane or the environment plane.
        #  While the other plane will have a misleading letter Y or something.
        #  The agent will only be rewarded if it recognizes the X (through the pixel information with decision_making method),
        #  and it will be punished if it is spending too much time switching between the planes OR the X appears on the non-target.
        #  Usually the 2 planes will be quite transparent, they will only be solid when the gaze point (visualized by the ball) fixed on it.
        #  And that is also when more information is disclosed and could be recognized.
        #  ------------------------------------------------------------------------------------------------------------
        #  Ultimately, we can simulate the scenario when the eye is perceiving information from the front plane,
        #  how the information from the back plane are losing.

        # ----------------------------------------------------------------------------------------------------
        # Finally confirm all the settings.
        mujoco.mj_forward(self._model, self._data)

    @property
    def _enable_key_mouse_actions(self):
        """
        This property method installs the keyboard and mouse actions by setting up the correspoding callback functions
        in the glfw.
        """
        glfw.set_key_callback(self._window, self._keyboard)
        glfw.set_cursor_pos_callback(self._window, self._mouse_move)
        glfw.set_mouse_button_callback(self._window, self._mouse_button)
        glfw.set_scroll_callback(self._window, self._scroll)

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
        dx = xpos - self._lastx
        dy = ypos - self._lasty
        self._lastx = xpos
        self._lasty = ypos

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

    def reset(self):
        # TODO outline
        #  Reset all variables. (replicate the __init__)

        # Reset the simulated environment.
        mujoco.mj_resetData(self._model, self._data)
        pass

    def step(self, action):
        # Advance the simulation
        mujoco.mj_step(self._model, self._data)  # nstep=self._run_parameters["frame_skip"]
        # TODO try using mj_forward, it should be enough.

        # Update environment.
        reward, done, info = self._update(action=action)

        # Update the simulated steps.
        self._steps += 1  # TODO declare this variable later.

        # Get observation.
        obs = self._get_obs()

        return obs, reward, done, info
        pass

    def _update(self, action):
        # TODO outline
        #  1. action update.
        #  2. environment update, details are listed below.

        # TODO in the text update part: <need to be checked/confirmed later>
        #  It seems that the text shown on the 'smart-glass' can not be updated dynamically by reloading different
        #  xml configurations, this is because the xml parser and compilation has already been finished before the run time.
        #  Hence that one possible way is creating multiple smart glasses with different content textures attached to them.
        #  And I will change their visibility accordingly to make it as the content is updating.

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
        # TODO ---------------------------------- temporary coding playground ------------------------------
        # I try to render using glfw with a custom window. TODO: this part should be in the step method.
        while not glfw.window_should_close(self._window):
            time_prev = self._data.time

            while self._data.time - time_prev < 1.0 / 60.0:
                mujoco.mj_step(self._model, self._data)

            if self._data.time >= self._simend:
                break
            # -------------------------------------------------------------------------------------------------------
            # Update the scene TODO debug delete later
            abs_dx = 0.005
            abs_da = 0.0015
            _my_timestep_interval = 10
            _my_num_changes = 100
            for i in range(_my_num_changes):
                if self._data.time <= i * _my_timestep_interval:
                    if i % 2 != 0:
                        dx = abs_dx
                        da = abs_da
                    else:
                        dx = - abs_dx
                        da = - abs_da
                    break

            self._cam.lookat[1] += dx

            # TODO find an established algorithm for this kind of control.
            original_sg_y = self._model.geom('smart-glass').pos[1]
            gain_alpha_change = 1.0 # The gain could be used later. In a non-linear control.
            y_abs_dist = abs(self._cam.lookat[1] - original_sg_y)
            dist_map = {
                'min_dist': 0,
                'max_dist': 2,
                'min_dist_alpha': 1,
                'max_dist_alpha': 0
            }
            mapping_range = dist_map['max_dist'] - dist_map['min_dist']
            if y_abs_dist >= dist_map['max_dist']:
                alpha = 0
            else:
                alpha = (dist_map['max_dist'] - y_abs_dist) / mapping_range

            # The elevation update. TODO
            x = self._original_viewport_height / np.abs(
                self._original_viewport_y - self._cam.lookat[1])
            elevation = - 180 * math.atan(x) / math.pi  # TODO debug delete later.

            # TODO set 1 - dynamic elevation
            self._cam.elevation = elevation
            self._cam.distance = self._original_viewport_height / math.sin(np.abs(x))

            # TODO set 2 - constant elevation = 0
            # self._cam.elevation = 0
            # self._cam.distance = self._original_viewport_dist + \
            #                      np.abs(self._init_cam_config['cam_lookat'][1] - self._cam.lookat[1])

            # Update the alpha from rgba.
            # Reference: https://github.com/deepmind/mujoco/issues/114#issuecomment-1020594654
            # From there I realized the model, which was stated to be treated as a constant object could still be changed!
            # self._model.geom(self._geom_smart_glass_name).rgba[3] -= da # TODO trial uncomment
            self._model.geom(self._geom_smart_glass_name).rgba[3] = alpha
            # TODO: design a looking algorithm that seems like the camera is fixed. Inverse kinematic algorithm.
            # To implement inverse kinematics for a camera, you will need to define the kinematic chain for the camera, which may include the camera's position, orientation, and various other parameters such as the distance from the camera to the lookat point, the azimuth and elevation angles, and so on. You will then need to define the desired position and orientation for the camera, and use an inverse kinematics algorithm to compute the joint angles that will allow the camera to reach that position and orientation.
            # Get the ball moving along. TODO: need a method to sync the ball and the focal point.
            self._data.qpos[0:3] = self._cam.lookat
            print('elevation: {}, qpos: {}'.format(self._cam.elevation, self._data.qpos[0:3]))
            #-------------------------------------------------------------------------------------------------------

            # Get framebuffer viewport.
            self._get_framebuffer_viewport

            # print camera configuration (help to initialize the view)
            if self._is_print_camera_config:
                print('cam.azimuth =', self._cam.azimuth, ';', 'cam.elevation =', self._cam.elevation, ';', 'cam.distance = ', self._cam.distance)
                print('cam.lookat =np.array([', self._cam.lookat[0], ',', self._cam.lookat[1], ',', self._cam.lookat[2], '])')

            # Update scene and render.
            self._update_scene_render

            # Read the current frame's pixels.
            #  Ref: mujoco mjr_readPixels: https://mujoco.readthedocs.io/en/stable/APIreference.html#mjr-readpixels
            #  Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer;
            #  client buffer starts at (0,0).
            mujoco.mjr_readPixels(rgb=self._rgb_buffer, depth=self._depth_buffer,
                                  viewport=self._viewport, con=self._context)
            if self._is_print_camera_rgb_depth:
                print('The size of rgb buffer: {}\nThe size of depth buffer: {}\n'.format(self._rgb_buffer.shape, self._depth_buffer.shape))
                print('The rgb buffer:\n {}\nThe depth buffer:\n {}\n'.format(self._rgb_buffer, self._depth_buffer))

            # Swap OpenGL buffers (blocking call due to v-sync)
            #  The glfw.swap_buffers function is used to swap the front and back buffers of a window.
            #  A window typically has two buffers: the front buffer and the back buffer.
            #  The front buffer is the one that is displayed on the screen, and the back buffer is used for drawing.
            #  When glfw.swap_buffers is called, it swaps the front and back buffers of the window,
            #  so that the image in the back buffer becomes visible on the screen.
            #  This allows the application to draw the next frame in the back buffer while the previous frame is being displayed in the front buffer.
            glfw.swap_buffers(self._window)

            # Process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        # glfw.terminate()

    def close(self):
        """
        This is an override method inherited from its parent class gym.Env,
        it is compulsory and might be called in the established RL pipeline.

        It terminates the glfw, end rendering and display.
        """
        glfw.destroy_window(self._window)
        self._window = None
        glfw.terminate()

    @property
    def _get_framebuffer_viewport(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        # viewport_width = self._width
        # viewport_height = self._height
        self._viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    @property
    def _update_scene_render(self):
        mujoco.mjv_updateScene(self._model, self._data, self._opt, None, self._cam, mujoco.mjtCatBit.mjCAT_ALL.value,
                               self._scene)
        mujoco.mjr_render(self._viewport, self._scene, self._context)

    @property
    def encoder(self):  # TODO implement a simple cnn to recognize pictures.
        """
        This method should encode the rgb data into something DL networks, such as CNN can interpret and classify.
        For example, the pure color (there must be rule-based algorithm to classify according to rgb values),
        the pixel numbers on the plane (ImageNet).
        """
        # return small_cnn(observation_shape=self._observation_shape, out_features=256)
        return
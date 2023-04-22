import os

import yaml
import cv2
from tqdm import tqdm
import numpy as np
from typing import Callable

import gym

import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from huc.utils.write_video import write_video
from huc.envs.moving_eye.MovingEye import MovingEye
from huc.envs.reading_eye.ReadingEye import ReadingEye
from huc.envs.context_switch.ContextSwitch import ContextSwitch
from huc.envs.context_switch_replication.ContextSwitchReplication import ContextSwitchReplication
from huc.envs.context_switch_replication.SwitchBack import SwitchBack, SwitchBackTrain, SwitchBackTest
from huc.envs.locomotion.Locomotion import LocomotionRelocationTrain, LocomotionRelocationTest
from huc.envs.locomotion.Relocation import RelocationTrain
from huc.envs.locomotion.RelocationStackFrame import RelocationStackFrame

_MODES = {
    'train': 'train',
    'continual_train': 'continual_train',
    'test': 'test',
    'debug': 'debug',
    'interact': 'interact'
}


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        The custom cnn feature extractor.
        Ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # (batch_size, hidden_channels * changed_width * changed_height)
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    Ref: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class RL:
    def __init__(self, config_file):
        """
        This is the reinforcement learning pipeline where MuJoCo environments are created, and models are trained and tested.
        This pipeline is derived from my trials: context_switch.

        Args:
            config_file: the YAML configuration file that records the configurations.
        """
        # Read the configurations from the YAML file.
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        try:
            self._config_rl = config['rl']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

        # Specify the pipeline mode.
        self._mode = self._config_rl['mode']

        # Print the configuration
        if 'foveate' in self._config_rl['train']['checkpoints_folder_name']:
            print('Configuration:\n    The foveated vision is applied.')
        else:
            print('Configuration:\n    The foveated vision is NOT applied.')
        print(
            f"    The mode is: {self._config_rl['mode']} \n"
            f"    The layout name is: {self._config_rl['test']['layout_name']}"
        )
        if self._mode == _MODES['continual_train'] or self._mode == _MODES['test']:
            print(
                f"    The loaded model checkpoints folder name is: {self._config_rl['train']['checkpoints_folder_name']}\n"
                f"    The loaded model checkpoint is: {self._config_rl['test']['loaded_model_name']}\n"
            )

        # Get an env instance for further constructing parallel environments.   TODO CHANGE ENV MANUALLY!!!
        # self._env = MovingEye()
        # self._env = ReadingEye()
        # self._env = ContextSwitch()

        # if self._mode == _MODES['train'] or self._mode == _MODES['continual_train']:
        #     self._env = SwitchBackTrain()
        # else:
        #     self._env = SwitchBackTest()

        # if self._mode == _MODES['train'] or self._mode == _MODES['continual_train']:
        #     self._env = LocomotionTrain()
        # else:
        #     self._env = LocomotionTrickyTest()

        # if self._mode == _MODES['train'] or self._mode == _MODES['continual_train']:
        #     self._env = RelocationTrain()
        # else:
        #     self._env = RelocationTrain()

        # if self._mode == _MODES['train'] or self._mode == _MODES['continual_train']:
        #     self._env = LocomotionRelocationTrain()
        # else:
        #     self._env = LocomotionRelocationTest()

        self._env = RelocationStackFrame()

        # Initialise parallel environments
        self._parallel_envs = make_vec_env(
            env_id=self._env.__class__,
            n_envs=self._config_rl['train']["num_workers"],
            seed=None,
            vec_env_cls=SubprocVecEnv,
        )

        # Identify the modes and specify corresponding initiates. TODO add valueError raisers as insurance later
        # Train the model, and save the logs and modes at each checkpoints.
        if self._mode == _MODES['train']:
            # Pipeline related variables.
            self._training_logs_path = os.path.join('training', 'logs')
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
            self._models_save_file_final = os.path.join(self._models_save_path,
                                                        self._config_rl['train']['checkpoints_folder_name'])
            # RL training related variable: total time-steps.
            self._total_timesteps = self._config_rl['train']['total_timesteps']
            # Configure the model.
            # Initialise model that is run with multiple threads. TODO finalise this later
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
                activation_fn=th.nn.LeakyReLU,
                net_arch=[128, 128],
                log_std_init=-1.0,
                normalize_images=False
            )
            self._model = PPO(
                policy="CnnPolicy",
                env=self._parallel_envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self._training_logs_path,
                n_steps=self._config_rl['train']["num_steps"],
                batch_size=self._config_rl['train']["batch_size"],
                # clip_range=linear_schedule(self._config_rl['train']["clip_range"]),
                # ent_coef=self._config_rl['train']["ent_coef"],
                # n_epochs=self._config_rl['train']["n_epochs"],
                learning_rate=self._config_rl['train']["learning_rate"],
                device=self._config_rl['train']["device"],
                seed=42,
            )
        # Load the pre-trained models and test.
        elif self._mode == _MODES['test'] or self._mode == _MODES['continual_train']:
            # Pipeline related variables.
            self._loaded_model_name = self._config_rl['test']['loaded_model_name']
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
            self._loaded_model_path = os.path.join(self._models_save_path, self._loaded_model_name)
            # RL testing related variable: number of episodes and number of steps in each episodes.
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._num_steps = self._env._ep_len
            # Load the model
            if self._mode == _MODES['test']:
                self._model = PPO.load(self._loaded_model_path, self._env)
            elif self._mode == _MODES['continual_train']:
                # Logistics.
                # Pipeline related variables.
                self._training_logs_path = os.path.join('training', 'logs')
                self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
                self._models_save_path = os.path.join('training', 'saved_models', self._checkpoints_folder_name)
                self._models_save_file_final = os.path.join(self._models_save_path,
                                                            self._config_rl['train']['checkpoints_folder_name'])
                # RL training related variable: total time-steps.
                self._total_timesteps = self._config_rl['train']['total_timesteps']
                # Model loading and register.
                self._model = PPO.load(self._loaded_model_path)
                self._model.set_env(self._parallel_envs)
        # The MuJoCo environment debugs. Check whether the environment and tasks work as designed.
        elif self._mode == _MODES['debug']:
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._loaded_model_name = 'debug'
            # self._num_steps = self._env.num_steps
        # The MuJoCo environment demo display with user interactions, such as mouse interactions.
        elif self._mode == _MODES['interact']:
            pass
        else:
            pass

    def _train(self):
        """Add comments """
        # Save a checkpoint every certain steps, which is specified by the configuration file.
        # Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
        # To account for the multi-envs' steps, save_freq = max(save_freq // n_envs, 1).
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model',
        )

        # Train the RL model and save the logs. The Algorithm and policy were given,
        # but it can always be upgraded to a more flexible pipeline later.
        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=checkpoint_callback,
        )

    def _continual_train(self):
        """
        This method perform the continual trainings.
        Ref: https://github.com/hill-a/stable-baselines/issues/599#issuecomment-569393193
        """
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model_continual',
        )

        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=checkpoint_callback,
            log_interval=1,
            tb_log_name=self._config_rl['test']['continual_logs_name'],
            reset_num_timesteps=False,
        )

        # Save the model as the rear guard.
        self._model.save(self._models_save_file_final)

    def _test(self):
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.
        """
        if self._mode == _MODES['debug']:
            print('\nThe MuJoCo env and tasks baseline: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')

        imgs = []
        imgs_eye = []
        for episode in range(1, self._num_episodes + 1):
            obs = self._env.reset()
            imgs.append(self._env.render()[0])
            imgs_eye.append(self._env.render()[1])
            done = False
            score = 0
            info = None

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test']:
                    action, _states = self._model.predict(obs, deterministic=True)
                else:
                    action = 0
                obs, reward, done, info = self._env.step(action)
                imgs.append(self._env.render()[0])
                imgs_eye.append(self._env.render()[1])

                # Save img frame by frame
                if self._env._steps <= 3:
                    # print('printed the {}th frame'.format(self._env._steps))
                    # Create a folder if not exist
                    folder_name = f"C:/Users/91584/Desktop/{self._config_rl['train']['checkpoints_folder_name']}"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    path = folder_name + '/{}.png'.format(self._env._steps)

                    cv2.imwrite(path, self._env.render()[1])

                score += reward
                # progress_bar.update(1)
            # progress_bar.close()  # Tip: this line's better before any update. Or it would be split.
            print(
                f'Episode:{episode}     Score:{score} \n'
                f'--------------------------------------------------\n'
            )

        return imgs, imgs_eye

        # if self._mode == _MODES['test']:
        #     # Use the official evaluation tool.
        #     evl = evaluate_policy(self._model, self._parallel_envs, n_eval_episodes=self._num_episodes, render=False)
        #     print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

    def run(self):
        """
        This method helps run the RL pipeline.
        Call it.
        """
        # Check train or not.
        if self._mode == _MODES['train']:
            self._train()
        elif self._mode == _MODES['continual_train']:
            self._continual_train()
        elif self._mode == _MODES['test'] or self._mode == _MODES['debug']:
            # Generate the results from the pre-trained model.
            rgb_images, rgb_eye_images = self._test()
            # Write a video. First get the rgb images, then identify the path.
            # video_folder_path = f"C:/Users/91584/Desktop/{self._config_rl['train']['checkpoints_folder_name']}"
            video_folder_path = os.path.join('training', 'videos', self._config_rl['train']['checkpoints_folder_name'])
            if os.path.exists(video_folder_path) is False:
                os.makedirs(video_folder_path)
            layout_name = self._config_rl['test']['layout_name']
            video_name_prefix = self._mode + '_' + self._config_rl['train']['checkpoints_folder_name'] + '_' + self._loaded_model_name + '_' + layout_name
            video_path = os.path.join(video_folder_path, video_name_prefix + '.avi')
            write_video(
                filepath=video_path,
                fps=int(self._env._action_sample_freq),
                rgb_images=rgb_images,
                width=rgb_images[0].shape[1],
                height=rgb_images[0].shape[0],
            )
            # Write the agent's visual perception
            video_path_eye = os.path.join(video_folder_path, video_name_prefix + '_eye.avi')
            write_video(
                filepath=video_path_eye,
                fps=int(self._env._action_sample_freq),
                rgb_images=rgb_eye_images,
                width=rgb_eye_images[0].shape[1],
                height=rgb_eye_images[0].shape[0],
            )
        else:
            pass

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )

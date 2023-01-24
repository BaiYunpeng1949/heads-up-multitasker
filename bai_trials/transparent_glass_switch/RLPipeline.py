import os

import yaml
from tqdm import tqdm
import numpy as np

import gym

import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from deprecatedShowerTemp import ShowerTemp
from GlassSwitch import GlassSwitch

_BASELINE = 'baseline'
_TESTING = 'testing'


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
        # n_input_channels = observation_space.shape[0]  # TODO specify
        n_input_channels = observation_space.shape[0]   # TODO change to the multi-input later.
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
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]    # TODO change to the multi-input mode later.

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class RLPipeline:
    def __init__(self, config_file):
        """
        This is the reinforcement learning pipeline where mujoco environments are created, models are trained and tested.

        Args: (now all are embedded in the YAML file)
            model_xml_path: mujoco models' relative path.
            total_timesteps: the total timesteps.
            num_episodes: the number of episodes.
            model_name: the custom RL model name the experimenter gives for saving model or reusing model.
            to_train: the flag that determines whether the pipeline would behave: generate baseline only (testing the environment)?
                        Or training and testing the RL model, or just testing the pre-trained model.
                      Choose from: None, True, False
        """
        # Read the configurations from the YAML file.
        with open(config_file) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        self._config_rl_pipeline = configs['rl_pipeline']

        # Get an env instance for further constructing parallel environments.
        self._env = GlassSwitch()
        
        # Initialise parallel environments.
        self._parallel_envs = make_vec_env(env_id=self._env.__class__,
                                           n_envs=self._config_rl_pipeline["num_workers"],
                                           seed=None,
                                           vec_env_cls=DummyVecEnv,
                                           )  # TODO the DummyVecEnv is usually the fastest way. What is the difference with SubprocVecEnv?
                                           # env_kwargs={"simulator_folder": simulator_folder})

        self._to_train = self._config_rl_pipeline['train']
        self._checkpoints_name = self._config_rl_pipeline['checkpoints_name']
        self._model_name = self._config_rl_pipeline['model_name']
        if self._to_train is not None:
            self._log_path = os.path.join('training', 'logs')
            # self._save_path = os.path.join('training', 'saved_models', self._model_name)  TODO deprecated for having the checkpoint saver.
        self._save_path = os.path.join('training', 'saved_models', self._checkpoints_name)
        self._load_path = os.path.join(self._save_path, self._model_name)

        self._num_steps = self._config_rl_pipeline['num_steps']
        self._total_timesteps = self._config_rl_pipeline['total_timesteps']
        self._num_episodes = self._config_rl_pipeline['num_episodes']

        if self._to_train is True:  # TODO automate the device later.
            # self._model = PPO(self._config_rl_pipeline['policy_type'], self._env, verbose=1,
            #                   tensorboard_log=self._log_path, device=self._config_rl_pipeline['device'])

            # Initialise model that is run with multiple threads. TODO finalise this later
            policy_kwargs = dict(   # TODO regulate later
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
            )
            self._model = PPO(policy=self._config_rl_pipeline["policy_type"],
                              env=self._parallel_envs,
                              verbose=1,
                              policy_kwargs=policy_kwargs, #self._config_rl_pipeline["policy_kwargs"], # TODO maybe use it later.
                              tensorboard_log=self._log_path,
                              n_steps=self._config_rl_pipeline["num_steps"],
                              batch_size=self._config_rl_pipeline["batch_size"],
                              device=self._config_rl_pipeline["device"])

        elif self._to_train is False:
            self._model = PPO.load(self._load_path, self._env)
        elif self._to_train is None:
            pass

    def _generate_results(self, mode):
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.

        Args:
            mode: the RL model testing mode or the baseline generating mode. Choose from: 'baseline' or 'testing'.
        """
        if mode == _BASELINE:
            print('\nThe baselines: ')
        elif mode == _TESTING:
            print('\nThe testing: ')

        for episode in range(1, self._num_episodes + 1):
            obs = self._env.reset()
            done = False
            score = 0
            progress_bar = tqdm(total=self._num_steps)
            while not done:
                if mode == _BASELINE:
                    action = self._env.action_space.sample()
                elif mode == _TESTING:
                    action, _states = self._model.predict(obs)
                obs, reward, done, info = self._env.step(action)
                score += reward
                progress_bar.update(1)
            progress_bar.close()    # Tip: this line's better before any update. Or it would be split.
            print('Episode:{}   Score:{}    Score Pct: {}%   '
                  '\nInfo details: '
                  '\n   optimal score: {}    num_on_glass: {}     num_on_env: {}'
                  '\n   glass B:   total time: {}     on time: {}     miss time: {}     on/miss pct: {}%'
                  '\n   env red:   total time: {}     on time: {}     miss time: {}     on/miss pct: {}%'    
                  '\n   glass X:   total time: {}     on time: {}     miss time: {}     on/miss pct: {}%'
                  '\n   total intermediate time: {}'
                  .format(episode, score, np.round(100*score/info['optimal_score'], 2),
                          info['optimal_score'], info['num_on_glass'], info['num_on_env'],
                          info['total_time_glass_B'], info['total_time_on_glass_B'], info['total_time_miss_glass_B'], np.round(100*info['total_time_miss_glass_B']/info['total_time_on_glass_B'], 2),
                          info['total_time_env_red'], info['total_time_on_env_red'], info['total_time_miss_env_red'], np.round(100*info['total_time_miss_env_red']/info['total_time_on_env_red'], 2),
                          info['total_time_glass_X'], info['total_time_on_glass_X'], info['total_time_miss_glass_X'], np.round(100*info['total_time_miss_glass_X']/info['total_time_on_glass_X'], 2),
                          info['total_time_intermediate']))

        if mode == _TESTING:
            # Use the official evaluation tool.
            evl = evaluate_policy(self._model, self._parallel_envs, n_eval_episodes=self._num_episodes, render=False)
            print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

    def _train(self):
        """Add comments """
        # Save a checkpoint every certain steps, which is specified by the configuration file.
        # Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
        # To account for the multi-envs' steps, save_freq = max(save_freq // n_envs, 1).
        save_freq = max(self._config_rl_pipeline['save_freq'] // self._config_rl_pipeline['num_workers'], 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._save_path,
            name_prefix='rl_model',
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # Train the RL model and save the logs. The Algorithm and policy were given,
        # but it can always be upgraded to a more flexible pipeline later.
        self._model.learn(total_timesteps=self._total_timesteps,
                          callback=checkpoint_callback,
                          progress_bar=True)

        # # Save the model.
        # self._model.save(self._save_path)
        # print('The model has been saved as: {} in {}'.format(self._model_name, self._save_path))

    def run(self):
        """
        This method helps run the RL pipeline.
        Call it.
        """
        # Check train or not.
        if self._to_train is True:
            self._train()
        elif self._to_train is False:
            # Generate the baseline.
            # TODO comment the baseline generation out because this is a deterministic problem, only apply once: -17.51%
            # self._generate_results(mode=_BASELINE)
            # Generate the results from the pre-trained model.
            self._generate_results(mode=_TESTING)
            # Write a video.
            video_folder_path = os.path.join('training', 'videos')
            if os.path.exists(video_folder_path) is False:
                os.makedirs(video_folder_path)
            video_path = os.path.join(video_folder_path, self._model_name + '.avi')
            self._env.write_video(filepath=video_path)
        else:
            # Generate the baseline.
            self._generate_results(mode=_BASELINE)

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print('***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************')

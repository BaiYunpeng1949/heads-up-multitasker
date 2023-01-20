import os

import yaml
from tqdm import tqdm
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from deprecatedShowerTemp import ShowerTemp
from GlassSwitch import GlassSwitch

_BASELINE = 'baseline'
_TESTING = 'testing'


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

        self._env = GlassSwitch(configs=configs)

        self._to_train = self._config_rl_pipeline['train']

        self._model_name = self._config_rl_pipeline['model_name']
        if self._to_train is not None:
            self._log_path = os.path.join('training', 'logs')
            self._save_path = os.path.join('training', 'saved_models', self._model_name)

        self._total_timesteps = self._config_rl_pipeline['total_timesteps']
        self._num_episodes = self._config_rl_pipeline['num_episodes']

        if self._to_train is True:  # TODO automate the device later.
            self._model = PPO(self._config_rl_pipeline['policy_type'], self._env, verbose=1, tensorboard_log=self._log_path,
                              device='cuda')
            # TODO the policy needs to be changed, e.g., MultiInputPolicy, when dealing with higher dimension observation space.
        elif self._to_train is False:
            self._model = PPO.load(self._save_path, self._env)
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
            progress_bar = tqdm(total=self._total_timesteps)
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

        # if mode == _TESTING:
        #     # Use the official evaluation tool.
        #     evl = evaluate_policy(self._model, self._env, n_eval_episodes=self._num_episodes, render=False)
        #     print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

    def _train(self):
        """Add comments """
        # Train the RL model and save the logs. The Algorithm and policy were given,
        # but it can always be upgraded to a more flexible pipeline later.
        self._model.learn(total_timesteps=self._total_timesteps)

        # Save the model.
        self._model.save(self._save_path)
        print('The model has been saved as: {} in {}'.format(self._model_name, self._save_path))

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
        print('***************************** RL pipeline ends. The mujoco environment of the pipeline has been destructed *************************************')

import os

import yaml
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from ShowerTemp import ShowerTemp
from GlassSwitch import GlassSwitch

_BASELINE = 'baseline'
_TESTING = 'testing'


class RLPipeline:
    def __init__(self, config):
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
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self._config_rl_pipeline = config['rl_pipeline']

        self._env = GlassSwitch(config=config)

        self._to_train = self._config_rl_pipeline['train']

        self._model_name = self._config_rl_pipeline['model_name']
        if self._to_train is not None:
            self._log_path = os.path.join('training', 'logs')
            self._save_path = os.path.join('training', 'saved_models', self._model_name)

        self._total_timesteps = self._config_rl_pipeline['total_timesteps']
        self._num_episodes = self._config_rl_pipeline['num_episodes']

        if self._to_train is True:
            self._model = PPO(self._config_rl_pipeline['policy_type'], self._env, verbose=1, tensorboard_log=self._log_path)
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
            print('Episode:{}   Score:{}   Info: {}'.format(episode, score, info))


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

    def run(self):
        """
        This method helps run the RL pipeline.
        Call it.
        """
        # Run the pipeline.
        self._generate_results(mode=_BASELINE)

        # Check train or not.
        if self._to_train is True:
            self._train()

        # Display the results.
        if self._to_train is not None:
            self._generate_results(mode=_TESTING)

            # Write a video.
            video_path = os.path.join('training', 'videos', self._model_name + '.avi')
            self._env.write_video(filepath=video_path)

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print('***************************** RL pipeline ends. The mujoco environment of the pipeline has been destructed *************************************')

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from BallSwitch import BallSwitch


class RLPipeline:
    def __init__(self, ball_switch_model_xml_path, total_timesteps, num_episodes, model_name, to_train):
        self._env = BallSwitch(model_xml_path=ball_switch_model_xml_path)

        self._log_path = os.path.join('training', 'logs')
        self._save_path = os.path.join('training', 'saved_models', model_name)

        self._to_train = to_train

        self._total_timesteps = total_timesteps
        self._num_episodes = num_episodes

        if self._to_train is True:
            self._model = PPO("MultiInputPolicy", self._env, verbose=1, tensorboard_log=self._log_path)
        elif self._to_train is False:
            self._model = PPO.load(self._save_path, self._env)

    def _generate_baseline(self):
        for episode in range(1, self._num_episodes + 1):
            state = self._env.reset()
            done = False
            score = 0
            while not done:
                action = self._env.action_space.sample()
                n_state, reward, done, info = self._env.step(action)
                score += reward
            print('Episode:{}   Score:{}   Info: {}'.format(episode, score, info))
        self._env.close()

    def _train(self):
        # Train the RL model and save the logs. The Algorithm and policy were given,
        # but it can always be upgraded to a more flexible pipeline later.
        self._model = PPO("MultiInputPolicy", self._env, verbose=1, tensorboard_log=self._log_path)
        self._model.learn(total_timesteps=self._total_timesteps)

        # Save the model.
        self._model.save(self._save_path)

    def _test(self):
        # Test and evaluate the effect of RL.
        for episode in range(1, self._num_episodes + 1):
            obs = self._env.reset()
            done = False
            score = 0
            while not done:
                action, _states = self._model.predict(obs)
                obs, reward, done, info = self._env.step(action)
                score += reward
            print('Episode:{}   Score:{}   Info: {}'.format(episode, score, info))

        # Use the official evaluation tool.
        evl = evaluate_policy(self._model, self._env, n_eval_episodes=self._num_episodes, render=False)
        print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

        self._env.close()

    def run(self):
        # Run the pipeline.
        self._generate_baseline()

        # Check train or not.
        if self._to_train:
            self._train()

        # Display the results.
        self._test()

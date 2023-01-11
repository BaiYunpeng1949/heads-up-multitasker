import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from ShowerTemp import ShowerTemp
from GlassSwitch import GlassSwitch
from RLPipeline import RLPipeline

# Load the 'mujoco' environment model.
glass_switch_path = os.path.join('glass-switch-env-v1224.xml')

# env = GlassSwitch(xml_path=glass_switch_path)
# _num_episodes = 1
# for episode in range(1, _num_episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{}   Score:{}   Info: {}'.format(episode, score, info))
# env.close()

# Train in the RL.
rl_pipeline = RLPipeline(model_xml_path=glass_switch_path,
                         total_timesteps=10000,
                         num_episodes=3,
                         model_name='glass_switch_10k',
                         to_train=True)     # None for just baselines, True for train and test, False for just test.
rl_pipeline.run()
# TODO fix the UserWarning Evaluation environment is not wrapped with a ``Monitor`` wrapper. This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. Consider wrapping environment first with ``Monitor`` wrapper.
# TODO make a progress visualizer later.

# env = ShowerTemp()
#
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=20000)
#
# evl = evaluate_policy(model, env, n_eval_episodes=10, render=False)
# print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

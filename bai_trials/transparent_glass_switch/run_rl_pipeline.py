import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from ShowerTemp import ShowerTemp
from GlassSwitch import GlassSwitch
from RLPipeline import RLPipeline

# Load the 'mujoco' environment model.
glass_switch_path = os.path.join('glass-switch-env-v1224.xml')

env = GlassSwitch(xml_path=glass_switch_path)
env.render()
env.close()

# Train in the RL.
# rl_pipeline = RLPipeline(glass_switch_model_xml_path=glass_switch_path,
#                        total_timesteps=20000,
#                        num_episodes=10,
#                        model_name='shower_temp_200k',
#                        to_train=True)
# rl_pipeline.run()


# env = ShowerTemp()
#
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=20000)
#
# evl = evaluate_policy(model, env, n_eval_episodes=10, render=False)
# print('The evaluation results are: Mean {}; STD {}'.format(evl[0], evl[1]))

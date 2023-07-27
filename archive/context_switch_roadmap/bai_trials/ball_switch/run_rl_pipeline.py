import os

from RLPipeline import RLPipeline

from BallSwitch import BallSwitch
from stable_baselines3 import PPO

# Load the 'mujoco' environment model.
ball_switch_path = os.path.join('ball_switch-v1223.xml')

# Train in the RL.
rl_pipeline = RLPipeline(ball_switch_model_xml_path=ball_switch_path,
                   total_timesteps=400000,
                   num_episodes=5,
                   model_name='ppo_model_1220_400k',
                   to_train=False)
rl_pipeline.run()

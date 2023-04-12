import os
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from huc.envs.context_switch_replication.SwitchBackLSTM import SwitchBackLSTM
from huc.utils.write_video import write_video

env = SwitchBackLSTM()

model = RecurrentPPO.load("training/saved_models/test/rl_model_50000_steps.zip")

imgs = []
imgs_eye = []

num_episodes = 1
for episode in range(num_episodes):

  obs = env.reset()
  imgs.append(env.render()[0])
  imgs_eye.append(env.render()[1])
  done = False
  score = 0
  info = None

  lstm_states = None
  num_envs = 1
  episode_starts = np.ones((num_envs,), dtype=bool)

  while not done:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
    obs, reward, done, info = env.step(action)
    episode_starts = done
    imgs.append(env.render()[0])
    imgs_eye.append(env.render()[1])
    score += reward

  print(f"sum of rewards: {score}")

video_folder_path = os.path.join('training', 'videos')
if os.path.exists(video_folder_path) is False:
  os.makedirs(video_folder_path)
video_path = os.path.join(video_folder_path, 'env_camera.avi')

write_video(
  filepath=video_path,
  fps=int(env._action_sample_freq),
  rgb_images=imgs,
  width=imgs[0].shape[1],
  height=imgs[0].shape[0],
)

# Write the agent's visual perception
video_path_eye = os.path.join(video_folder_path, 'eye_camera.avi')
write_video(
  filepath=video_path_eye,
  fps=int(env._action_sample_freq),
  rgb_images=imgs_eye,
  width=imgs_eye[0].shape[1],
  height=imgs_eye[0].shape[0],
)

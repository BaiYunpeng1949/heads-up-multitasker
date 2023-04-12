import yaml
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import torch

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

from huc.envs.context_switch_replication.SwitchBackLSTM import SwitchBackLSTM
from huc.RL import CustomCombinedExtractor
from huc.sb3.recurrent_policies import RecurrentMultiInputActorCriticPolicyTanhActions

if __name__ == "__main__":

  # Read the configurations from the YAML file.
  config_file='config.yaml'
  with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config_rl = config['rl']

  # Initialise parallel environments
  env = SwitchBackLSTM()
  parallel_envs = make_vec_env(
    env_id=env.__class__,
    n_envs=config_rl['train']["num_workers"],
    seed=None,
    vec_env_cls=SubprocVecEnv,
  )

  # Pipeline related variables.
  training_logs_path = os.path.join('training', 'logs')
  checkpoints_folder_name = config_rl['train']['checkpoints_folder_name']
  models_save_path = os.path.join('training', 'saved_models', checkpoints_folder_name)
  models_save_file_final = os.path.join(models_save_path, config_rl['train']['checkpoints_folder_name'])

  # RL training related variable: total time-steps.
  total_timesteps = config_rl['train']['total_timesteps']

  # Configure the model.
  # Initialise model that is run with multiple threads.
  policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(vision_features_dim=128, proprioception_features_dim=32),
    activation_fn=torch.nn.LeakyReLU,
    net_arch=[256, 256],
    log_std_init=-1.0,
    normalize_images=False,
    lstm_hidden_size=32,
    n_lstm_layers=6
  )
  model = RecurrentPPO(
    policy=RecurrentMultiInputActorCriticPolicyTanhActions,
    env=parallel_envs,
    # env=env,
    verbose=1,
    learning_rate=config_rl['train']["learning_rate"],
    target_kl=config_rl['train']["target_kl"],
    policy_kwargs=policy_kwargs,
    tensorboard_log=training_logs_path,
    n_steps=config_rl['train']["num_steps"],
    batch_size=config_rl['train']["batch_size"],
    device=config_rl['train']["device"]
  )

  # Save a checkpoint every certain steps, which is specified by the configuration file.
  # Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
  # To account for the multi-envs' steps, save_freq = max(save_freq // n_envs, 1).
  save_freq = config_rl['train']['save_freq']
  n_envs = config_rl['train']['num_workers']
  save_freq = max(save_freq // n_envs, 1)
  checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=models_save_path,
    name_prefix='rl_model',
  )

  # Initialise wandb
  run = wandb.init(project="huc", name=checkpoints_folder_name, config=config, sync_tensorboard=True, save_code=True,
                   dir=os.path.dirname(os.path.realpath(__file__)))

  # Train the RL model and save the logs. The Algorithm and policy were given,
  # but it can always be upgraded to a more flexible pipeline later.
  model.learn(
    total_timesteps=total_timesteps,
    callback=[WandbCallback(verbose=2), checkpoint_callback],
  )
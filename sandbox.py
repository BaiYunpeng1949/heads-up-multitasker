from huc.envs.moving_eye.MovingEye import MovingEye
import matplotlib.pyplot as pp

env = MovingEye()

env.reset()

for idx in range(10000):

  # Take random actions
  action = env.action_space.sample()

  # Step the env
  obs, reward, terminate, info = env.step(action)

  if reward == 1:
    pp.imshow(obs)
    pp.show()

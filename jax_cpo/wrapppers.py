from gym import Wrapper


class ActionRepeat(Wrapper):

  def __init__(self, env, repeat):
    assert repeat >= 1, 'Expects at least one repeat.'
    super(ActionRepeat, self).__init__(env)
    self.repeat = repeat

  def step(self, action):
    done = False
    total_reward = 0.0
    current_step = 0
    info = {'steps': 0}
    while current_step < self.repeat and not done:
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      current_step += 1
    info['steps'] = current_step
    return obs, total_reward, done, info  # noqa

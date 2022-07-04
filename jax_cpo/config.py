import os

from typing import Optional, List, AnyStr

import jax_cpo

BASE_PATH = os.path.join(os.path.dirname(jax_cpo.__file__))


def validate_config(config):
  assert config.time_limit % config.action_repeat == 0, ('Action repeat '
                                                         'should '
                                                         ''
                                                         'be a factor of time '
                                                         ''
                                                         'limit')
  return config


# Acknowledgement: https://github.com/danijar
def load_config(args: Optional[List[AnyStr]] = None):
  import argparse
  import ruamel.yaml as yaml
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['defaults'])
  # Load all config parameters and infer their types and default values.
  args, remaining = parser.parse_known_args(args)
  with open(os.path.join(BASE_PATH, 'configs.yaml')) as file:
    configs = yaml.safe_load(file)
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  updated_remaining = []
  # Collect all the user inputs that override the default parameters.
  for idx in range(0, len(remaining), 2):
    stripped = remaining[idx].strip('-')
    # Allow the user to override specific values within dictionaries.
    if '.' in stripped:
      params_group, key = stripped.split('.')
      # Override the default value within a dictionary.
      defaults[params_group][key] = yaml.safe_load(remaining[idx + 1])
    else:
      updated_remaining.append(remaining[idx])
      updated_remaining.append(remaining[idx + 1])
  remaining = updated_remaining
  parser = argparse.ArgumentParser()
  # Add arguments from the defaults to create the default parameters namespace.
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    parser.add_argument(f'--{key}', type=yaml.safe_load, default=value)
  # Parse the remaining arguments into the parameters' namespace.
  return validate_config(parser.parse_args(remaining))

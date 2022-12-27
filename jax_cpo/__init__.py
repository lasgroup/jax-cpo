import logging as lg
import os
import warnings
if 'LOG' not in os.environ:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  lg.getLogger().setLevel('ERROR')
  warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=UserWarning)




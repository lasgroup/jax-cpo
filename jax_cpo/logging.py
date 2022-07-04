import json
import os
from collections import defaultdict
from queue import Queue
from threading import Thread
from typing import Optional

import cloudpickle
import numpy as np
from tensorboardX import SummaryWriter
from tensorflow import metrics


class TrainingLogger:

  def __init__(self, log_dir):
    self._writer = SummaryWriter(log_dir, flush_secs=60)
    self._metrics = defaultdict(metrics.Mean)
    self.step = 0
    self.log_dir = log_dir

  def __getitem__(self, item: str):
    return self._metrics[item]

  def __setitem__(self, key: str, value: float):
    self._metrics[key].update_state(value)

  def flush(self):
    self._writer.flush()

  def log_summary(self,
                  summary: dict,
                  step: Optional[int] = None,
                  flush: bool = False):
    step = step if step is not None else self.step
    for k, v in summary.items():
      self._writer.add_scalar(k, float(v), step)
    with open(os.path.join(self.log_dir, 'summary.json'), 'a') as file:
      file.write(json.dumps({'step': step, **summary}) + '\n')
    if flush:
      self._writer.flush()

  def log_metrics(self, step: Optional[int] = None, flush: bool = False):
    step = step if step is not None else self.step
    print("\n----Training step {} summary----".format(step))
    for k, v in self._metrics.items():
      val = float(v.result())
      print("{:<40} {:<.4f}".format(k, val))
      self._writer.add_scalar(k, val, step)
      v.reset_states()
    if flush:
      self._writer.flush()

  def log_video(self,
                images,
                name='policy',
                fps=30,
                step: Optional[int] = None,
                flush: bool = False):
    step = step if step is not None else self.step
    # (N, T, C, H, W)
    self._writer.add_video(
        name,
        np.array(images, copy=False).transpose([0, 1, 4, 2, 3]),
        step,
        fps=fps)
    if flush:
      self._writer.flush()

  def log_figure(self, figure, name='policy'):
    self._writer.add_figure(name, figure, self.step)
    self._writer.flush()

  def __getstate__(self):
    self._writer.close()
    self._metrics.clear()
    state = self.__dict__.copy()
    del state['_writer']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._writer = SummaryWriter(self.log_dir)

  def close(self):
    self._writer.close()


class StateWriter:

  def __init__(self, log_dir: str):
    self.log_dir = log_dir
    self.queue = Queue(maxsize=5)
    self._thread = Thread(name="state_writer", target=self._worker)
    self._thread.start()

  def write(self, data: dict):
    state_bytes = cloudpickle.dumps(data)
    self.queue.put(state_bytes)
    # Lazily open up a thread and let it drain the work queue. Thread exits
    # when there's no more work to do.
    if not self._thread.is_alive():
      self._thread = Thread(name="state_writer", target=self._worker)
      self._thread.start()

  def _worker(self):
    while not self.queue.empty():
      state_bytes = self.queue.get(timeout=1)
      with open(os.path.join(self.log_dir, 'state.pkl'), 'wb') as f:
        f.write(state_bytes)
        self.queue.task_done()

  def close(self):
    self.queue.join()
    if self._thread.is_alive():
      self._thread.join()

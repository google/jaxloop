# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The loop with some statistics library for JAX models."""

import time
from typing import Any, Iterator, Optional, Sequence, Tuple

from jaxloop import loop
from jaxloop import step as step_lib
from jaxloop import types

Loop = loop.Loop
Output = types.Output
State = types.TrainState
Step = step_lib.Step

STAT_PREFIX = 'jaxloop'
STAT_NUM_PARAMS = f'{STAT_PREFIX}/num_params'
STAT_NUM_FLOPS = f'{STAT_PREFIX}/num_flops'
STAT_INIT_TIME = f'{STAT_PREFIX}/init_time'
STAT_BEGIN_TIME = f'{STAT_PREFIX}/begin_time'
STAT_LIFE_TIME_SECS = f'{STAT_PREFIX}/life_time_secs'
STAT_LOOP_TIME_SECS = f'{STAT_PREFIX}/loop_time_secs'
STAT_STEPS_PER_SEC = f'{STAT_PREFIX}/steps_per_sec'


class StatLoop(Loop):
  """The loop with some statistics.

  This class extends the base loop class and adds some statistics like number of
  model parameters and flops to the loop outputs.
  """

  def __init__(
      self,
      step: Step,
      stat_names: Optional[Sequence[str]] = None,
  ):
    super().__init__(step)
    self._stat_names = stat_names or []
    self._init_time = time.time()
    self._begin_time = None
    self._begin_step = None

  def begin(
      self, state: State, dataset: Iterator[Any]
  ) -> Tuple[State, Iterator[Any]]:
    """Begins the loop.

    Args:
      state: The model state.
      dataset: The dataset iterator.

    Returns:
      A tuple of the model state and output.
    """
    if any([
        STAT_BEGIN_TIME in self._stat_names,
        STAT_LOOP_TIME_SECS in self._stat_names,
        STAT_STEPS_PER_SEC in self._stat_names,
    ]):
      self._begin_time = time.time()
    if STAT_STEPS_PER_SEC in self._stat_names:
      self._begin_step = int(state.step)
    return super().begin(state, dataset)

  def run(
      self,
      state: State,
      dataset: Iterator[Any],
      num_steps: Optional[int] = None,
      log_num_flops: bool = False,
      **kwargs
  ) -> Tuple[State, Optional[Output]]:
    """Runs the loop.

    This method should be overridden if custom `run` logic is needed.

    Args:
      state: The model state.
      dataset: The dataset iterator.
      num_steps: The number of steps to run.
      log_num_flops: Whether to log the number of flops of the step function.
      **kwargs: Additional keyword arguments for step function.

    Returns:
      A tuple of the model state and output.
    """
    log_num_flops = log_num_flops or STAT_NUM_FLOPS in self._stat_names
    return super().run(state, dataset, num_steps, log_num_flops, **kwargs)

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    """Ends the loop by adding the statistics to the outputs.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    outputs = outputs or {}
    if STAT_NUM_PARAMS in self._stat_names:
      if self._step.num_params is None:
        self._step.compute_num_params(state)
      outputs[STAT_NUM_PARAMS] = self._step.num_params
    if STAT_NUM_FLOPS in self._stat_names:
      outputs[STAT_NUM_FLOPS] = self._step.num_flops or 0
    if STAT_INIT_TIME in self._stat_names:
      outputs[STAT_INIT_TIME] = self._init_time
    if STAT_BEGIN_TIME in self._stat_names:
      outputs[STAT_BEGIN_TIME] = self._begin_time
    if STAT_LIFE_TIME_SECS in self._stat_names:
      outputs[STAT_LIFE_TIME_SECS] = time.time() - self._init_time
    if (STAT_LOOP_TIME_SECS in self._stat_names or
        STAT_STEPS_PER_SEC in self._stat_names):
      runtime_secs = time.time() - self._begin_time
      if STAT_LOOP_TIME_SECS in self._stat_names:
        outputs[STAT_LOOP_TIME_SECS] = runtime_secs
      if STAT_STEPS_PER_SEC in self._stat_names:
        outputs[STAT_STEPS_PER_SEC] = (
            (int(state.step) - self._begin_step) / runtime_secs
        )
    return super().end(state, outputs)

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

"""The loop library for JAX models."""

import collections
from typing import Any, Iterator, Optional, Tuple

from jaxloop import step as step_lib
from jaxloop import types


Output = types.Output
State = types.TrainState
Step = step_lib.Step


class Loop:
  """The loop class for JAX models.

  This is the base class for running steps repeatedly in a loop. More
  functionalities can be added by extending this class.
  """

  def __init__(self, step: Step):
    self._step = step

  def begin(
      self, state: State, dataset: Iterator[Any]
  ) -> Tuple[State, Iterator[Any]]:
    """Begins the loop.

    This method should be overridden if custom `begin` logic is needed.

    Args:
      state: The model state.
      dataset: The dataset iterator.

    Returns:
      A tuple of the model state and dataset iterator.
    """
    return state, dataset

  def update_outputs(
      self, loop_outputs: Output, step_outputs: Optional[Output]) -> Output:
    """Updates the outputs of the loop.

    This method should be overridden if custom `update_outputs` logic is needed.

    Args:
      loop_outputs: The output of the loop.
      step_outputs: The output of the step.

    Returns:
      The updated output.
    """
    if step_outputs is not None:
      for key, value in step_outputs.items():
        loop_outputs[key].append(value)
    return loop_outputs

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
    step = 0
    loop_outputs = collections.defaultdict(list)
    for batch in dataset:
      log_num_flops = log_num_flops and step == 0
      state, step_outputs = self._step(
          state, batch, log_num_flops=log_num_flops, **kwargs
      )
      loop_outputs = self.update_outputs(loop_outputs, step_outputs)
      step += 1
      if num_steps is not None and step >= num_steps:
        break
    return state, loop_outputs

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    """Ends the loop.

    This logic should be implemented if custom `end` logic is needed.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    return state, outputs

  def __call__(
      self,
      state: State,
      dataset: Iterator[Any],
      num_steps: Optional[int] = None,
      **kwargs
  ) -> Tuple[State, Optional[Output]]:
    """Invokes the loop.

    This method invokes the loop by running the `begin`, `run` and `end` methods
    in order. This method should generally not be overridden and always be used
    to invoke the loop.

    Args:
      state: The model state.
      dataset: The dataset iterator.
      num_steps: The number of steps to run.
      **kwargs: Additional keyword arguments for step function.

    Returns:
      A tuple of the model state and output.
    """
    state, dataset = self.begin(state, dataset)
    state, outputs = self.run(state, dataset, num_steps, **kwargs)
    return self.end(state, outputs)

  @property
  def step(self) -> Step:
    return self._step

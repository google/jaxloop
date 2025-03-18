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

"""The action library for JAX models."""

import typing
from typing import Any, Callable, Optional, Protocol

from absl import logging
from clu import metric_writers
import jax
from jaxloop import types
from jaxloop.step_number_writer import step_number_writer
import jaxtyping
import orbax.checkpoint as ocp


Output = types.Output
State = types.TrainState
Scalar = types.Scalar
PyTree = jaxtyping.PyTree
ApplyFn = Callable[[PyTree, PyTree], PyTree]


@typing.runtime_checkable
class Action(Protocol):
  """Base class for actions.

  Action is an operation to be performed periodically in the training loop,
  evaluation loop, or step.
  """

  def __init__(self, interval: int = 1):
    """Initializes the action.

    Args:
      interval: The interval of the action. It should be a positive integer. The
        action will be invoked every `interval` inner loops or steps, depending
        on if the value is passed to a `Loop` or `Step`. Default is 1.
    """
    super().__init__()
    if interval <= 0:
      raise ValueError('`interval` must be positive.')
    self._interval = interval

  def __call__(
      self, state: State, outputs: Optional[Output], **kwargs
  ) -> Optional[Any]:
    """Invokes the action.

    When used in the context of Jaxloop, the return value is ignored;
    however, using an Action outside of the context of a Jaxloop may require
    a returned value, so a return value is allowed.

    Args:
      state: The model state.
      outputs: The model output.
      **kwargs: Additional keyword arguments for the action.

    Returns:
      Optional return value or None.
    """
    raise NotImplementedError('Action\'s `__call__` method is not implemented.')

  @property
  def interval(self) -> int:
    return self._interval


class SummaryAction(Action):
  """Summary action."""

  def __init__(
      self, summary_writer: metric_writers.MetricWriter, interval: int = 1
  ):
    super().__init__(interval=interval)
    self._summary_writer = summary_writer

  @property
  def summary_writer(self) -> metric_writers.MetricWriter:
    return self._summary_writer

  def __call__(self, state: State, outputs: Optional[Output], **kwargs) -> None:
    step = int(state.step)
    self._summary_writer.write_scalars(step, outputs)


class CheckpointAction(Action):
  """Checkpointing action."""

  def __init__(self, ckpt_manager: ocp.CheckpointManager, interval: int = 1):
    super().__init__(interval=interval)
    self._ckpt_manager = ckpt_manager
    self._data_transfer_fn = jax.device_get

  @property
  def ckpt_manager(self) -> ocp.CheckpointManager:
    return self._ckpt_manager

  def set_data_transfer_fn(self, data_transfer_fn: Callable[[Any], Any]):
    self._data_transfer_fn = data_transfer_fn

  def save(self, step: int, state: State, outputs: Optional[Output]) -> bool:
    # Move `train_state` and `metrics` to host before doing async checkpointing.
    # This saves HBM usage during checkpointing.
    # Otherwise, async checkpointing will fail when jax.jit output is donated.
    return self._ckpt_manager.save(
        step,
        metrics=self._data_transfer_fn(outputs),
        items=self._data_transfer_fn(state),
    )

  def __call__(self, state: State, outputs: Optional[Output], **kwargs) -> None:
    step = int(state.step)
    succeeded = self.save(
        step,
        state,
        outputs,
    )
    if succeeded:
      logging.info(
          'Saved checkpoint to %s on step %d.',
          self._ckpt_manager.directory,
          step,
      )
    else:
      logging.error(
          'Saving checkpoint to %s on step %d failed.',
          self._ckpt_manager.directory,
          step,
      )


class StepNumberAction(Action):
  """Action to write the step number."""

  def __init__(
      self, step_writer: step_number_writer.StepNumberWriter, interval: int = 1
  ):
    super().__init__(interval=interval)
    self._step_number_writer = step_writer

  def __call__(
      self, state: State, _: Optional[Output] = None, **kwargs
  ) -> None:
    step = int(state.step)
    self._step_number_writer.write(step)

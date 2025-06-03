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

import collections
import typing
from typing import Any, Callable, Optional, Protocol
from collections.abc import Mapping

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

MetricWithMetadata = types.MetricWithMetadata
MetricType = types.MetricType


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
      raise ValueError("`interval` must be positive.")
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
    raise NotImplementedError("Action's `__call__` method is not implemented.")

  @property
  def interval(self) -> int:
    return self._interval


class SummaryAction(Action):
  """Summary action.

  Only the types included in the supported_types set are supported by the
  summary action.
  """

  supported_types = {
      MetricType.SCALAR,
      MetricType.HISTOGRAM,
      MetricType.IMAGE,
      MetricType.VIDEO,
      MetricType.TEXT,
  }

  def __init__(
      self,
      summary_writer: metric_writers.MetricWriter,
      interval: int = 1,
      flush_each_call: bool = False,
  ):
    super().__init__(interval=interval)
    self._flush_each_call = flush_each_call
    self._summary_writer = summary_writer

  @property
  def summary_writer(self) -> metric_writers.MetricWriter:
    return self._summary_writer

  def split_outputs(
      self, outputs: Output | None
  ) -> Mapping[MetricType, dict[str, Any]]:
    """Splits the outputs by type for dispatching to different writing methods.

    Args:
      outputs: The model output.

    Returns:
      A dictionary of outputs, split by metric type.
    """
    if outputs is None:
      return {}

    outputs_by_type = collections.defaultdict(dict)
    for key, value in outputs.items():
      # This means it is legacy usage with bare type.
      if not isinstance(value, MetricWithMetadata):
        outputs_by_type[MetricType.SCALAR][key] = value
      else:
        # Not all types are supported at the moment.
        if value.type not in self.supported_types:
          raise ValueError(
              f"Metric type {value.type} is not supported by the summary"
              f" writer, supported types are: {self.supported_types}"
          )
        outputs_by_type[value.type][key] = value.value
    return outputs_by_type

  def __call__(self, state: State, outputs: Optional[Output], **kwargs) -> None:
    """Writes the data to the summary writer."""

    step = int(state.step)
    # We write the data according to the type, using a different writing
    # method for each of the supported types.
    for metric_type, data in self.split_outputs(outputs).items():
      if metric_type == MetricType.SCALAR:
        self._summary_writer.write_scalars(step, data)
      elif metric_type == MetricType.HISTOGRAM:
        self._summary_writer.write_histograms(step, data)
      elif metric_type == MetricType.IMAGE:
        self._summary_writer.write_images(step, data)
      elif metric_type == MetricType.VIDEO:
        self._summary_writer.write_videos(step, data)
      elif metric_type == MetricType.TEXT:
        self._summary_writer.write_texts(step, data)
      else:
        raise ValueError(
            f"Split outputs contains unsupported metric type: {metric_type}"
        )
    if self._flush_each_call:
      self._summary_writer.flush()


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
        args=ocp.args.PyTreeSave(self._data_transfer_fn(state)),
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
          "Saved checkpoint to %s on step %d.",
          self._ckpt_manager.directory,
          step,
      )
    else:
      logging.error(
          "Saving checkpoint to %s on step %d failed.",
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

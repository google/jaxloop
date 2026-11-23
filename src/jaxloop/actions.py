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
from collections.abc import Mapping
import math
import typing
from typing import Any, Callable, Optional, Protocol

from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
from jaxloop import stop_handler
from jaxloop import types
from jaxloop.step_number_writer import step_number_writer
import jaxtyping
import orbax.checkpoint as ocp


MetricMonitoringStopHandler = stop_handler.MetricMonitoringStopHandler
StopTrainingError = stop_handler.StopTrainingError
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
    # TODO(varunsng): Due to downstream waymo dependencies, we need to keep
    # this function as a class attribute. Remove this once the dependencies are
    # removed.
    self._data_transfer_fn = lambda x: x

  @property
  def ckpt_manager(self) -> ocp.CheckpointManager:
    return self._ckpt_manager

  def save(self, step: int, state: State, outputs: Optional[Output]) -> bool:
    return self._ckpt_manager.save(
        step,
        metrics=outputs,
        args=ocp.args.PyTreeSave(state),
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


class ContinuousCheckpointAction(CheckpointAction):
  """Calls `CheckpointManager.save()` at the end of every step.

  Use it as a `Step` end action. The manager's save decision policies decide
  when to save; build them with
  `ocp.checkpoint_managers.continuous_checkpointing_options()`. Steps that
  don't save don't wait for the device. `OuterLoop` calls `finalize()` at the
  end of training.

  The action must run once per step (`Step` end action with `interval=1`), and
  `state.step` must grow by one per call. It counts steps on the host and reads
  `state.step` only on save steps, so a caller that skips steps can miss target
  steps.

  Give the action its own `CheckpointManager`. `OuterLoop` raises an error if a
  train loop `CheckpointAction` uses the same manager or directory.

  Each save step evaluates the save decision policies twice: once before
  reading `state.step` and once inside `CheckpointManager.save()`.
  """

  # Tells `Step` not to wait for the outputs. Steps that don't save never read
  # them.
  wait_for_outputs = False

  def __init__(self, ckpt_manager: ocp.CheckpointManager):
    """Initializes the action.

    Args:
      ckpt_manager: The checkpoint manager. Its save decision policies decide on
        each step whether to save.
    """
    super().__init__(ckpt_manager, interval=1)
    self._host_step: Optional[int] = None
    self._last_outputs: Optional[Output] = None

  def save(
      self,
      step: int,
      state: State,
      outputs: Optional[Output],
      force: bool = False,
  ) -> bool:
    """Saves `state` as `step`. `finalize()` calls it with `force=True`.

    Override this to change what is saved; both `__call__` and `finalize()`
    use it, so overrides must accept `force`.

    Args:
      step: The step to save.
      state: The model state.
      outputs: One step's outputs. They are passed to the manager as metrics
        unchanged. Orbax uses metrics only if the manager has a `best_fn`. With
        a `best_fn` and array outputs, override `save()` to convert them to
        Python values, for example with `jax.device_get` and `tolist`.
      force: Save even if the save decision policies decline.

    Returns:
      Whether a save was started.
    """
    return self._ckpt_manager.save(
        step,
        metrics=outputs,
        args=ocp.args.PyTreeSave(state),
        force=force,
    )

  def __call__(self, state: State, outputs: Optional[Output], **kwargs) -> None:
    # Keeps a reference only; `finalize()` stores the last step's outputs.
    self._last_outputs = outputs
    if self._host_step is None:
      # One device read, on the first call only.
      self._host_step = int(state.step)
    else:
      self._host_step += 1
    # Steps that don't save stop here, without reading the device.
    if not self._ckpt_manager.should_save(self._host_step):
      return
    # Read the device step before saving, so a checkpoint is never written
    # under the wrong step. This catches callers that skip steps.
    device_step = int(state.step)
    if device_step != self._host_step:
      logging.error(
          "ContinuousCheckpointAction host step %d does not match state.step"
          " %d. Make sure the action runs once per training step. Using"
          " state.step.",
          self._host_step,
          device_step,
      )
      self._host_step = device_step
    step = self._host_step
    if not self.save(step, state, outputs):
      return
    logging.info(
        "Saved checkpoint to %s on step %d.",
        self._ckpt_manager.directory,
        step,
    )

  def finalize(self, state: State, outputs: Optional[Output] = None) -> bool:
    """Saves the final step if it is not saved yet, then waits for saves.

    Call this on all hosts when training ends. It does not close the manager.
    If the action never ran (no training step ran), it saves nothing.

    Args:
      state: The final model state.
      outputs: One step's outputs to store as metrics. Defaults to the outputs
        of the last step the action saw, so all saves store the same kind of
        metrics.

    Returns:
      Whether this call started a save.
    """
    # Let in-flight saves finish, or roll back, before reading the latest step.
    self._ckpt_manager.wait_until_finished()
    if self._host_step is None:
      return False
    if outputs is None:
      outputs = self._last_outputs
    step = int(state.step)
    latest = self._ckpt_manager.latest_step()
    saved = False
    if latest is None or latest < step:
      saved = self.save(step, state, outputs, force=True)
      if saved:
        logging.info(
            "Saved final checkpoint to %s on step %d.",
            self._ckpt_manager.directory,
            step,
        )
      self._ckpt_manager.wait_until_finished()
    self._host_step = step
    return saved


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


class EarlyStoppingAction(Action):
  """Action to trigger early stopping.

  Delegates state management and stopping logic to the
  MetricMonitoringStopHandler.
  Typically added to 'eval_end_actions' in a Jaxloop Trainer.
  """

  def __init__(self, handler: MetricMonitoringStopHandler):
    """Initializes the early stopping action.

    Args:
      handler: The early stopping handler to use.
    """
    super().__init__()
    self._handler = handler
    self.monitored_metric = handler.monitored_metric

  def __call__(
      self, state: State, outputs: Optional[Output], **kwargs
  ) -> Optional[Output]:
    """Typically called at the end of an evaluation loop.

    Args:
      state: The model state.
      outputs: The model output.
      **kwargs: Additional keyword arguments for the action.

    Returns:
      The unmodified outputs dictionary.

    Raises:
      StopTrainingError: If early stopping is triggered.
    """
    current_step = int(state.step)

    if outputs is None or self.monitored_metric not in outputs:
      logging.warning(
          "Early stopping action '%s' is monitoring metric '%s' which is not"
          " available in the outputs dictionary. Available metrics are: %s",
          self.monitored_metric,
          current_step,
          outputs.keys() if outputs is not None else None,
      )
      return outputs

    current_metric = float(jnp.mean(jnp.array(outputs[self.monitored_metric])))

    if math.isnan(current_metric):
      logging.error(
          "[EarlyStoppingAction] Metric '%s' is NaN at step %d. Stopping"
          " training.",
          self.monitored_metric,
          current_step,
      )
      best_info = self._handler.get_best_info()
      raise StopTrainingError(
          f"Metric '{self.monitored_metric}' was NaN",
          best_step=best_info[1],
          best_metric=best_info[0],
      )

    self._handler.update_state(current_metric, current_step)

    if self._handler.should_stop():
      best_metric, best_step = self._handler.get_best_info()

      logging.debug(
          "[EarlyStoppingAction] Stopping training at step %d."
          " Metric '%s' did not improve for %d cycles. Best metric %.4f at"
          " step %d.",
          current_step,
          self.monitored_metric,
          self._handler.patience,
          best_metric,
          best_step,
      )

      raise StopTrainingError(
          f"Early stopping triggered for metric '{self.monitored_metric}'",
          best_step=best_step,
          best_metric=best_metric,
      )

    return outputs

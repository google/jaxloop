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

"""The outer loop library for JAX models."""

import dataclasses
from typing import Any, Iterable, Iterator, List, Optional, Protocol, Tuple

from absl import logging
from etils import epath
import jax
from jaxloop import actions
from jaxloop import eval_loop as eval_loop_lib
from jaxloop import loop
from jaxloop import step as step_lib
from jaxloop import train_loop as train_loop_lib
from jaxloop import types
from orbax import checkpoint


# pylint: disable=logging-fstring-interpolation

Loop = loop.Loop
Output = types.Output
State = types.TrainState
Step = step_lib.Step


STAT_PREFIX = 'jaxloop'
STOP_LOOP = f'{STAT_PREFIX}/stop_loop'

_ACTION_TIMEOUT_SECS = 1800  # 30 minutes.
_STOP_FILE_NAME = 'STOPPED'
_DEFAULT_CHECKPOINT_TIMEOUT = 0


class EvalContext:
  """The context in which the eval loop is running.

  Attributes:
    is_continuous_eval: Whether the eval loop is running continuously.
    step_num: If in continuous eval mode, this is the step number of the
      checkpoint being evaluated. If in non-continuous eval mode, this is the
      step number of the training loop.
    loop_count: The number of training loops that have been run. In continuous
      eval mode, this number will always be 0.
  """

  is_continuous_eval: bool
  step_num: int
  loop_count: int

  def __init__(
      self,
      is_continuous_eval: bool,
      step_num: int,
      loop_count: int,
      **_: dict[str, Any],
  ):
    self.is_continuous_eval = is_continuous_eval
    self.step_num = step_num
    self.loop_count = loop_count


class ShouldEvalFn(Protocol):
  """The function to determine whether to run the eval loop."""

  def __call__(self, context: EvalContext) -> bool:
    ...


class EvalSpec:
  """The eval loop configuration.

  For every eval loop in the outer loop, create a eval spec to control how to
  run the eval loop. There is a one to one match for eval loop and eval spec.

  Attributes:
    dataset: The dataset used in the eval loop.
    num_steps: The number of steps to run in the eval loop. If None, the eval
      loop will run to the end of the dataset.
    eval_loop_interval: The loop interval to trigger the eval loop. If None, the
      eval loop will be triggered after every inner training loop. The interval
      is always applied.
    should_eval_fn: An optional function to determine whether to run the eval
      loop. The function argument is an :class:`EvalContext` dataclass object.
      When this function is provided, it is used in parallel to
      eval_loop_interval above i.e. both this function and eval_loop_interval
      must be satisfied for the eval loop to run.
    kwargs: Additional keyword arguments to pass to the eval loop if necessary.
  """

  dataset: Iterable[Any]
  num_steps: Optional[int] = None
  eval_loop_interval: Optional[int] = None
  should_eval_fn: Optional[ShouldEvalFn] = None

  def __init__(
      self,
      dataset: Iterable[Any],
      num_steps: Optional[int] = None,
      eval_loop_interval: Optional[int] = None,
      should_eval_fn: Optional[ShouldEvalFn] = None,
      **_: dict[str, Any],
  ):
    self.dataset = dataset
    self.num_steps = num_steps
    self.eval_loop_interval = eval_loop_interval
    self.should_eval_fn = should_eval_fn


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
  """Defines how checkpoints are handled in training and eval.

  These options apply to the train loop and all eval loops.
  """

  # The path to write checkpoints.
  checkpoint_dir: epath.Path
  # The number of seconds to wait between checking for new checkpoints. Only
  # used during standalone eval.
  iterate_interval_secs: int = _DEFAULT_CHECKPOINT_TIMEOUT
  # A function that returns True when training is finished and no new
  # checkpoints will be written. By default this looks for a `STOPPED` file in
  # checkpoint_dir. Only used during standalone eval.
  iterate_stop_fn: Optional[types.IterateStopFn] = None
  # An optional transform representation to be applied when restoring
  # checkpoints.
  transforms: Optional[Any] = None


class OuterLoop:
  """The outer loop class."""

  def __init__(
      self,
      train_loop: Optional[train_loop_lib.TrainLoop] = None,
      eval_loops: Optional[List[eval_loop_lib.EvalLoop]] = None,
      checkpoint_spec: Optional[CheckpointSpec] = None,
  ):
    """Initializes the outer loop.

    Args:
      train_loop: The training loop to be executed if provided.
      eval_loops: One or more evaluation loops to be executed if provided.
      checkpoint_spec: Configuration for checkpoints, used for loading existing
        checkpoints during training and standalone eval experiments.
    """
    if train_loop is None and eval_loops is None:
      raise ValueError('Either `train_loop` or `eval_loops` must be provided.')
    if train_loop is None and checkpoint_spec is None:
      raise ValueError(
          'Either `train_loop` or `checkpoint_spec` must be provided.'
      )
    self._train_loop = train_loop
    self._eval_loops = eval_loops
    self._checkpoint_spec = checkpoint_spec

  def _run_precheck(
      self,
      train_dataset: Optional[Iterator[Any]] = None,
      train_loop_steps: Optional[int] = None,
      eval_specs: Optional[list[EvalSpec]] = None,
  ) -> None:
    """Prechecks the parameters for running the outer loop.

    Args:
      train_dataset: The training dataset iterator.
      train_loop_steps: The number of steps to run in each train loop.
      eval_specs: The evaluation datasets, number of steps and intervals.
    """
    if self._train_loop is not None:
      if train_dataset is None:
        raise ValueError('`train_dataset` must be provided.')

      if train_loop_steps is not None:
        if train_loop_steps <= 0:
          raise ValueError('`train_loop_steps` must be positive.')

    if self._eval_loops is not None:
      if eval_specs is None:
        raise ValueError('`eval_specs` must be provided.')
      if len(self._eval_loops) != len(eval_specs):
        raise ValueError(
            'Number of eval loops must match number of eval specs.'
        )

  def _restore_state(
      self,
      step: Step,
      state: State,
      checkpoint_dir: Optional[epath.Path] = None,
      step_num: Optional[int] = None,
      dataset_iter: Optional[Iterator[Any]] = None,
  ) -> State:
    """Restores the model state from the checkpoint.

    Args:
      step: The step object to restore.
      state: The model state.
      checkpoint_dir: The checkpoint directory. It not provided, the checkpoint
        directory from the checkpoint spec will be used.
      step_num: The step number to restore. If not provided, the step object
        will automatically determine the latest step.
      dataset_iter: The dataset iterator.

    Returns:
      The model state containing the restored checkpoint.
    """
    del dataset_iter  # Unused.
    if self._checkpoint_spec is None:
      raise ValueError('`checkpoint_spec` must be provided.')

    if checkpoint_dir is None:
      checkpoint_dir = self._checkpoint_spec.checkpoint_dir
      if checkpoint_dir is None:
        raise ValueError('`checkpoint_dir` must be provided.')

    return step.restore_model(
        state,
        checkpoint_dir,
        step=step_num,
        transforms=self._checkpoint_spec.transforms,
    )

  def _get_eval_checkpoint_iterator(self) -> Iterator[int]:
    """Returns an iterator for the eval checkpoint steps.

    Returns:
      An iterator of checkpoint steps.
    """
    if self._checkpoint_spec is None:
      raise ValueError('`checkpoint_spec` must be provided.')

    checkpoint_dir = self._checkpoint_spec.checkpoint_dir

    def default_timeout_fn():
      stopped_file = checkpoint_dir / _STOP_FILE_NAME
      return stopped_file.exists()

    if self._checkpoint_spec.iterate_stop_fn is not None:
      timeout_fn = self._checkpoint_spec.iterate_stop_fn
    else:
      timeout_fn = default_timeout_fn

    return checkpoint.checkpoint_utils.checkpoints_iterator(
        checkpoint_dir,
        timeout=self._checkpoint_spec.iterate_interval_secs,
        timeout_fn=timeout_fn,
    )

  def _run_eval_loops(
      self,
      state: State,
      eval_specs: Optional[list[EvalSpec]] = None,
  ) -> Tuple[State, Optional[Output]]:
    """Runs the eval loop continuously.

    Args:
      state: The model state.
      eval_specs: The evaluation datasets, number of steps and intervals.

    Returns:
      A tuple of the model state and output. The last eval_state and output
      from the last eval loop are returned.
    """
    if self._eval_loops is None:
      raise ValueError('`eval_loops` must be provided.')

    eval_state = state
    outputs = None
    for step_num in self._get_eval_checkpoint_iterator():
      for eval_loop, spec in zip(self._eval_loops, eval_specs):
        if spec.should_eval_fn is not None and not spec.should_eval_fn(
            EvalContext(
                is_continuous_eval=True,
                step_num=step_num,
                loop_count=0,
            )
        ):
          continue

        eval_state = self._restore_state(
            eval_loop.step, state, step_num=step_num
        )
        eval_state, outputs = eval_loop(
            eval_state, iter(spec.dataset), spec.num_steps
        )
    return eval_state, outputs

  def _get_stop_loop(self, outputs: Optional[Output]) -> bool:
    """Gets the stop_loop value in the outputs.

    Args:
      outputs: The model output.

    Returns:
      Value of `outputs[STOP_LOOP]` or False if outputs is None.
    """
    return outputs.get(STOP_LOOP, False) if outputs is not None else False

  def __call__(
      self,
      state: State,
      train_dataset: Optional[Iterator[Any]] = None,
      train_total_steps: Optional[int] = None,
      train_loop_steps: Optional[int] = None,
      eval_specs: Optional[list[EvalSpec]] = None,
      log_train_total_steps: bool = True,
  ) -> Tuple[State, Optional[Output]]:
    """Invokes the outer loop.

    Args:
      state: The model state.
      train_dataset: The training dataset. This could be an iterable dataset or
        an iterator of the dataset. If the former is given, Jaxloop will
        internally convert the iterable dataset to an iterator.
      train_total_steps: The total number of training steps to run in the outer
        loop.
      train_loop_steps: The number of steps to run in each train loop.
      eval_specs: The evaluation specs including datasets, number of steps and
        intervals. There is a one to one mapping between eval_loops and
        eval_specs.
      log_train_total_steps: Whether to log the total number of training steps.

    Returns:
      A tuple of the model state and output.
    """
    train_dataset = iter(train_dataset) if train_dataset is not None else None
    self._run_precheck(train_dataset, train_loop_steps, eval_specs)

    if self._train_loop is None:
      return self._run_eval_loops(
          state=state,
          eval_specs=eval_specs,
      )

    if self._checkpoint_spec is not None:
      state = self._restore_state(
          self._train_loop.step, state, dataset_iter=train_dataset
      )

    step = int(state.step)
    if log_train_total_steps:
      logging.info(
          f'train    | step: {step: 6d} | training until step '
          f'{train_total_steps: 6d}.'
      )

    if step and train_loop_steps:
      self._train_loop.loop_count = step // train_loop_steps
      num_steps = train_loop_steps - (step % train_loop_steps)
    else:
      num_steps = train_loop_steps

    outputs = None
    stop_loop = False
    while step < train_total_steps and not stop_loop:
      num_steps = min(num_steps, train_total_steps - step)
      state, new_outputs = self._train_loop(state, train_dataset, num_steps)
      num_steps = train_loop_steps
      if step == int(state.step):
        logging.warning(
            'Breaking out of the training loop because step did not advance. '
            'This is likely because the training dataset is exhausted before '
            'reaching the desired number of steps.'
        )
        break
      step = int(state.step)
      outputs = new_outputs
      stop_loop = self._get_stop_loop(outputs)

      if self._eval_loops is not None and eval_specs is not None:
        for eval_loop, spec in zip(self._eval_loops, eval_specs):
          if (
              spec.eval_loop_interval is not None
              and self._train_loop.loop_count % spec.eval_loop_interval != 0
          ):
            continue
          elif spec.should_eval_fn is not None and not spec.should_eval_fn(
              EvalContext(
                  is_continuous_eval=False,
                  step_num=self._train_loop.loop_count * train_loop_steps,
                  loop_count=self._train_loop.loop_count,
              )
          ):
            continue

          state, eval_outputs = eval_loop(
              state, iter(spec.dataset), spec.num_steps
          )
          stop_loop = stop_loop or self._get_stop_loop(eval_outputs)

    # Make sure that the CheckpointManager is properly closed, and all the write
    # in threads are committed.
    if self._train_loop and self._train_loop.end_actions:
      for action in self._train_loop.end_actions:
        if isinstance(action, actions.CheckpointAction):
          action._ckpt_manager.close()  # pylint: disable=protected-access

    if self._checkpoint_spec is not None and jax.process_index() == 0:
      stopped_file = self._checkpoint_spec.checkpoint_dir / _STOP_FILE_NAME
      logging.info(f'Writing stopped file: {stopped_file}')
      if not self._checkpoint_spec.checkpoint_dir.exists():
        self._checkpoint_spec.checkpoint_dir.mkdir(parents=True)
      stopped_file.write_text('')

    if outputs is not None:
      outputs[STOP_LOOP] = stop_loop
    return state, outputs

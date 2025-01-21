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

"""The eval loop library for JAX models."""

import enum
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from absl import logging
from jaxloop import loop
from jaxloop import pipeline_loop
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import types

# pylint: disable=logging-fstring-interpolation

Loop = loop.Loop
State = types.TrainState
Step = step_lib.Step


@enum.unique
class EvalMode(enum.Enum):
  """The eval mode for running model evaluation logic.

  The current available modes are:
    LOOP: A for-loop based implementation running model evaluation.
    PIPELINE: A ML Analysis Toolkit based implementation running model
      evaluation.
  """

  LOOP = 'LOOP'
  PIPELINE = 'PIPELINE'


class EvalLoop(pipeline_loop.PipelineLoop):
  """The loop for evaluation.

  This class can be used to define the loop for evaluation. It provides some
  basic functionalities like running actions and calculating evaluation time.
  """

  def __init__(
      self,
      step: Step,
      stat_names: Optional[Sequence[str]] = None,
      **kwargs,
  ):
    if stat_names is None:
      stat_names = [stat_loop.STAT_LOOP_TIME_SECS]
    super().__init__(step, stat_names=stat_names, **kwargs)

  def run(
      self,
      state: State,
      dataset: Iterator[Any],
      num_steps: Optional[int] = None,
      mode: EvalMode = EvalMode.LOOP,
      **kwargs,
  ) -> Tuple[State, Optional[Dict[str, Any]]]:
    """Runs the loop for evaluation.

    Args:
      state: The model state.
      dataset: The dataset iterator.
      num_steps: The number of steps to run.
      mode: The eval mode for running model evaluation logic. For more details,
        please check out EvalMode.
      **kwargs: Addtional keyword arguments.

    Returns:
      A tuple of the model state and output.
    """
    step = int(state.step)
    if num_steps is None:
      logging.info(
          f'eval     | step: {step: 6d} | running until the end of eval data.'
      )
    else:
      logging.info(
          f'eval     | step: {step: 6d} | running {num_steps} eval steps.'
      )
    if mode == EvalMode.LOOP:
      return super(pipeline_loop.PipelineLoop, self).run(
          state, dataset, num_steps, **kwargs
      )
    if mode == EvalMode.PIPELINE:
      return super().run(state, dataset, num_steps, **kwargs)
    raise ValueError(f'Unknown eval mode: {mode}')

  def end(
      self, state: State, outputs: Optional[Dict[str, Any]]
  ) -> Tuple[State, Optional[Dict[str, Any]]]:
    """Ends the loop for evaluation.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    state, outputs = super().end(state, outputs)
    if stat_loop.STAT_LOOP_TIME_SECS in outputs:
      step = int(state.step)
      eval_time = outputs[stat_loop.STAT_LOOP_TIME_SECS]
      logging.info(
          f'eval     | step: {step: 6d} | eval time: {eval_time: 4.2f} seconds.'
      )
    return state, outputs

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

"""The loop with actions library for JAX models."""

from typing import Any, Iterator, List, Optional, Tuple

import jax
from jaxloop import actions
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import types

Action = actions.Action
Output = types.Output
State = types.TrainState
Step = step_lib.Step


class ActionLoop(stat_loop.StatLoop):
  """The loop with actions.

  This class extends the base loop class and adds the functionality of being
  able to run actions before and/or after the loop.
  """

  def __init__(
      self,
      step: Step,
      begin_actions: Optional[List[Action]] = None,
      end_actions: Optional[List[Action]] = None,
      **kwargs,
  ):
    super().__init__(step, **kwargs)
    self._begin_actions = begin_actions
    self._end_actions = end_actions
    self._loop_count = 0

  def begin(
      self, state: State, dataset: Iterator[Any]
  ) -> Tuple[State, Iterator[Any]]:
    """Begins the loop by running the actions if needed.

    Args:
      state: The model state.
      dataset: The dataset iterator.

    Returns:
      A tuple of the model state and dataset iterator.
    """
    self._loop_count += 1
    if self._begin_actions is not None:
      for action in self._begin_actions:
        if self._loop_count % action.interval == 0:
          action(state, None)
    return super().begin(state, dataset)

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    """Ends the loop by running the actions if needed.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    state, outputs = super().end(state, outputs)
    jax.block_until_ready((state, outputs))
    if self._end_actions is not None:
      for action in self._end_actions:
        if self._loop_count % action.interval == 0:
          action(state, outputs)
    return state, outputs

  @property
  def begin_actions(self) -> Optional[List[Action]]:
    return self._begin_actions

  @property
  def end_actions(self) -> Optional[List[Action]]:
    return self._end_actions

  @property
  def loop_count(self) -> int:
    return self._loop_count

  @loop_count.setter
  def loop_count(self, loop_count: int):
    self._loop_count = loop_count

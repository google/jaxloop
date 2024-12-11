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

"""The train loop library for JAX models."""

from typing import Optional, Sequence, Tuple

from absl import logging
from jaxloop import action_loop
from jaxloop import loop
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import types


# pylint: disable=logging-fstring-interpolation

Loop = loop.Loop
Output = types.Output
State = types.TrainState
Step = step_lib.Step


class TrainLoop(action_loop.ActionLoop):
  """The loop for training.

  This class can be used to define the loop for training. It provides some basic
  functionalities like running actions and calculating simple training metrics
  like number of steps per second.
  """

  def __init__(
      self,
      step: Step,
      stat_names: Optional[Sequence[str]] = None,
      **kwargs,
  ):
    stat_names = stat_names or [stat_loop.STAT_STEPS_PER_SEC]
    super().__init__(step, stat_names=stat_names, **kwargs)

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    """Ends the inner loop for training.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    state, outputs = super().end(state, outputs)
    if stat_loop.STAT_STEPS_PER_SEC in outputs:
      step = int(state.step)
      steps_per_sec = outputs[stat_loop.STAT_STEPS_PER_SEC]
      logging.info(
          f'train    | step: {step: 6d} | steps_per_sec: {steps_per_sec: 6.1f}'
      )
    return state, outputs

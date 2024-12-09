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

"""Unit tests for actions."""

# (TODO: b/347771130): Add unit tests for other actions.

from absl.testing import absltest
from jaxloop import actions
from jaxloop import types
from jaxloop.step_number_writer import step_number_writer
import optax

State = types.TrainState


class FakeStepNumberWriter(step_number_writer.StepNumberWriter):
  """Track the step numbers in a list."""

  def __init__(self):
    self._step_numbers: list[int] = []

  def write(self, step_number: int) -> None:
    self._step_numbers.append(step_number)

  def step_numbers(self) -> list[int]:
    return self._step_numbers


class ActionsTest(absltest.TestCase):

  def test_action_interval(self):
    action = actions.Action(interval=2)
    self.assertEqual(action.interval, 2)

  def test_step_number_action(self):
    state = State.create(
        apply_fn=None,
        params={'params': 1},
        tx=optax.identity(),
    ).replace(step=1)
    step_writer = FakeStepNumberWriter()
    action = actions.StepNumberAction(step_writer)
    action(state)
    self.assertEqual(step_writer.step_numbers(), [1])


if __name__ == '__main__':
  absltest.main()

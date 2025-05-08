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


import dataclasses
from typing import cast
from unittest import mock

from absl.testing import absltest
import chex
import jax.numpy as jnp
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
        params={"params": 1},
        tx=optax.identity(),
    ).replace(step=1)
    step_writer = FakeStepNumberWriter()
    action = actions.StepNumberAction(step_writer)
    action(state)
    self.assertEqual(step_writer.step_numbers(), [1])


class SummaryActionTest(absltest.TestCase):

  def test_summary_action(self):
    @dataclasses.dataclass
    class MockedState:
      step: int

    histograms = []
    scalars = []

    mocked_writer = mock.MagicMock()
    mocked_writer.write_histograms = lambda step, data: histograms.append(
        (step, data)
    )
    mocked_writer.write_scalars = lambda step, data: scalars.append(
        (step, data)
    )
    action = actions.SummaryAction(
        summary_writer=mocked_writer,
        interval=1,
        flush_each_call=True,
    )

    outputs = {
        "a": actions.MetricWithMetadata(
            jnp.array([1.0, 2.0]), type=types.MetricType.HISTOGRAM
        ),
        "b": actions.MetricWithMetadata(
            jnp.array([3.0, 4.0]), type=types.MetricType.HISTOGRAM
        ),
        "c": jnp.array([5.0, 6.0]),
        "d": actions.MetricWithMetadata(1.5, type=types.MetricType.SCALAR),
    }
    action(cast(types.TrainState, MockedState(1)), outputs)

    chex.assert_trees_all_equal(
        histograms,
        [
            (1, {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}),
        ],
    )
    chex.assert_trees_all_equal(
        scalars, [(1, {"c": jnp.array([5.0, 6.0]), "d": 1.5})]
    )

  def test_unsupported_type(self):
    @dataclasses.dataclass
    class MockedState:
      step: int

    histograms = []
    scalars = []

    mocked_writer = mock.MagicMock()
    mocked_writer.write_histograms = lambda step, data: histograms.append(
        (step, data)
    )
    mocked_writer.write_scalars = lambda step, data: scalars.append(
        (step, data)
    )
    action = actions.SummaryAction(
        summary_writer=mocked_writer,
        interval=1,
        flush_each_call=True,
    )

    outputs = {
        # Adding unsupported type here. This should raise an error.
        "a": actions.MetricWithMetadata(
            jnp.array([1.0, 2.0]), type=types.MetricType.AUDIO
        ),
    }
    with self.assertRaises(ValueError):
      action(cast(types.TrainState, MockedState(1)), outputs)



if __name__ == "__main__":
  absltest.main()

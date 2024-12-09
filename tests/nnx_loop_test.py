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

"""Unit tests for the loop library."""

from typing import Any, Iterator, Optional, Tuple

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jaxloop import loop
from jaxloop import step as step_lib
from jaxloop import types
import optax

Batch = types.Batch
Output = types.Output
State = types.TrainState
Step = step_lib.Step


class TestModel(nnx.Module):

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x):
    x = self.linear(x)
    x = nnx.log_softmax(x)
    return x


class TestStep(Step):

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:
    return state.replace(step=state.step + 1), {'step': state.step}


class TestLoop(loop.Loop):

  def __init__(self, step: Step):
    self.begin_step = None
    self.end_step = None
    super().__init__(step)

  def begin(
      self, state: State, dataset: Iterator[Any]
  ) -> Tuple[State, Iterator[Any]]:
    self.begin_step = state.step
    return super().begin(state, dataset)

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    self.end_step = state.step
    return super().end(state, outputs)


class LoopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.step = TestStep(
        jax.random.PRNGKey(0),
        TestModel,
        optimizer=optax.adam(1e-4),
        nnx_model_args=(1, 10),
    )
    self.loop = TestLoop(self.step)
    self.shape = [1, 28, 28, 1]
    self.dataset = iter([jnp.ones(self.shape)] * 10)

  def test_loop_dataset(self):
    state = self.step.initialize_model(self.shape)
    state, outputs = self.loop(state, self.dataset)
    self.assertEqual(self.loop.begin_step, 0)
    self.assertEqual(self.loop.end_step, 10)
    self.assertEqual(state.step, 10)
    self.assertEqual(outputs['step'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  def test_loop_num_steps(self):
    state = self.step.initialize_model(self.shape)
    state, outputs = self.loop(state, self.dataset, num_steps=5)
    self.assertEqual(self.loop.begin_step, 0)
    self.assertEqual(self.loop.end_step, 5)
    self.assertEqual(state.step, 5)
    self.assertEqual(outputs['step'], [0, 1, 2, 3, 4])

  def test_loop_continue_num_steps(self):
    state = self.step.initialize_model(self.shape)
    state = state.replace(step=5)
    state, outputs = self.loop(state, self.dataset, num_steps=5)
    self.assertEqual(self.loop.begin_step, 5)
    self.assertEqual(self.loop.end_step, 10)
    self.assertEqual(state.step, 10)
    self.assertEqual(outputs['step'], [5, 6, 7, 8, 9])

  def test_model_args_validation(self):
    self.assertRaises(
        ValueError,
        lambda: TestStep(
            jax.random.PRNGKey(0),
            TestModel,
            optimizer=optax.adam(1e-4),
        ),
    )


if __name__ == '__main__':
  absltest.main()

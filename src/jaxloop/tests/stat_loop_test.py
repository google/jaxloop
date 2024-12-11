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

"""Unit tests for the statistic loop library."""

from typing import Optional, Tuple

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import types
import optax

Batch = types.Batch
Output = types.Output
State = types.TrainState
Step = step_lib.Step


class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


class TestStep(Step):

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:
    return state.replace(step=state.step + 1), None


class StatLoopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.step = TestStep(
        jax.random.PRNGKey(0), self.model, optax.adam(1e-4)
    )
    self.stat_names = [
        stat_loop.STAT_NUM_PARAMS,
        stat_loop.STAT_NUM_FLOPS,
        stat_loop.STAT_LIFE_TIME_SECS,
        stat_loop.STAT_STEPS_PER_SEC,
    ]
    self.loop = stat_loop.StatLoop(self.step, stat_names=self.stat_names)
    self.shape = [1, 28, 28, 1]
    self.dataset = iter([jnp.ones(self.shape)] * 10)

  def test_stat_loop(self):
    state = self.step.initialize_model(self.shape)
    _, outputs = self.loop(state, self.dataset)
    self.assertTrue(all(name in outputs for name in self.stat_names))


if __name__ == '__main__':
  absltest.main()

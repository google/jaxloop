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

"""Unit tests for pipeline loop."""

import collections
from typing import Any, Optional, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import pipeline_loop
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
    return state, {'step': 1}


class PipelineLoopTest(parameterized.TestCase):

  def test_pipeline_with_checkpoint(self):
    shape = [1, 28, 28, 1]
    model = TestModel()
    dataset = iter([jnp.ones(shape)] * 10)
    test_step = TestStep(jax.random.PRNGKey(0), model)
    state = test_step.initialize_model(shape)
    loop = pipeline_loop.PipelineLoop(
        test_step,
    )
    _, result = loop(state, dataset)
    self.assertEqual(result['step'], [1] * 10)


if __name__ == '__main__':
  absltest.main()

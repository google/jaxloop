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

"""Unit tests for eval loop."""

from typing import Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import eval_loop
from jaxloop import loop
from jaxloop import pipeline_loop
from jaxloop import step as step_lib
from jaxloop import types

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


class EvalLoopTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='loop', mode=eval_loop.EvalMode.LOOP),
      dict(testcase_name='pipeline', mode=eval_loop.EvalMode.PIPELINE),
  )
  def test_eval_loop(self, mode: eval_loop.EvalMode):
    shape = [1, 28, 28, 1]
    model = TestModel()
    dataset = iter([jnp.ones(shape)] * 10)
    test_step = TestStep(jax.random.PRNGKey(0), model)
    state = test_step.initialize_model(shape)
    loop = eval_loop.EvalLoop(test_step)
    _, result = loop(state, dataset, mode=mode)
    self.assertEqual(result['step'], [1] * 10)

  @parameterized.named_parameters(
      dict(testcase_name='loop', mode=eval_loop.EvalMode.LOOP),
      dict(testcase_name='pipeline', mode=eval_loop.EvalMode.PIPELINE),
  )
  @mock.patch.object(pipeline_loop.PipelineLoop, 'run', autospec=True)
  @mock.patch.object(loop.Loop, 'run', autospec=True)
  def test_eval_loop_with_mock(
      self,
      loop_run: mock.Mock,
      pipeline_loop_run: mock.Mock,
      mode: eval_loop.EvalMode,
  ):
    shape = [1, 28, 28, 1]
    model = TestModel()
    dataset = iter([jnp.ones(shape)] * 10)
    test_step = TestStep(jax.random.PRNGKey(0), model)
    state = test_step.initialize_model(shape)
    pipeline_loop_run.return_value = (state, None)
    loop_run.return_value = (state, None)
    eval_loop.EvalLoop(test_step)(state, dataset, mode=mode)
    if mode == eval_loop.EvalMode.LOOP:
      loop_run.assert_called_once()
      pipeline_loop_run.assert_not_called()
    if mode == eval_loop.EvalMode.PIPELINE:
      loop_run.assert_not_called()
      pipeline_loop_run.assert_called_once()


if __name__ == '__main__':
  absltest.main()

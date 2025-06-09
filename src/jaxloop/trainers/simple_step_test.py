# Copyright 2025 Google LLC
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

"""Unit tests for the step library."""

import os

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import types
from jaxloop.trainers import simple_step
import optax
from orbax import checkpoint

Batch = types.Batch
Output = types.Output
State = types.TrainState


class TestModel(nn.Module):
  """A fully-connected neural network model with 3 layers."""

  with_dropout: bool = False

  @nn.compact
  def __call__(self, x, train: bool = True, **kwargs):
    for _ in range(3):
      x = nn.Dense(features=16)(x)
      x = nn.relu(x)
      if self.with_dropout:
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
    x = nn.Dense(features=1)(x)
    return x


class StepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.step = simple_step.SimpleStep(
        jax.random.PRNGKey(0),
        self.model,
        optimizer=optax.adam(1e-4),
        train=True,
    )
    self.checkpoint_dir = os.path.join(
        self.create_tempdir().full_path, 'checkpoint'
    )
    self.checkpoint_manager = checkpoint.CheckpointManager(
        self.checkpoint_dir,
        checkpoint.PyTreeCheckpointer(),
    )
    self.spec = {'input_features': [2, 3], 'output_features': [2, 1]}
    self.batch = {
        'input_features': jnp.ones([2, 3]),
        'output_features': jnp.ones([2, 1]),
    }

  def test_initialize_model(self):
    state = self.step.initialize_model(self.spec)
    self.assertEqual(state.step, 0)
    self.assertIn('Dense_0', state.params)
    self.assertIn('Dense_1', state.params)
    self.assertIn('Dense_2', state.params)

  def test_restore_model(self):
    state = self.step.initialize_model(self.spec)
    self.checkpoint_manager.save(0, args=checkpoint.args.PyTreeSave(state))
    self.checkpoint_manager.wait_until_finished()
    state = self.step.restore_model(state, self.checkpoint_dir)
    self.assertEqual(state.step, 0)
    self.assertIn('Dense_0', state.params)
    self.assertIn('Dense_1', state.params)
    self.assertIn('Dense_2', state.params)

  def test_restore_model_no_latest_step(self):
    state = self.step.initialize_model(self.spec)
    with self.assertLogs(level='INFO') as cm:
      restored_state = self.step.restore_model(state, self.checkpoint_dir)
      self.assertEqual(state, restored_state)
    self.assertRegex(cm.output[0], r'No checkpoint found in.*')

  def test_compile(self):
    state = self.step.initialize_model(self.spec)
    self.assertEqual(state.step, 0)

    self.step.compile()
    self.assertEqual(state.step, 0)

    state, _ = self.step(state, self.batch)
    self.assertEqual(state.step, 1)

    # `keep_unused` is a valid `jax.jit` compile flag.
    self.step.compile(keep_unused=True)
    state, _ = self.step(state, self.batch)
    self.assertEqual(state.step, 2)

    # An invalid `jax.jit` compile flag will result in an error.
    with self.assertRaisesRegex(
        TypeError,
        '.*got an unexpected keyword argument.*',
    ):
      self.step.compile(some_flag_that_does_not_exist=True)

  def test_step_with_prng_list(self):
    self.step = simple_step.SimpleStep(
        {'params': jax.random.PRNGKey(0), 'others': jax.random.PRNGKey(1)},
        self.model,
        optimizer=optax.adam(1e-4),
    )
    state = self.step.initialize_model(self.spec)
    state, _ = self.step(state, self.batch)
    self.assertEqual(state.step, 1)

  def test_step(self):
    state = self.step.initialize_model(self.spec)
    state, output = self.step(state, self.batch)
    self.assertEqual(state.step, 1)
    self.assertIn('loss', output)
    self.assertIn('output_features_pred', output)

    loss_1 = output['loss']

    for i in range(3):
      state, output = self.step(state, self.batch)
      self.assertEqual(state.step, 2 + i)
      self.assertIn('loss', output)
      self.assertIn('output_features_pred', output)

    self.assertLess(output['loss'].total.item(), loss_1.total.item())

  def test_step_with_dropout(self):
    model = TestModel(with_dropout=True)
    step = simple_step.SimpleStep(
        {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)},
        model,
        optimizer=optax.adam(1e-4),
        train=True,
    )
    state = step.initialize_model(self.spec)
    state, output = step(state, self.batch)
    self.assertEqual(state.step, 1)
    self.assertIn('loss', output)
    self.assertIn('output_features_pred', output)

    loss_1 = output['loss']

    for i in range(3):
      state, output = step(state, self.batch)
      self.assertEqual(state.step, 2 + i)
      self.assertIn('loss', output)
      self.assertIn('output_features_pred', output)

    self.assertLess(output['loss'].total.item(), loss_1.total.item())


if __name__ == '__main__':
  absltest.main()

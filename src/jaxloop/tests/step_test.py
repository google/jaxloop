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

"""Unit tests for the step library."""

import collections
import os
from typing import Optional, Tuple, cast

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import partition
from jaxloop import step
from jaxloop import types
import optax
from orbax import checkpoint

Batch = types.Batch
Output = types.Output
State = types.TrainState

Point = collections.namedtuple('Point', ['x', 'y'])


class TestModel(nn.Module):

  @nn.compact
  def __call__(self, inputs):
    x = inputs['x']
    z = inputs['y']['z']
    a = inputs['a']
    x = nn.Dense(features=10)(x)
    z = nn.Dense(features=10)(z)
    a = nn.Dense(features=10)(a)
    x = x + z + a
    x = nn.log_softmax(x)
    return x


class TestStep(step.Step):

  def __init__(
      self,
      base_prng: types.PRNGType,
      model: nn.Module,
      optimizer: Optional[optax.GradientTransformation] = None,
      partitioner: partition.Partitioner = partition.SingleDevicePartitioner(),
  ):
    super().__init__(base_prng, model, optimizer, partitioner)  # pyrefly: ignore[bad-argument-type]
    self.begin_step = None
    self.end_step = None

  def begin(self, state: State, batch: Batch) -> tuple[State, Batch]:
    self.begin_step = state.step
    return state, batch

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:  # pyrefly: ignore[bad-override]
    return state.replace(step=state.step + 1), None

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    self.end_step = state.step
    return super().end(state, outputs)


class StepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.step = TestStep(
        jax.random.PRNGKey(0), self.model, optimizer=optax.adam(1e-4)
    )
    self.checkpoint_dir = os.path.join(
        self.create_tempdir().full_path, 'checkpoint'
    )
    self.checkpoint_manager = checkpoint.CheckpointManager(
        self.checkpoint_dir,
        checkpoint.PyTreeCheckpointer(),
    )
    x = jnp.ones([2, 3])
    self.batch = {
        'x': x,
        'y': {'z': jnp.ones([2, 4], dtype=jnp.float16)},
        'a': jnp.ones(6, dtype=jnp.bfloat16),
        # Unused in the test model but checks that we can handle namedtuples in
        # batches.
        'p': Point(x=x, y=x),
    }
    self.spec = jax.tree.map(lambda x: (x.shape, x.dtype), self.batch)

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

    self.step.compile()
    state, _ = self.step(state, self.batch)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(state.step, 1)
    self.assertEqual(self.step.begin_step, 0)
    self.assertEqual(self.step.end_step, 1)

    # `keep_unused` is a valid `jax.jit` compile flag.
    self.step.compile(keep_unused=True)
    state, _ = self.step(state, self.batch)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(state.step, 2)
    self.assertEqual(self.step.begin_step, 1)
    self.assertEqual(self.step.end_step, 2)

    # An invalid `jax.jit` compile flag will result in an error.
    with self.assertRaisesRegex(
        TypeError,
        '.*got an unexpected keyword argument.*',
    ):
      self.step.compile(some_flag_that_does_not_exist=True)

  def test_step(self):
    state = self.step.initialize_model(self.spec)
    state, _ = self.step(state, self.batch)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(state.step, 1)
    self.assertEqual(self.step.begin_step, 0)
    self.assertEqual(self.step.end_step, 1)

  def test_step_log_num_flops(self):
    state = self.step.initialize_model(self.spec)
    self.step(state, self.batch, log_num_flops=True)  # pyrefly: ignore[bad-argument-type]
    self.assertGreater(self.step.num_flops, 0)  # pyrefly: ignore[no-matching-overload]

  def test_step_with_prng_list(self):
    self.step = TestStep(
        {'params': jax.random.PRNGKey(0), 'others': jax.random.PRNGKey(1)},
        self.model,
        optimizer=optax.adam(1e-4),
    )
    state = self.step.initialize_model(self.spec)
    state, _ = self.step(state, self.batch)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(state.step, 1)
    self.assertEqual(self.step.begin_step, 0)
    self.assertEqual(self.step.end_step, 1)

  def test_state_class_defaults_to_train_state(self):
    self.assertIs(self.step._STATE_CLASS, State)
    self.assertIsInstance(self.step.initialize_model(self.spec), State)


class CustomState(State):
  """A `TrainState` subclass carrying an extra model variable collection."""

  counters: Optional[dict[str, jnp.ndarray]] = None

  @classmethod
  def create(cls, *, apply_fn, tx, **variables):
    known = {
        k: v
        for k, v in variables.items()
        if k in ('params', 'opt_state', 'batch_stats')
    }
    extra = {k: v for k, v in variables.items() if k not in known}
    return super().create(
        apply_fn=apply_fn, tx=tx, counters=extra or None, **known
    )


class CustomStateModel(nn.Module):

  @nn.compact
  def __call__(self, inputs):
    # Registers a non-param variable collection so that `_STATE_KEYS` has
    # something beyond `params` to forward.
    counter = self.variable('counters', 'calls', jnp.zeros, ())
    del counter
    return nn.Dense(features=2)(inputs['x'])


class CustomStateStep(step.Step):
  """A step that opts into a custom train state via `_STATE_CLASS`."""

  _STATE_CLASS = CustomState
  _STATE_KEYS = ('step', 'params', 'batch_stats', 'counters')

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:  # pyrefly: ignore[bad-override]
    return state, None


class StateClassTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.step = CustomStateStep(
        jax.random.PRNGKey(0), CustomStateModel(), optimizer=optax.adam(1e-4)
    )
    self.spec = jax.tree.map(
        lambda x: (x.shape, x.dtype), {'x': jnp.ones([2, 3])}
    )

  def test_initialize_model_uses_the_overridden_state_class(self):
    state = self.step.initialize_model(self.spec)

    self.assertIsInstance(state, CustomState)

  def test_overridden_state_class_receives_extra_variables(self):
    # The point of the hook: collections outside the default `_STATE_KEYS` can
    # be threaded into a custom state without rebinding `step.State`.
    state = cast(CustomState, self.step.initialize_model(self.spec))

    self.assertIsNotNone(state.counters)
    self.assertIn('counters', state.counters or {})

  def test_overriding_does_not_affect_the_module_level_state(self):
    self.step.initialize_model(self.spec)

    self.assertIs(step.State, State)

  def test_other_steps_are_unaffected(self):
    other = TestStep(
        jax.random.PRNGKey(0), TestModel(), optimizer=optax.adam(1e-4)
    )
    self.step.initialize_model(self.spec)

    batch = {
        'x': jnp.ones([2, 3]),
        'y': {'z': jnp.ones([2, 4], dtype=jnp.float16)},
        'a': jnp.ones(6, dtype=jnp.bfloat16),
    }
    state = other.initialize_model(
        jax.tree.map(lambda x: (x.shape, x.dtype), batch)
    )

    self.assertNotIsInstance(state, CustomState)


if __name__ == '__main__':
  absltest.main()

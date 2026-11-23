# Copyright 2026 Google LLC
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

"""Unit tests for the outer loop with continuous checkpointing."""

import itertools
from typing import Any, Callable, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from etils import epath
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import actions
from jaxloop import outer_loop
from jaxloop import step
from jaxloop import train_loop
from jaxloop import types
import optax
import orbax.checkpoint as ocp

Batch = types.Batch
Output = types.Output
State = types.TrainState


class Model(nn.Module):

  @nn.compact
  def __call__(self, batch):
    return nn.Dense(features=2)(batch['x'])


class CountingStep(step.Step):

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:  # pyrefly: ignore[bad-override]
    return state.replace(step=state.step + 1), None


class MetricsStep(step.Step):
  """Returns a `loss` of 1 / step, as a device array."""

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:  # pyrefly: ignore[bad-override]
    new_step = state.step + 1
    return state.replace(step=new_step), {'loss': 1.0 / new_step}


class HostMetricsCheckpointAction(actions.ContinuousCheckpointAction):
  """Copies metrics to the host, as `best_fn` needs with device arrays."""

  def save(self, step, state, outputs, force=False):
    metrics = jax.tree.map(lambda x: x.tolist(), jax.device_get(outputs))
    return super().save(step, state, metrics, force=force)


class OuterLoopContinuousCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.checkpoint_dir = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.batch = {'x': jnp.ones([2, 3])}
    self.spec = jax.tree.map(lambda x: (x.shape, x.dtype), self.batch)

  def _run(
      self,
      total_steps: int,
      restore: bool,
      best_fn: Optional[Callable[[Any], float]] = None,
  ) -> Tuple[State, ocp.CheckpointManager]:
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(5),
        best_fn=best_fn,
    )
    mngr = ocp.CheckpointManager(self.checkpoint_dir, options=options)
    s = CountingStep(
        jax.random.PRNGKey(0),
        Model(),
        optimizer=optax.sgd(1e-3),
        train=True,
        end_actions=[actions.ContinuousCheckpointAction(mngr)],
    )
    loop = outer_loop.OuterLoop(
        train_loop=train_loop.TrainLoop(s),
        checkpoint_spec=(
            outer_loop.CheckpointSpec(checkpoint_dir=self.checkpoint_dir)
            if restore
            else None
        ),
    )
    state = s.initialize_model(self.spec)
    state, _ = loop(
        state,
        train_dataset=itertools.repeat(self.batch),
        train_total_steps=total_steps,
        train_loop_steps=3,
    )
    # A fresh reader with the same options sees what was written.
    return state, ocp.CheckpointManager(self.checkpoint_dir, options=options)

  def test_final_step_is_saved_and_training_resumes(self):
    state, reader = self._run(total_steps=7, restore=False)
    with reader:
      self.assertEqual(int(state.step), 7)
      # 5 by policy, 7 by the final save.
      self.assertEqual(reader.all_steps(), [5, 7])

    # Restart from the latest save and train to step 12.
    state, reader = self._run(total_steps=12, restore=True)
    with reader:
      self.assertEqual(int(state.step), 12)
      self.assertEqual(reader.all_steps(), [5, 7, 10, 12])

  def test_no_extra_save_when_policy_saved_last_step(self):
    state, reader = self._run(total_steps=10, restore=False)
    with reader:
      self.assertEqual(int(state.step), 10)
      self.assertEqual(reader.all_steps(), [5, 10])

  def test_rejects_manager_shared_with_train_loop_checkpoint_action(self):
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(5)
    )
    with ocp.CheckpointManager(self.checkpoint_dir, options=options) as mngr:
      s = CountingStep(
          jax.random.PRNGKey(0),
          Model(),
          optimizer=optax.sgd(1e-3),
          train=True,
          end_actions=[actions.ContinuousCheckpointAction(mngr)],
      )
      with self.assertRaisesRegex(ValueError, 'same `CheckpointManager`'):
        outer_loop.OuterLoop(
            train_loop=train_loop.TrainLoop(
                s, end_actions=[actions.CheckpointAction(mngr)]
            )
        )

  def test_separate_managers_for_step_and_train_loop_actions(self):
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(5)
    )
    loop_dir = self.checkpoint_dir.parent / 'loop_ckpt'
    step_mngr = ocp.CheckpointManager(self.checkpoint_dir, options=options)
    loop_mngr = ocp.CheckpointManager(loop_dir)
    s = CountingStep(
        jax.random.PRNGKey(0),
        Model(),
        optimizer=optax.sgd(1e-3),
        train=True,
        end_actions=[actions.ContinuousCheckpointAction(step_mngr)],
    )
    loop = outer_loop.OuterLoop(
        train_loop=train_loop.TrainLoop(
            s, end_actions=[actions.CheckpointAction(loop_mngr)]
        )
    )
    state, _ = loop(
        s.initialize_model(self.spec),
        train_dataset=itertools.repeat(self.batch),
        train_total_steps=6,
        train_loop_steps=3,
    )
    self.assertEqual(int(state.step), 6)
    with ocp.CheckpointManager(self.checkpoint_dir, options=options) as reader:
      self.assertEqual(reader.all_steps(), [5, 6])
    with ocp.CheckpointManager(loop_dir) as reader:
      self.assertEqual(reader.all_steps(), [3, 6])

  def test_final_save_stores_last_step_metrics(self):
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(5),
        best_fn=lambda metrics: metrics['loss'],
        best_mode='min',
        max_to_keep=1,
    )
    mngr = ocp.CheckpointManager(self.checkpoint_dir, options=options)
    s = MetricsStep(
        jax.random.PRNGKey(0),
        Model(),
        optimizer=optax.sgd(1e-3),
        train=True,
        end_actions=[HostMetricsCheckpointAction(mngr)],
    )
    loop = outer_loop.OuterLoop(train_loop=train_loop.TrainLoop(s))
    state, _ = loop(
        s.initialize_model(self.spec),
        train_dataset=itertools.repeat(self.batch),
        train_total_steps=7,
        train_loop_steps=3,
    )
    self.assertEqual(int(state.step), 7)
    with ocp.CheckpointManager(self.checkpoint_dir, options=options) as reader:
      # Both saves store one step's `loss`, so `best_fn` can rank them. The
      # loss falls every step, so the final step is the best one kept.
      self.assertEqual(reader.all_steps(), [7])
      metrics = reader.metrics(7)
      self.assertIsNotNone(metrics)
      self.assertAlmostEqual(metrics['loss'], 1.0 / 7, places=5)

  def test_rejects_continuous_action_in_train_loop(self):
    with ocp.CheckpointManager(self.checkpoint_dir) as mngr:
      s = CountingStep(
          jax.random.PRNGKey(0), Model(), optimizer=optax.sgd(1e-3), train=True
      )
      with self.assertRaisesRegex(ValueError, 'train step end action'):
        outer_loop.OuterLoop(
            train_loop=train_loop.TrainLoop(
                s, end_actions=[actions.ContinuousCheckpointAction(mngr)]
            )
        )

  def test_rejects_directory_shared_with_train_loop_checkpoint_action(self):
    with ocp.CheckpointManager(
        self.checkpoint_dir
    ) as step_mngr, ocp.CheckpointManager(self.checkpoint_dir) as loop_mngr:
      s = CountingStep(
          jax.random.PRNGKey(0),
          Model(),
          optimizer=optax.sgd(1e-3),
          train=True,
          end_actions=[actions.ContinuousCheckpointAction(step_mngr)],
      )
      with self.assertRaisesRegex(ValueError, 'or directory'):
        outer_loop.OuterLoop(
            train_loop=train_loop.TrainLoop(
                s, end_actions=[actions.CheckpointAction(loop_mngr)]
            )
        )

  def test_no_save_when_no_step_ran(self):
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(5)
    )
    mngr = ocp.CheckpointManager(self.checkpoint_dir, options=options)
    s = CountingStep(
        jax.random.PRNGKey(0),
        Model(),
        optimizer=optax.sgd(1e-3),
        train=True,
        end_actions=[actions.ContinuousCheckpointAction(mngr)],
    )
    loop = outer_loop.OuterLoop(train_loop=train_loop.TrainLoop(s))
    state, _ = loop(
        s.initialize_model(self.spec),
        train_dataset=iter([]),
        train_total_steps=7,
        train_loop_steps=3,
    )
    self.assertEqual(int(state.step), 0)
    with ocp.CheckpointManager(self.checkpoint_dir, options=options) as reader:
      self.assertEqual(reader.all_steps(), [])

  def _loop_with_mocked_close(self):
    mngr = ocp.CheckpointManager(self.checkpoint_dir)
    action = actions.ContinuousCheckpointAction(mngr)
    s = CountingStep(
        jax.random.PRNGKey(0),
        Model(),
        optimizer=optax.sgd(1e-3),
        train=True,
        end_actions=[action],
    )
    loop = outer_loop.OuterLoop(train_loop=train_loop.TrainLoop(s))
    close = self.enter_context(
        mock.patch.object(mngr, 'close', wraps=mngr.close)
    )
    return loop, s.initialize_model(self.spec), action, close

  def test_closes_manager(self):
    loop, state, _, close = self._loop_with_mocked_close()
    loop(
        state,
        train_dataset=itertools.repeat(self.batch),
        train_total_steps=2,
        train_loop_steps=2,
    )
    close.assert_called_once()

  def test_closes_manager_when_finalize_raises(self):
    loop, state, action, close = self._loop_with_mocked_close()
    self.enter_context(
        mock.patch.object(
            action, 'finalize', side_effect=RuntimeError('finalize failed')
        )
    )
    with self.assertRaisesRegex(RuntimeError, 'finalize failed'):
      loop(
          state,
          train_dataset=itertools.repeat(self.batch),
          train_total_steps=2,
          train_loop_steps=2,
      )
    close.assert_called_once()


if __name__ == '__main__':
  absltest.main()

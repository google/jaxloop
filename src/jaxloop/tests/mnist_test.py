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

"""MNIST model test for train and eval loops."""

import os
import shutil
import tempfile
import threading
import time
from typing import Any, Iterator, Optional, Tuple

from absl.testing import absltest
from etils import epath
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import actions
from jaxloop import eval_loop as eval_loop_lib
from jaxloop import outer_loop as outer_loop_lib
from jaxloop import step as step_lib
from jaxloop import train_loop as train_loop_lib
from jaxloop import types
import optax
from orbax import checkpoint
import tensorflow as tf
import tensorflow_datasets as tfds

Batch = types.Batch
Output = types.Output
State = types.TrainState
Step = step_lib.Step


def mnist_datasets(batch_size, data_dir):
  def map_fn(x):
    return {
        'image': tf.cast(x['image'], tf.float32) / 255.0,
        'label': tf.cast(x['label'], tf.int32),
    }

  train_dir = os.path.join(data_dir, 'train')
  test_dir = os.path.join(data_dir, 'test')
  train_ds = (
      tfds.load('mnist', data_dir=train_dir, split='train', shuffle_files=True)
      .map(map_fn)
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
      .repeat()
  )
  eval_ds = (
      tfds.load('mnist', data_dir=test_dir, split='test', shuffle_files=False)
      .map(map_fn)
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
      .cache()
      .repeat()
  )
  return train_ds, eval_ds


class MnistModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


class MnistStep(Step):

  def begin(self, state: State, batch: Batch) -> tuple[State, Batch]:
    if isinstance(batch['image'], tf.Tensor):
      batch['image'] = batch['image'].numpy()
    if isinstance(batch['label'], tf.Tensor):
      batch['label'] = batch['label'].numpy()
    return state, batch

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:
    images, labels = batch['image'], batch['label']

    def loss_fn(params):
      logits = state.apply_fn({'params': params}, images)
      one_hot = jax.nn.one_hot(labels, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
      return loss, logits

    if self.train:
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, logits), grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)
    else:
      loss, logits = loss_fn(state.params)

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': accuracy}


class EarlyStopTrainLoop(train_loop_lib.TrainLoop):
  """Training loop with an early stop end function."""

  def end(
      self, state: types.TrainState, outputs: Optional[types.Output]
  ) -> Tuple[types.TrainState, Optional[types.Output]]:
    if outputs is not None:
      outputs[outer_loop_lib.STOP_LOOP] = True
    return super().end(state, outputs)


class EarlyStopEvalLoop(eval_loop_lib.EvalLoop):
  """Eval loop with an early stop end function."""

  def end(
      self, state: types.TrainState, outputs: Optional[types.Output]
  ) -> Tuple[types.TrainState, Optional[types.Output]]:
    if outputs is not None:
      outputs[outer_loop_lib.STOP_LOOP] = True
    return super().end(state, outputs)


class IteratorTrainLoop(train_loop_lib.TrainLoop):
  """Training loop with enforced dataset recording logic."""

  def __init__(
      self,
      step: Step,
  ):
    super().__init__(step)
    self.enforced_dataset = None

  def run(
      self,
      state: State,
      dataset: Iterator[Any],
      num_steps: Optional[int] = None,
  ) -> tuple[State, Optional[Output]]:
    self.enforced_dataset = dataset
    return super().run(state, dataset, num_steps)


class EvalFirstOnly:
  """A Callable that returns True only on the first call."""

  def __init__(self):
    self._has_evaled = False

  def set_evaled(self):
    self._has_evaled = True

  def __call__(self, context: outer_loop_lib.EvalContext) -> bool:
    if self._has_evaled:
      return False
    self.set_evaled()
    return True


class MnistTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data_dir = tempfile.mkdtemp()
    cls.train_ds, cls.eval_ds = mnist_datasets(
        batch_size=32, data_dir=cls.data_dir
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    shutil.rmtree(cls.data_dir, ignore_errors=True)

  def setUp(self):
    super().setUp()

    self.model = MnistModel()
    self.train_step = MnistStep(
        jax.random.PRNGKey(0),
        self.model,
        optimizer=optax.sgd(learning_rate=0.005, momentum=0.9),
        train=True,
    )
    self.eval_step = MnistStep(
        jax.random.PRNGKey(0),
        self.model,
        train=False,
    )
    self.state = self.train_step.initialize_model([1, 28, 28, 1])

  def test_train_loop(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    outer_loop = outer_loop_lib.OuterLoop(train_loop=train_loop)
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
    )
    self.assertEqual(state.step, 100)
    self.assertLen(outputs['loss'], 10)
    self.assertLen(outputs['accuracy'], 10)

  def test_early_stop_from_train_loop(self):
    train_loop = EarlyStopTrainLoop(self.train_step)
    outer_loop = outer_loop_lib.OuterLoop(train_loop=train_loop)
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
    )
    self.assertEqual(state.step, 10)
    self.assertTrue(outputs[outer_loop_lib.STOP_LOOP])

  def test_early_stop_from_eval_loop(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    eval_loop = EarlyStopEvalLoop(self.eval_step)
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop]
    )
    eval_specs = [outer_loop_lib.EvalSpec(dataset=self.eval_ds, num_steps=100)]
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )
    self.assertEqual(state.step, 10)
    self.assertTrue(outputs[outer_loop_lib.STOP_LOOP])

  def test_train_loop_continue_from_step(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    outer_loop = outer_loop_lib.OuterLoop(train_loop=train_loop)
    self.state.replace(step=55)
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
    )
    self.assertEqual(state.step, 100)
    self.assertEqual(train_loop.loop_count, 10)
    self.assertLen(outputs['loss'], 10)
    self.assertLen(outputs['accuracy'], 10)

  def test_train_loop_from_checkpoint(self):
    ckpt_writer = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
    ckpt_manager_options = checkpoint.CheckpointManagerOptions(max_to_keep=3)
    checkpoint_dir = epath.Path(
        os.path.join(self.create_tempdir().full_path, 'checkpoints')
    )
    ckpt_manager = checkpoint.CheckpointManager(
        checkpoint_dir, ckpt_writer, ckpt_manager_options
    )

    # Write one checkpoint beforehand.
    step = 100
    ckpt_state = self.train_step.initialize_model([1, 28, 28, 1]).replace(
        step=jnp.array(step)
    )
    ckpt_manager.save(
        step=step,
        items=jax.device_get(ckpt_state),
    )

    train_loop = train_loop_lib.TrainLoop(self.train_step)
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop,
        checkpoint_spec=outer_loop_lib.CheckpointSpec(
            checkpoint_dir=checkpoint_dir
        ),
    )
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
    )
    self.assertEqual(state.step, 100)
    self.assertEqual(train_loop.loop_count, 10)
    self.assertIsNone(outputs)

  def test_train_loop_with_exhausted_dataset(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    outer_loop = outer_loop_lib.OuterLoop(train_loop=train_loop)
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.take(50).as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
    )
    self.assertEqual(state.step, 50)
    self.assertIn('loss', outputs)
    self.assertIn('accuracy', outputs)

  def test_eval_loop(self):
    eval_loop = eval_loop_lib.EvalLoop(self.eval_step)
    _, outputs = eval_loop(
        self.state,
        self.eval_ds.as_numpy_iterator(),
        num_steps=100,
    )
    self.assertEqual(eval_loop.loop_count, 1)
    self.assertLen(outputs['loss'], 100)
    self.assertLen(outputs['accuracy'], 100)

  def test_train_and_eval_loop(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    eval_loop = eval_loop_lib.EvalLoop(self.eval_step)
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop]
    )
    eval_specs = [outer_loop_lib.EvalSpec(dataset=self.eval_ds, num_steps=100)]
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )
    self.assertEqual(eval_loop.loop_count, 10)
    self.assertEqual(state.step, 100)
    self.assertLen(outputs['loss'], 10)
    self.assertLen(outputs['accuracy'], 10)

  def test_continuous_eval(self):
    checkpoints_count = 2

    ckpt_writer = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
    ckpt_manager_options = checkpoint.CheckpointManagerOptions(
        max_to_keep=checkpoints_count
    )
    checkpoint_dir = epath.Path(
        os.path.join(self.create_tempdir().full_path, 'checkpoints')
    )
    ckpt_manager = checkpoint.CheckpointManager(
        checkpoint_dir, ckpt_writer, ckpt_manager_options
    )

    def spawn_checkpoints():
      initial_state = self.train_step.initialize_model([1, 28, 28, 1])
      for step in range(1, checkpoints_count + 1):
        ckpt_manager.save(
            step=step,
            items=jax.device_get(initial_state),
        )
        time.sleep(3)
      f = epath.Path(
          os.path.join(checkpoint_dir, outer_loop_lib._STOP_FILE_NAME)
      ).open('w')
      f.close()

    def test_should_eval(context: outer_loop_lib.EvalContext) -> bool:
      return context.step_num % 2 == 0

    eval_loop1 = eval_loop_lib.EvalLoop(self.eval_step)
    eval_loop2 = eval_loop_lib.EvalLoop(self.eval_step)
    eval_loop3 = eval_loop_lib.EvalLoop(self.eval_step)
    eval_loop4 = eval_loop_lib.EvalLoop(self.eval_step)
    outer_loop = outer_loop_lib.OuterLoop(
        eval_loops=[eval_loop1, eval_loop2, eval_loop3, eval_loop4],
        checkpoint_spec=outer_loop_lib.CheckpointSpec(
            checkpoint_dir=checkpoint_dir
        ),
    )

    never_eval_callable = EvalFirstOnly()
    never_eval_callable.set_evaled()

    eval_specs = [
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds.as_numpy_iterator(), num_steps=100
        ),
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds.as_numpy_iterator(),
            num_steps=100,
            should_eval_fn=test_should_eval,
        ),
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds.as_numpy_iterator(),
            num_steps=100,
            should_eval_fn=EvalFirstOnly(),
        ),
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds.as_numpy_iterator(),
            num_steps=100,
            should_eval_fn=never_eval_callable,
        ),
    ]
    threading.Thread(target=spawn_checkpoints).start()
    ckpt_manager.wait_until_finished()
    time.sleep(3)
    state, outputs = outer_loop(
        self.state,
        eval_specs=eval_specs,
    )

    self.assertEqual(eval_loop1.loop_count, checkpoints_count)
    self.assertEqual(eval_loop2.loop_count, checkpoints_count // 2)
    self.assertEqual(eval_loop3.loop_count, 1)
    self.assertEqual(eval_loop4.loop_count, 0)

    self.assertEqual(state.step, 0)
    self.assertLen(outputs['loss'], 100)
    self.assertLen(outputs['accuracy'], 100)

  def test_single_shot_eval(self):
    ckpt_writer = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
    ckpt_manager_options = checkpoint.CheckpointManagerOptions(max_to_keep=3)
    checkpoint_dir = epath.Path(
        os.path.join(self.create_tempdir().full_path, 'checkpoints')
    )
    ckpt_manager = checkpoint.CheckpointManager(
        checkpoint_dir, ckpt_writer, ckpt_manager_options
    )

    # Write one checkpoint beforehand.
    initial_state = self.train_step.initialize_model([1, 28, 28, 1])
    ckpt_manager.save(
        step=1,
        items=jax.device_get(initial_state),
    )

    eval_loop = eval_loop_lib.EvalLoop(self.eval_step)
    outer_loop = outer_loop_lib.OuterLoop(
        eval_loops=[eval_loop],
        checkpoint_spec=outer_loop_lib.CheckpointSpec(
            checkpoint_dir=checkpoint_dir,
            iterate_stop_fn=lambda: True,
        ),
    )
    eval_specs = [outer_loop_lib.EvalSpec(dataset=self.eval_ds, num_steps=100)]
    # Eval once then quit.
    state, outputs = outer_loop(
        self.state,
        eval_specs=eval_specs,
    )
    self.assertEqual(eval_loop.loop_count, 1)
    self.assertEqual(state.step, 0)
    self.assertLen(outputs['loss'], 100)
    self.assertLen(outputs['accuracy'], 100)

  def test_iterator_conversion(self):
    train_loop = IteratorTrainLoop(self.train_step)
    eval_loop = eval_loop_lib.EvalLoop(self.eval_step)
    eval_specs = [outer_loop_lib.EvalSpec(dataset=self.eval_ds, num_steps=100)]
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop]
    )
    state, _ = outer_loop(
        self.state,
        train_dataset=tfds.as_numpy(self.train_ds),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )
    self.assertEqual(state.step, 100)
    self.assertIsInstance(train_loop.enforced_dataset, Iterator)

  def test_iterator_input_is_enforced_dataset(self):
    train_loop = IteratorTrainLoop(self.train_step)
    eval_loop = eval_loop_lib.EvalLoop(self.eval_step)
    train_ds_iterator = self.train_ds.as_numpy_iterator()
    eval_ds_iterator = self.eval_ds.as_numpy_iterator()
    eval_specs = [
        outer_loop_lib.EvalSpec(dataset=eval_ds_iterator, num_steps=100)
    ]
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop]
    )
    state, _ = outer_loop(
        self.state,
        train_dataset=train_ds_iterator,
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )
    self.assertEqual(state.step, 100)
    self.assertIs(train_ds_iterator, train_loop.enforced_dataset)

  class TestAction(actions.Action):
    """An action for testing purpose.

    It records how many times the action is triggered.
    """

    def __init__(self, interval: int = 1):
      super().__init__(interval=interval)
      self.counter = 0

    def __call__(
        self, state: State, outputs: Optional[Output], **kwargs
    ) -> Optional[Any]:
      self.counter += 1

  def test_multi_eval(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    all_actions = [self.TestAction(), self.TestAction()]

    eval_loop1 = eval_loop_lib.EvalLoop(
        self.eval_step, end_actions=[all_actions[0]]
    )
    eval_loop2 = eval_loop_lib.EvalLoop(
        self.eval_step, end_actions=[all_actions[1]]
    )
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop1, eval_loop2]
    )

    eval_specs = [
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds, num_steps=10, eval_loop_interval=1
        ),
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds, num_steps=10, eval_loop_interval=2
        ),
    ]
    state, _ = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )

    self.assertEqual(state.step, 100)
    self.assertEqual(all_actions[0].counter, 10)
    self.assertEqual(all_actions[1].counter, 5)

  def test_should_eval(self):
    train_loop = train_loop_lib.TrainLoop(self.train_step)
    all_actions = [
        self.TestAction(),
        self.TestAction(),
    ]

    eval_loop1 = eval_loop_lib.EvalLoop(
        self.eval_step, end_actions=[all_actions[0]]
    )
    eval_loop2 = eval_loop_lib.EvalLoop(
        self.eval_step, end_actions=[all_actions[1]]
    )
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop,
        eval_loops=[eval_loop1, eval_loop2],
    )

    eval_specs = [
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds, num_steps=100, eval_loop_interval=10
        ),
        outer_loop_lib.EvalSpec(dataset=self.eval_ds, num_steps=100),
    ]
    state, _ = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )

    self.assertEqual(state.step, 100)
    self.assertEqual(all_actions[0].counter, 1)
    self.assertEqual(all_actions[1].counter, 10)

  def test_train_loop_with_action(self):
    all_actions = [
        self.TestAction(interval=2),
        self.TestAction(interval=2),
        self.TestAction(interval=5),
        self.TestAction(interval=1),
    ]
    begin_actions = [all_actions[0]]
    end_actions = [all_actions[1], all_actions[2]]
    train_loop = train_loop_lib.TrainLoop(
        self.train_step, begin_actions=begin_actions, end_actions=end_actions
    )
    eval_actions = [all_actions[3]]
    eval_loop = eval_loop_lib.EvalLoop(self.eval_step, end_actions=eval_actions)
    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop, eval_loops=[eval_loop]
    )
    eval_specs = [
        outer_loop_lib.EvalSpec(
            dataset=self.eval_ds, num_steps=100, eval_loop_interval=5
        )
    ]
    state, outputs = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=100,
        train_loop_steps=10,
        eval_specs=eval_specs,
    )
    self.assertEqual(state.step, 100)
    self.assertLen(outputs['loss'], 10)
    self.assertLen(outputs['accuracy'], 10)
    self.assertEqual(all_actions[0].counter, 5)
    self.assertEqual(all_actions[1].counter, 5)
    self.assertEqual(all_actions[2].counter, 2)
    self.assertEqual(all_actions[3].counter, 2)

  def test_checkpointing(self):
    ckpt_writer = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
    ckpt_manager_options = checkpoint.CheckpointManagerOptions(max_to_keep=2)
    checkpoint_dir = epath.Path(
        os.path.join(self.create_tempdir().full_path, 'checkpoints')
    )
    ckpt_manager = checkpoint.CheckpointManager(
        checkpoint_dir, ckpt_writer, ckpt_manager_options
    )

    train_loop = train_loop_lib.TrainLoop(
        step=self.train_step,
        begin_actions=[],
        end_actions=[actions.CheckpointAction(ckpt_manager)],
    )

    outer_loop = outer_loop_lib.OuterLoop(
        train_loop=train_loop,
    )
    state, _ = outer_loop(
        self.state,
        train_dataset=self.train_ds.as_numpy_iterator(),
        train_total_steps=10,
        train_loop_steps=2,
    )
    self.assertEqual(state.step, 10)

    restored = self.train_step.restore_model(
        self.state, checkpoint_dir, step=10
    )
    self.assertEqual(restored.step, 10)
    self.assertEqual(
        list(restored.params.keys()),
        ['Conv_0', 'Conv_1', 'Dense_0', 'Dense_1'],
    )


if __name__ == '__main__':
  absltest.main()

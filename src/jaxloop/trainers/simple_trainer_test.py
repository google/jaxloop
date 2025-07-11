from typing import Optional, Tuple

from absl.testing import absltest
from etils import epath
import flax.linen as nn
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jaxloop import partition
from jaxloop import train_loop as train_loop_lib
from jaxloop import types
from jaxloop.trainers import data_loader
from jaxloop.trainers import simple_trainer
from jaxloop.trainers import trainer_utils


class TestModel(nn.Module):
  """A fully-connected neural network model with 3 layers."""

  with_dropout: bool = False

  @nn.compact
  def __call__(self, x, train: bool = True, **kwargs):
    for _ in range(3):
      x = nn.Dense(features=4)(x)
      x = nn.relu(x)
      if self.with_dropout:
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
    x = nn.Dense(features=1)(x)
    return x


class TestTrainLoop(train_loop_lib.TrainLoop):
  calls = 0

  def end(
      self, state: types.TrainState, outputs: Optional[types.Output]
  ) -> Tuple[types.TrainState, Optional[types.Output]]:
    TestTrainLoop.calls += 1
    return super().end(state, outputs)


class SimpleTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.epochs = 2
    self.steps_per_epoch = 3

    self.model = TestModel(with_dropout=False)
    self.spec = {'input_features': [2, 3], 'output_features': [2, 1]}
    self.batches = [
        {
            'input_features': jnp.ones([2, 3]),
            'output_features': jnp.ones([2, 1]),
        }
        for _ in range(self.epochs * self.steps_per_epoch)
    ]
    self.dataset = {
        'input_features': jnp.ones([self.epochs * self.steps_per_epoch, 3]),
        'output_features': jnp.ones([self.epochs * self.steps_per_epoch, 1]),
    }

  def test_basic_training(self):
    trainer = simple_trainer.SimpleTrainer(
        self.model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
    )

    outputs = trainer.train(self.batches)
    self.assertEqual(
        trainer.model_state.step, self.epochs * self.steps_per_epoch
    )
    self.assertIsNotNone(outputs)
    self.assertIsNotNone(outputs['loss'])
    self.assertLen(outputs['loss'], self.steps_per_epoch)

  def test_basic_training_with_prng(self):
    model = TestModel(with_dropout=True)

    trainer = simple_trainer.SimpleTrainer(
        model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
        base_prng={
            'params': jax.random.PRNGKey(0),
            'dropout': jax.random.PRNGKey(1),
        },
    )

    outputs = trainer.train(self.batches)
    self.assertEqual(
        trainer.model_state.step, self.epochs * self.steps_per_epoch
    )
    self.assertIsNotNone(outputs)
    self.assertIsNotNone(outputs['loss'])
    self.assertLen(outputs['loss'], self.steps_per_epoch)

  def test_custom_loops(self):
    trainer = simple_trainer.SimpleTrainer(
        self.model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
        train_loop_class=TestTrainLoop,
    )

    trainer.train(self.batches)
    self.assertEqual(
        trainer.model_state.step, self.epochs * self.steps_per_epoch
    )
    self.assertEqual(TestTrainLoop.calls, self.epochs)

  def test_partitioning(self):
    data_parallel_mesh = sharding.Mesh(
        mesh_utils.create_device_mesh((jax.device_count(),)), ('data',)
    )

    trainer = simple_trainer.SimpleTrainer(
        self.model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
        partitioner=partition.DataParallelPartitioner(
            data_parallel_mesh, 'data'
        ),
    )

    trainer.train(self.batches)
    self.assertEqual(
        trainer.model_state.step, self.epochs * self.steps_per_epoch
    )

  def test_checkpointing(self):
    checkpoint_dir = epath.Path(self.create_tempdir().full_path, 'checkpoint')

    trainer = simple_trainer.SimpleTrainer(
        self.model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
        checkpointing_config=trainer_utils.CheckpointingConfig(
            checkpoint_dir,
            checkpoint_interval=1,
            max_checkpoints=2,
        ),
    )

    trainer.train(self.batches)
    self.assertEqual(
        trainer.model_state.step, self.steps_per_epoch * self.epochs
    )
    self.assertTrue(checkpoint_dir.exists())
    self.assertTrue(epath.Path(checkpoint_dir, 'checkpoints').exists())
    self.assertLen(list(epath.Path(checkpoint_dir, 'checkpoints').iterdir()), 2)

  def test_with_simple_data_loader(self):
    trainer = simple_trainer.SimpleTrainer(
        self.model,
        epochs=self.epochs,
        steps_per_epoch=self.steps_per_epoch,
        batch_spec=self.spec,
    )

    data_iterator = iter(self.batches)
    simple_loader = data_loader.SimpleDataLoader(dataset=data_iterator)

    outputs = trainer.train(simple_loader)
    self.assertEqual(
        trainer.model_state.step, self.epochs * self.steps_per_epoch
    )
    self.assertIsNotNone(outputs)
    self.assertIsNotNone(outputs['loss'])
    self.assertLen(outputs['loss'], self.steps_per_epoch)


if __name__ == '__main__':
  absltest.main()

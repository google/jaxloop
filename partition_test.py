"""Unit test for jaxloop partition."""

from typing import Any
from absl.testing import absltest
import flax.linen as nn
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jaxloop import partition
from jaxloop import types
import numpy as np
import optax

Batch = types.Batch
TrainState = types.TrainState

Mesh = sharding.Mesh


class SimpleModel(nn.Module):
  """A simple model in FLAX."""

  @nn.compact
  def __call__(self, x: Any):
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


def create_train_state(
    model: nn.Module, partitioner: partition.Partitioner
) -> TrainState:
  batch = jnp.zeros([1, 28, 28, 1])
  optimizer = optax.sgd(learning_rate=0.005, momentum=0.9)
  return partitioner.shard_init_fn(
      lambda batch: types.TrainState.create(
          apply_fn=model.apply,
          tx=optimizer,
          **model.init(jax.random.key(0), batch)
      )
  )(batch)


class PartitionTest(absltest.TestCase):

  def test_access_sharding_and_mesh_properties(self):
    device_count = jax.device_count()
    # Test sharding and mesh properties in data parallel partitioner.
    data_parallel_mesh = Mesh(
        mesh_utils.create_device_mesh((device_count,)), ("data",)
    )
    data_parallel_partitioner = partition.DataParallelPartitioner(
        mesh=data_parallel_mesh, data_axis="data"
    )
    self.assertEqual(data_parallel_partitioner.mesh, data_parallel_mesh)
    self.assertEqual(
        data_parallel_partitioner.sharding,
        sharding.NamedSharding(data_parallel_mesh, sharding.PartitionSpec()),
    )
    # Test sharding and mesh properties in SPMD partitioner.
    spmd_mesh = Mesh(
        mesh_utils.create_device_mesh((device_count // 2, 2)), ("data", "model")
    )
    spmd_partitioner = partition.SPMDPartitioner(
        mesh=spmd_mesh, data_axis="data"
    )
    self.assertEqual(spmd_partitioner.mesh, spmd_mesh)
    _ = create_train_state(SimpleModel(), spmd_partitioner)
    # After explicit sharding propagation, partition sharding becomes TrainState
    # which contains sharding annotations of all parameters at tree leaves.
    self.assertEqual(
        spmd_partitioner.sharding.params["Dense_0"]["kernel"],  # pytype: disable=attribute-error
        sharding.NamedSharding(spmd_mesh, sharding.PartitionSpec()),
    )

  def test_data_parallel_partioner(self):
    # partition() expects a Callable[[TrainState, Batch], Any]
    def fn(state: TrainState, batch: Batch) -> Any:
      x = batch["input"]
      y = jnp.ones((128, 256))
      return {"dot_output": jnp.matmul(x, y), "input_batch": x}

    num_local_devices = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((num_local_devices,)), ("data",))
    partitioner = partition.DataParallelPartitioner(mesh=mesh, data_axis="data")

    input_arg = {"input": np.ones((4096, 128))}

    input_sharded = partitioner.shard_batch(input_arg)
    # The first dim of the test array is sharded onto 8 TPU cores.
    self.assertEqual(input_sharded["input"].shape, (4096, 128))
    expected_input_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("data")
    )
    self.assertEqual(input_sharded["input"].sharding, expected_input_sharding)
    self.assertEqual(
        input_sharded["input"].addressable_data(0).shape, (512, 128)
    )

    p_fn = partitioner.partition(fn)

    # a replicated train_state
    train_state = create_train_state(SimpleModel(), partitioner)
    self.assertNotEmpty(jax.tree.leaves(train_state.params))
    for leaf in jax.tree.leaves(train_state.params):
      self.assertLen(leaf.devices(), num_local_devices)

    outputs = p_fn(train_state, input_sharded)

    expected_output_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("data")
    )
    self.assertEqual(outputs["input_batch"].sharding, expected_output_sharding)
    self.assertEqual(outputs["dot_output"].sharding, expected_output_sharding)
    self.assertEqual(
        outputs["input_batch"].addressable_data(0).shape, (512, 128)
    )
    self.assertEqual(
        outputs["dot_output"].addressable_data(0).shape, (512, 256)
    )

    host_outputs = jax.device_get(outputs)
    self.assertIsInstance(host_outputs["dot_output"], np.ndarray)
    self.assertIsInstance(host_outputs["input_batch"], np.ndarray)
    self.assertEqual(host_outputs["dot_output"].shape, (4096, 256))
    self.assertEqual(host_outputs["input_batch"].shape, (4096, 128))


if __name__ == "__main__":
  absltest.main()

"""Utility classes and functions related to partitioning."""

import abc
from typing import Any, Callable

from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
from jax import sharding
from jax import stages
from jaxloop import types
import jaxtyping
import numpy as np

Batch = types.Batch
TrainState = types.TrainState


class Partitioner(abc.ABC):
  """An abstract class defining partitioning logic for data and computation."""

  @abc.abstractmethod
  def shard_init_fn(
      self, init_fn: Callable[[Batch], TrainState]
  ) -> Callable[[Batch], TrainState]:
    """Shards the initialization function."""
    pass

  @abc.abstractmethod
  def shard_batch(self, batch: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Shards the input batch."""
    pass

  @abc.abstractmethod
  def partition(
      self, fn: Callable[[TrainState, Batch], Any], **kwargs
  ) -> stages.Wrapped:
    """Shards the model computation."""
    pass

  @property
  @abc.abstractmethod
  def sharding(self) -> sharding.Sharding | None:
    """Gets the sharding."""
    pass

  @property
  @abc.abstractmethod
  def mesh(self) -> jax.sharding.Mesh | None:
    """Gets the mesh."""
    pass


class SingleDevicePartitioner(Partitioner):
  """A simple partitioner for single device use case."""

  def __init__(self) -> None:
    self._sharding = None
    self._mesh = None

  def shard_init_fn(
      self, init_fn: Callable[[Batch], TrainState]
  ) -> Callable[[Batch], TrainState]:
    """Shard the initialization function."""
    return jax.jit(init_fn)

  def shard_batch(self, batch: jaxtyping.PyTree) -> jaxtyping.PyTree:
    return batch

  def partition(
      self, fn: Callable[[TrainState, Batch], Any], **kwargs
  ) -> stages.Wrapped:
    return jax.jit(fn, **kwargs)

  @property
  def sharding(self) -> sharding.Sharding | None:
    return self._sharding

  @property
  def mesh(self) -> jax.sharding.Mesh | None:
    return self._mesh


class DataParallelPartitioner(SingleDevicePartitioner):
  """A simple sharding wrapper for data parallel use case."""

  def __init__(
      self,
      mesh: sharding.Mesh,
      data_axis: str,
  ) -> None:
    super().__init__()
    self._mesh = mesh
    self._data_axis = data_axis
    self._sharding = sharding.NamedSharding(
        self._mesh, sharding.PartitionSpec()
    )

  def shard_init_fn(
      self, init_fn: Callable[[Batch], TrainState]
  ) -> Callable[[Batch], TrainState]:
    """Shard the initialization function across all devices."""
    return jax.jit(
        init_fn,
        out_shardings=sharding.NamedSharding(
            self._mesh, sharding.PartitionSpec()
        ),
    )

  def shard_batch(self, batch: jaxtyping.PyTree) -> jaxtyping.PyTree:
    return _reshard_for_pjit(self._mesh, self._data_axis, batch)

  def partition(
      self, fn: Callable[[TrainState, Batch], Any], **kwargs
  ) -> stages.Wrapped:
    return jax.jit(
        fn,
        in_shardings=(
            self._sharding,  # state
            sharding.NamedSharding(
                self._mesh, sharding.PartitionSpec(self._data_axis)
            ),  # data
        ),
        donate_argnames='state',
        **kwargs,
    )


class SPMDPartitioner(SingleDevicePartitioner):
  """A simple sharding wrapper for SPMD use case."""

  def __init__(
      self,
      mesh: sharding.Mesh,
      data_axis: str,
  ) -> None:
    super().__init__()
    self._mesh = mesh
    self._data_axis = data_axis

  def shard_init_fn(
      self, init_fn: Callable[[Batch], TrainState]
  ) -> Callable[[Batch], TrainState]:
    """Shard the initialization function."""

    @jax.jit
    def sharded_init(*args):
      state = init_fn(*args)
      # TrainingState uses a python-integer "step" counter which needs to be
      # converted into a proper jax array for sharding purposes.
      state = state.replace(step=jnp.array(state.step))
      partition_specs = nn.get_partition_spec(state)
      self._sharding = jax.tree.map(
          lambda x: sharding.NamedSharding(self._mesh, x), partition_specs
      )
      return lax.with_sharding_constraint(state, self._sharding)

    return sharded_init

  def shard_batch(self, batch: jaxtyping.PyTree) -> jaxtyping.PyTree:
    return _reshard_for_pjit(self._mesh, self._data_axis, batch)

  def partition(
      self, fn: Callable[[TrainState, Batch], Any], **kwargs
  ) -> stages.Wrapped:
    return jax.jit(
        fn,
        in_shardings=(
            self._sharding,  # state
            sharding.NamedSharding(
                self._mesh, sharding.PartitionSpec(self._data_axis)
            ),  # data
        ),
        donate_argnames='state',
        **kwargs,
    )


def _reshard_for_pjit(mesh, data_axis, batch):
  """Reshard the data for pjit use case."""

  local_devices = mesh.local_devices
  local_device_count = jax.local_device_count()

  def _get_global_input_shape_dtype(x):
    """Get global input shape/dtype assuming fully sharded batch dim."""
    assert len(x.shape) >= 1
    # Assume fully sharded batch dim.
    # x.shape[0] is per host batch size.
    global_batch_size = x.shape[0] * (
        mesh.shape[data_axis] // jax.local_device_count()
    )
    x_shape = (global_batch_size,) + x.shape[1:]
    return jax.ShapeDtypeStruct(x_shape, x.dtype)

  global_shapes = jax.tree_util.tree_map(_get_global_input_shape_dtype, batch)

  def _put_to_devices(x):
    try:
      per_device_arrays = np.split(x, local_device_count, axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f'Unable to put to devices shape {x.shape} with '
          f'local device count {local_device_count}'
      ) from array_split_error
    device_buffers = [
        jax.device_put(arr, d)
        for arr, d in zip(per_device_arrays, local_devices)
    ]
    return device_buffers

  device_buffers = jax.tree_util.tree_map(_put_to_devices, batch)

  def _shard(global_shape, dbs):
    named_sharding = sharding.NamedSharding(
        mesh,
        sharding.PartitionSpec(
            data_axis,
        ),
    )
    return jax.make_array_from_single_device_arrays(
        global_shape.shape, named_sharding, dbs
    )

  return jax.tree_util.tree_map(_shard, global_shapes, device_buffers)

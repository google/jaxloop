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

"""The step library for JAX models."""

import typing
from typing import Any, Mapping, Optional, Protocol, Tuple, Type

from absl import logging
from etils import epath
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jaxloop import actions
from jaxloop import partition
from jaxloop import types
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils

DType = types.DType
Batch = types.Batch
BatchSpec = types.BatchSpec
Output = types.Output
State = types.TrainState

# pylint: disable=logging-fstring-interpolation


def get_zeroed_batch(spec: BatchSpec) -> Batch:
  """Returns a zeroed batch based on the input data spec."""

  # Prevent treating namedtuples as potential specs.
  is_plain_tuple = lambda x: isinstance(x, tuple) and not hasattr(x, '_fields')

  def is_shape(spec):
    if isinstance(spec, int):
      return True
    return (is_plain_tuple(spec) or isinstance(spec, list)) and all(
        isinstance(i, int) for i in spec
    )

  def map_fn(elem):
    if is_shape(elem):
      shape, dtype = elem, jnp.float32
    else:
      shape, dtype = elem
    return jnp.zeros(shape, dtype=dtype)

  def is_leaf(spec):
    if is_shape(spec):
      return True
    if is_plain_tuple(spec) and len(spec) == 2:
      return is_leaf(spec[0])
    return False

  return jax.tree.map(map_fn, spec, is_leaf=is_leaf)


@typing.runtime_checkable
class Step(Protocol):
  """The step class for JAX Linen models.

  This class is used to define the training or evaluation step for a JAX model
  and is designed to be used with the `Loop` class in the loop library.

  Example usage (nn.Module):
  ```
  class TestModel(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(features=10)(x)
      x = nn.log_softmax(x)
      return x

  self.model = TestModel()
  self.step = TestStep(
        jax.random.PRNGKey(0), self.model, optimizer=optax.adam(1e-4))
  ```

  Example usage (nnx.Module):
  ```
  class TestModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
      self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
      x = self.linear(x)
      x = nnx.log_softmax(x)
      return x

  self.step = TestStep(
        jax.random.PRNGKey(0),
        TestModel,
        optimizer=optax.adam(1e-4),
        model_args=(1, 10),
    )
  ```

  Parameters:
    base_prng: The base prng key or a list of prng keys.
    model: The model to be trained or evaluated; either as a nn.Module or a
      nnx.Module Type.
    optimizer: The optimizer to be used for training.
    partitioner: The partitioner to be used for sharding the model.
    train: Whether the step is for training or evaluation.
    should_shard_batch: Whether the batch should be sharded before running the
      step. Users can set this to `False` if they want to manually control batch
      sharding by calling `shard_batch`.
    nnx_model_args: Model __init__ arguments for nnx.Module. Note: this field is
      only used when model is an nnx.Module Type and ignored otherwise.
    nnx_model_kwargs: Model _init__ keyword arguments for nnx.Module. Note: this
      field is only used when model is an nnx.Module Type and ignored otherwise.
    begin_actions: The actions to be run at the beginning of the step.
    end_actions: The actions to be run at the end of the step. Note, including
      such actions, when triggered on a particular step, will require all JAX
      computations to be finished before the action can be invoked. This can
      cause a slight performance penalty for such steps.
    chkpt_item_name: The name of the checkpoint item to be restored. See Orbax
      Checkpointer for more details.
  """

  _STATE_KEYS = ('step', 'params', 'batch_stats')

  def __init__(
      self,
      base_prng: types.PRNGType,
      model: nn.Module | Type[nnx.Module],
      optimizer: optax.GradientTransformation = optax.identity(),
      partitioner: partition.Partitioner = partition.SingleDevicePartitioner(),
      train: bool = False,
      should_shard_batch: bool = True,
      nnx_model_args: Optional[Tuple[Any, ...]] = None,
      begin_actions: Optional[list[actions.Action]] = None,
      end_actions: Optional[list[actions.Action]] = None,
      chkpt_item_name: str = 'default',
      **nnx_model_kwargs: Any,
  ):
    self._nnx_precheck(model, nnx_model_args)

    self._base_prng = base_prng
    self._model = (
        model
        if isinstance(model, nn.Module)
        else nnx.bridge.ToLinen(model, nnx_model_args, kwargs=nnx_model_kwargs)
    )
    self._optimizer = optimizer
    self._partitioner = partitioner
    # TODO(b/369260798): Simplify should_shard_batch in step.
    # Whether the batch should be sharded before running the step. Users can
    # set this to `False` if they want to manually control batch sharding by
    # calling `shard_batch`.
    self._should_shard_batch = should_shard_batch
    self._train = train
    self._cached_run = None
    self._num_params = None
    self._num_flops = None
    self._begin_actions = begin_actions
    self._end_actions = end_actions
    self._chkpt_item_name = chkpt_item_name

  def _nnx_precheck(
      self,
      model: nn.Module | Type[nnx.Module],
      model_args: Optional[tuple[Any, ...]],
  ):
    if isinstance(model, type) and model_args is None:
      raise ValueError(
          'To initialize an nnx.Module, model_args must be provided.'
      )
    elif not isinstance(model, type) and model_args is not None:
      raise ValueError(
          'model_args should only be provided when initializing an nnx.Module.'
      )

  def preprocess_batch(self, batch: Batch) -> Batch:
    """Preprocesses the input data batch before running the step."""
    return batch

  def initialize_model(
      self, spec: BatchSpec, log_num_params: bool = False, **kwargs
  ) -> State:
    """Initializes the model state based on the input data batch.

    Args:
      spec: The input data spec.
      log_num_params: Whether to log the number of parameters of the model.
      **kwargs: Additional keyword arguments to initialize the model.

    Returns:
      The model state.
    """

    def init_fn(batch):
      variables = self._model.init(self._base_prng, batch, **kwargs)
      return State.create(
          apply_fn=self._model.apply,
          tx=self._optimizer,
          **{k: v for k, v in variables.items() if k in self._STATE_KEYS},
      )

    batch = get_zeroed_batch(spec)
    batch = self.preprocess_batch(batch)
    if self._should_shard_batch:
      batch = self.shard_batch(batch)
    state = self._partitioner.shard_init_fn(init_fn)(batch)
    if log_num_params:
      if self._num_params is None:
        self.compute_num_params(state)
      logging.info(f'Initialized model with {self.num_params} parameters.')
    return state

  def restore_model(
      self,
      state: State,
      checkpoint_dir: epath.PathLike,
      step: Optional[int] = None,
      transforms: Optional[Any] = None,
  ) -> State:
    """Restores the model state from a checkpoint.

    Args:
      state: The model state.
      checkpoint_dir: The checkpoint directory.
      step: An optional integer specifying the checkpoint step.
      transforms: An optional transform representation to be applied.

    Returns:
      The model state containing the restored checkpoint. If no checkpoint is
      found, the originally given model state is returned.
    """
    step = step or self._latest_step(checkpoint_dir)
    if step is None:
      logging.info(
          f'No checkpoint found in {checkpoint_dir}. Returning '
          'the original model state.'
      )
      return state

    logging.info(f'Restoring model from {checkpoint_dir} at step {step}.')

    if self._partitioner.sharding is None:
      # SingleDevicePartitioner
      sharding_tree = None
    elif isinstance(self._partitioner.sharding, jax.sharding.Sharding):
      # DataParallelPartitioner
      sharding_tree = jax.tree.map(lambda x: self._partitioner.sharding, state)
    else:
      # SPMDPartitioner
      sharding_tree = self._partitioner.sharding

    restore_args = checkpoint_utils.construct_restore_args(
        state, sharding_tree=sharding_tree
    )
    restore_kwargs = {
        'restore_args': restore_args,
        'transforms': transforms or {r'params/.*': ocp.Transform()},
    }

    # Create the abstract state and release the TPU memory. This is important
    # for models that are bounded by HBM usage.
    abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
    jax.tree.map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None, state
    )

    return self._restore(
        step,
        checkpoint_dir,
        abstract_state=abstract_state,
        restore_kwargs=restore_kwargs,
    )

  def prng_key(self, step: int) -> jax.Array:
    """Returns the prng key used inside the training step.

    This is deterministic based on the step ID.

    Args:
      step: The number indicating the current step ID.

    Returns:
      The prng keys for random number generations.
    """
    return jax.tree.map(lambda x: jax.random.fold_in(x, step), self._base_prng)

  def compile(self, **kwargs):
    """This method explicitly calls jax.jit to compile the step function.

    Call this method explicitly if the program needs more jax.jit compile flags
    besides input sharding.

    Args:
      **kwargs: The compile flags used by jax.jit
    """
    self._cached_run = self._partitioner.partition(self.run, **kwargs)

  def begin(self, state: State, batch: Batch) -> Tuple[State, Batch]:
    """Begins the step.

    This method should be overridden if custom `begin` logic is needed.

    Args:
      state: The model state.
      batch: The input data batch.

    Returns:
      A tuple of the model state and input data batch.
    """
    return state, batch

  def _run_begin_actions(self, state: State, step: int):
    if self._begin_actions is not None:
      for action in self._begin_actions:
        if step % action.interval == 0:
          action(state, None)

  def run(
      self, state: State, batch: Batch, **kwargs
  ) -> Tuple[State, Optional[Output]]:
    """Runs the step.

    This method should be overridden in the subclass to define the train or eval
    step for the model. The overridden method must be jittable and will be
    wrapped into `jax.jit`.

    Args:
      state: The model state.
      batch: The input data batch.
      **kwargs: Additional keyword arguments for running the step.

    Returns:
      A tuple of the model state and output.
    """
    raise NotImplementedError('`run` method is not implemented.')

  def end(
      self, state: State, outputs: Optional[Output]
  ) -> Tuple[State, Optional[Output]]:
    """Ends the step.

    This method should be overridden if custom `end` logic is needed.

    Args:
      state: The model state.
      outputs: The model output.

    Returns:
      A tuple of the model state and output.
    """
    return state, outputs

  def _run_end_actions(
      self, state: State, outputs: Optional[Output], step: int
  ):
    if self._end_actions is not None:
      for action in self._end_actions:
        if step % action.interval == 0:
          jax.block_until_ready(outputs)
          action(state, outputs)

  def shard_batch(self, batch: Batch) -> Batch:
    """Shards the input data batch based on the partitioner."""
    return self._partitioner.shard_batch(batch)

  def __call__(
      self,
      state: State,
      batch: Batch,
      per_loop_step_number: int = 0,
      log_num_flops: bool = False,
      **kwargs,
  ) -> Tuple[State, Optional[Output]]:
    """Invokes the step.

    This method invokes the step defined in the subclass by running the `begin`,
    `run` and `end` methods in order. It calls `jax.jit` on the `run` method
    using the partitioner and shards the input data batch based on the
    partitioner. This method should generally not be overriden and always be
    used to invoke the step.

    Args:
      state: The model state.
      batch: The input data batch.
      per_loop_step_number: The number indicating the current step within the
        loop.
      log_num_flops: Whether to log the number of flops of the jitted `run`
        function.
      **kwargs: Additional keyword arguments for running the step.
        batch_preprocessed (optional args in kwargs): whether the batch was
        already preprocessed. If True, the batch will not be preprocessed again
        (e.g. could be done as a performance optimization)

    Returns:
      A tuple of the model state and output.
    """
    if self._cached_run is None:
      self.compile()

    if not kwargs.get('batch_preprocessed', False):
      batch = self.preprocess_batch(batch)
    if self._should_shard_batch:
      batch = self.shard_batch(batch)
    self._run_begin_actions(state, per_loop_step_number)
    state, batch = self.begin(state, batch)

    if log_num_flops and self.num_flops is None:
      self.compute_num_flops(state, batch)
      logging.info(f'Step flops: {self.num_flops}')

    state, outputs = self._cached_run(state, batch, **kwargs)
    state, outputs = self.end(state, outputs)
    self._run_end_actions(state, outputs, per_loop_step_number)

    return state, outputs

  def compute_num_params(self, state: State) -> int:
    """Computes the number of parameters in the model state.

    Args:
      state: The model state.

    Returns:
      The number of parameters in the model state.
    """
    self._num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(state.params)
    )
    return self.num_params

  def compute_num_flops(self, state: State, batch: Batch) -> float:
    """Computes the number of flops in the jitted `run` function.

    Args:
      state: The model state.
      batch: The input data batch.

    Returns:
      The number of flops in the jitted `run` function.
    """
    if self._cached_run is None:
      self.compile()

    assert self._cached_run is not None  # Make Pytype happy.
    analysis = self._cached_run.lower(state, batch).compile().cost_analysis()

    self._num_flops = analysis.get('flops', 0)
    return self.num_flops

  @property
  def train(self) -> bool:
    return self._train

  @property
  def num_params(self) -> int | None:
    return self._num_params

  @property
  def num_flops(self) -> float | None:
    return self._num_flops

  @property
  def model(self) -> nn.Module:
    return self._model

  def _latest_step(
      self,
      checkpoint_dir: epath.PathLike,
  ) -> int | None:
    """Finds the latest step from the given checkpoint directory path."""
    latest_metadata = ocp.path.step.latest_step_metadata(
        checkpoint_dir, ocp.path.step.standard_name_format()
    )
    return latest_metadata.step if latest_metadata is not None else None

  def _restore(
      self,
      step: int,
      checkpoint_dir: epath.PathLike,
      abstract_state: Mapping[str, Any],
      restore_kwargs: Mapping[str, Any],
  ) -> State:
    """Restores the model state using an asynchronous checkpointer."""
    checkpointer = ocp.AsyncCheckpointer(
        ocp.CompositeCheckpointHandler(),
        checkpoint_metadata_store=ocp.metadata.metadata_store(
            enable_write=False  # Disable writing metadata during restore.
        ),
    )
    return checkpointer.restore(
        ocp.path.step.build_step_path(
            checkpoint_dir, ocp.path.step.standard_name_format(), step
        ),
        args=ocp.args.Composite(**{
            self._chkpt_item_name: ocp.args.PyTreeRestore(
                abstract_state, **restore_kwargs
            )  # pytype: disable=wrong-arg-count
        }),
    )[self._chkpt_item_name]

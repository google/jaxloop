"""The step library for JAX models."""

import typing
from typing import Any, List, Mapping, Optional, Protocol, Tuple, Type

from absl import logging
from etils import epath
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jaxloop import partition
from jaxloop import types
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_utils

DType = types.DType
Batch = types.Batch
BatchSpec = types.BatchSpec
Output = types.Output
State = types.TrainState

_DEFAULT_ITEM_NAME = 'default'
_STATE_KEYS = ('step', 'params', 'batch_stats')

# pylint: disable=logging-fstring-interpolation


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
    base_prng: The base prng key.
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
  """

  def __init__(
      self,
      base_prng: jax.Array,
      model: nn.Module | Type[nnx.Module],
      optimizer: optax.GradientTransformation = optax.identity(),
      partitioner: partition.Partitioner = partition.SingleDevicePartitioner(),
      train: bool = False,
      should_shard_batch: bool = True,
      nnx_model_args: Optional[Tuple[Any, ...]] = None,
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

    def is_shape(spec):
      if isinstance(spec, int):
        return True
      return (isinstance(spec, tuple) or isinstance(spec, list)) and all(
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
      if isinstance(spec, tuple) and len(spec) == 2:
        return is_leaf(spec[0])
      return False

    def init_fn(batch):
      variables = self._model.init(self._base_prng, batch, **kwargs)
      return State.create(
          apply_fn=self._model.apply,
          tx=self._optimizer,
          **{k: v for k, v in variables.items() if k in _STATE_KEYS},
      )

    batch = jax.tree_util.tree_map(map_fn, spec, is_leaf=is_leaf)
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
      The prng key for random number generations.
    """
    return jax.random.fold_in(self._base_prng, step)

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

  def shard_batch(self, batch: Batch) -> Batch:
    """Shards the input data batch based on the partitioner."""
    return self._partitioner.shard_batch(batch)

  def __call__(
      self, state: State, batch: Batch, log_num_flops: bool = False, **kwargs
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
      log_num_flops: Whether to log the number of flops of the jitted `run`
        function.
      **kwargs: Additional keyword arguments for running the step.

    Returns:
      A tuple of the model state and output.
    """
    if self._cached_run is None:
      self.compile()

    if self._should_shard_batch:
      batch = self.shard_batch(batch)
    state, batch = self.begin(state, batch)

    if log_num_flops and self.num_flops is None:
      self.compute_num_flops(state, batch)
      logging.info(f'Step flops: {self.num_flops}')

    state, outputs = self._cached_run(state, batch, **kwargs)
    return self.end(state, outputs)

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

  def compute_num_flops(self, state: State, batch: Batch) -> int:
    """Computes the number of flops in the jitted `run` function.

    Args:
      state: The model state.
      batch: The input data batch.

    Returns:
      The number of flops in the jitted `run` function.
    """
    if self._cached_run is None:
      self.compile()
    with jax.spmd_mode('allow_all'):
      analysis = self._cached_run.lower(state, batch).cost_analysis()
    self._num_flops = analysis.get('flops', 0)
    return self.num_flops

  @property
  def train(self) -> bool:
    return self._train

  @property
  def num_params(self) -> int | None:
    return self._num_params

  @property
  def num_flops(self) -> int | None:
    return self._num_flops

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
    pytree_checkpoint_handler = ocp.PyTreeCheckpointHandler()
    checkpointer = ocp.AsyncCheckpointer(
        ocp.CompositeCheckpointHandler(
            **{_DEFAULT_ITEM_NAME: pytree_checkpoint_handler},
        ),
        checkpoint_metadata_store=ocp.metadata.metadata_store(
            enable_write=False  # Disable writing metadata during restore.
        ),
    )
    _, restore_ckpt_arg_cls = checkpoint_args.get_registered_args_cls(
        pytree_checkpoint_handler
    )

    return checkpointer.restore(
        ocp.path.step.build_step_path(
            checkpoint_dir, ocp.path.step.standard_name_format(), step
        ),
        args=ocp.args.Composite(**{
            _DEFAULT_ITEM_NAME: restore_ckpt_arg_cls(
                abstract_state, **restore_kwargs
            )  # pytype: disable=wrong-arg-count
        }),
    )[_DEFAULT_ITEM_NAME]

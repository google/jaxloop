"""A simple trainer for JAX models using Jaxloop, Orbax, and Flax."""

from collections.abc import Iterable
from typing import Optional, Type

from clu import metric_writers
import flax.linen as nn
import jax
from jaxloop import action_loop as action_loop_lib
from jaxloop import actions
from jaxloop import outer_loop
from jaxloop import partition
from jaxloop import step
from jaxloop import train_loop as train_loop_lib
from jaxloop import types
from jaxloop.trainers import simple_step
from jaxloop.trainers import trainer_utils
import optax
from orbax import checkpoint


class SimpleTrainer:
  """A simple trainer for JAX models.

  This class is a simple implementation of a trainer for JAX models. It
  provides a basic framework for training a model with a singular input/output
  that utilizes a simple feed-forward network. Additionally, it provides easy
  access to common functionalities such as checkpointing, logging, and
  evaluation, as well as state management.

  Params:
    model: The Flax model to be trained.
    epochs: The number of epochs to train for.
    steps_per_epoch: The number of steps to run per epoch.
    batch_spec: The batch spec of the model.
    checkpointing_config: The checkpointing config; if None, no checkpointing
      will be performed.
    summary_config: The summary config; if None, no summaries will be written.
    log_num_params: Whether to log the number of parameters during
    initialization.
    optimizer: The optimizer to use.
    partioner: The partitioner to use.
    step_class: The step class to use.
    train_loop_class: The train loop class to use.
    eval_loop_class: The eval loop class to use.
    outer_loop_class: The outer loop class to use.
    base_prng: The base prng to use.
    additional_begin_actions: Additional begin actions to perform.
    additional_end_actions: Additional end actions to perform.
    **kwargs: Keyword arguments to pass to the step and loop constructors.
  """

  def __init__(
      self,
      model: nn.Module,
      epochs: int,
      steps_per_epoch: int,
      batch_spec: types.BatchSpec | trainer_utils.TrainerBatchSpec,
      checkpointing_config: Optional[trainer_utils.CheckpointingConfig] = None,
      summary_config: Optional[trainer_utils.SummaryConfig] = None,
      log_num_params: bool = False,
      optimizer: optax.GradientTransformation = optax.adam(1e-4),
      partitioner: partition.Partitioner = partition.SingleDevicePartitioner(),
      step_class: Type[step.Step] = simple_step.SimpleStep,
      train_loop_class: Type[
          train_loop_lib.TrainLoop
      ] = train_loop_lib.TrainLoop,
      eval_loop_class: Type[
          action_loop_lib.ActionLoop
      ] = action_loop_lib.ActionLoop,
      outer_loop_class: Type[outer_loop.OuterLoop] = outer_loop.OuterLoop,
      base_prng: types.PRNGType | None = None,
      additional_begin_actions: list[actions.Action] = [],
      additional_end_actions: list[actions.Action] = [],
      **kwargs,
  ):
    self._model = model
    self._epochs = epochs
    self._steps_per_epoch = steps_per_epoch
    self._batch_spec = batch_spec
    self._log_num_params = log_num_params
    self._checkpointing_config = checkpointing_config
    self._summary_config = summary_config

    if base_prng is None:
      base_prng = {"params": jax.random.PRNGKey(0)}

    self._train_step = step_class(
        model=model,
        base_prng=base_prng,
        train=True,
        optimizer=optimizer,
        partitioner=partitioner,
        **kwargs,
    )
    begin_actions, end_actions = self._build_actions(
        additional_begin_actions, additional_end_actions
    )
    self._train_loop = train_loop_class(
        step=self._train_step,
        begin_actions=begin_actions,
        end_actions=end_actions,
        **kwargs,
    )

    self._eval_step = step_class(
        base_prng=base_prng,
        model=model,
        train=False,
        **kwargs,
    )
    self._eval_loop = eval_loop_class(
        step=self._eval_step,
        **kwargs,
    )

    self._outer_loop = outer_loop_class(
        self._train_loop,
        checkpoint_spec=self._checkpointing_config.checkpoint_spec
        if self._checkpointing_config is not None
        else None,
        **kwargs,
    )

    self._setup(**kwargs)

  def _build_actions(
      self,
      additional_begin_actions: list[actions.Action],
      additional_end_actions: list[actions.Action],
  ) -> tuple[list[actions.Action], list[actions.Action]]:
    """Builds the actions for the train and eval loops."""
    begin_actions = additional_begin_actions
    end_actions = additional_end_actions
    if self._checkpointing_config is not None:
      ckpt_manager = checkpoint.CheckpointManager(
          self._checkpointing_config.checkpoint_spec.checkpoint_dir
          / "checkpoints",
          checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler()),
          checkpoint.CheckpointManagerOptions(
              max_to_keep=self._checkpointing_config.max_checkpoints
          ),
      )

      end_actions.append(
          actions.CheckpointAction(
              ckpt_manager, self._checkpointing_config.checkpoint_interval
          )
      )

    if self._summary_config is not None:
      train_metrics_writer = metric_writers.create_default_writer(
          self._summary_config.path,
          just_logging=jax.process_index() > 0,
          asynchronous=self._summary_config.asynchronous,
      )

      begin_actions.append(
          actions.SummaryAction(
              train_metrics_writer,
              interval=self._summary_config.interval,
              flush_each_call=self._summary_config.flush_each_call,
          )
      )

    return begin_actions, end_actions

  def _setup(self, **kwargs):
    """Performs internal setup of loops."""
    self._model_state = self._train_step.initialize_model(
        self._batch_spec, self._log_num_params, **kwargs
    )
    self._eval_step.initialize_model(self._batch_spec, False, **kwargs)

  def train(
      self,
      train_dataset: Iterable[types.Batch],
      eval_specs: Optional[list[outer_loop.EvalSpec]] = None,
      **kwargs,
  ) -> Optional[types.Output]:
    """Trains the model on the given dataset."""
    self._model_state, outputs = self._outer_loop(
        self._model_state,
        iter(train_dataset),
        self._epochs * self._steps_per_epoch,
        self._steps_per_epoch,
        eval_specs,
        **kwargs,
    )

    return outputs

  @property
  def model_state(self) -> types.TrainState:
    """Returns the model state."""
    return self._model_state

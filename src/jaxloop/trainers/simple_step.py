"""A simple jaxloop step implementation that performs a simple forward pass with an MSE loss function."""

from typing import Tuple

import clu.metrics as clu_metrics
import jax
from jax import numpy as jnp
from jaxloop import step
from jaxloop import types
import jaxtyping
import optax


class SimpleStep(step.Step):
  """A simple step that trains or evaluates a model on a batch of data.

  This simple implementation is useful for basic usage in the context of simple
  regression or classification models.

  When using this step, it is expected that the dataset provides batches of
  dicts with keys "input_features" and "output_features", representing the input
  and output features. If this is not the case, the `_get_input_features` and
  `_get_output_features` can be overridden to extract the correct features from
  the batch.

  In either case, the predicted output is provided in `pred_y` in the output
  dict.

  Example usage:
  ```
  # Replace with real dataset
  train_ds = {
      "input_features": jnp.ones((batch_size, features)),
      "output_features": jnp.ones((batch_size, 1)),
  }

  step = SimpleStep(
    base_prng=prng,
    model=nn_model,
    optimizer=optax.adam(learning_rate),
    train=True,
  )

  inner_loop = train_loop_lib.TrainLoop(step)
  ctl = outer_loop.OuterLoop(step)

  train_state = train_step.initialize_model(train_ds.shape)

  # Train the model
  trained_state, outputs = ctl(
    train_state,
    train_dataset=_batch(train_ds, batch_size),
    train_total_steps=num_train_steps,
    train_loop_steps=steps_per_epoch,
  )
  ```
  """

  def initialize_model(
      self, spec: types.BatchSpec, log_num_params: bool = False, **kwargs
  ) -> types.TrainState:
    input_spec = self._get_input_features(spec)

    return super().initialize_model(
        input_spec, log_num_params=log_num_params, **kwargs
    )

  def run(
      self, state: step.State, batch: step.Batch
  ) -> Tuple[step.State, step.Output]:
    """Train on a batch of data."""
    input_features = self._get_input_features(batch)
    output_features = self._get_output_features(batch)

    # Apply gradients if training, otherwise just compute loss.
    state, loss, output_features_pred = self._predict_and_compute_loss(
        state, input_features, output_features
    )

    output = {
        "loss": clu_metrics.Average.from_model_output(loss),
        "output_features_pred": output_features_pred,
    }

    return state, output

  def _get_input_features(
      self, batch: step.Batch | types.BatchSpec
  ) -> jaxtyping.PyTree:
    """Extracts the input features from the batch.

    By default, it is assumed that all batches have input_features in the key
    "input_features". If this is not the case, this method can be overridden to
    extract the correct features from the batch.

    Args:
      batch: One batch of data from the dataset.

    Returns:
      The input features of the batch.
    """
    return batch["input_features"]

  def _get_output_features(self, batch: step.Batch) -> jaxtyping.PyTree:
    """Extracts the output features from the batch.

    By default, it is assumed that all batches have output_features in the key
    "output_features". If this is not the case, this method can be overridden to
    extract the correct features from the batch.

    Output features are used to compute the loss between the predicted output
    and the true output. They can signify labels (in classification tasks) or
    the target value (in regression tasks).

    Args:
      batch: One batch of data from the dataset.

    Returns:
      The output features of the batch.
    """
    return batch["output_features"]

  def _predict_and_compute_loss(
      self,
      state: types.TrainState,
      input_features: jax.Array,
      output_features: jax.Array,
  ):
    """Apply the model and compute MSE loss, applying gradients if training.

    By default, performs a forward pass of the model and applies the loss
    function specified in self._loss_fn. Based on the value of self._train,
    the gradients are applied to the model in training contexts.

    Args:
      state: The model state.
      input_features: A batch of input features to the model.
      output_features: A batch of output features to the model (ground-truth
        labels or target values).

    Returns:
      The model state (updated if training), the loss, and the predicted output
      features.
    """

    def prediction_and_loss_computation_fn(
        params: jaxtyping.PyTree,
    ) -> Tuple[jax.Array, Tuple[jax.Array, types.TrainState]]:
      output_features_pred = state.apply_fn(
          {"params": params},
          input_features,
          **{"train": self._train},
      )

      loss = self.loss_fn(output_features_pred, output_features)

      return loss, output_features_pred

    if self._train:
      gradient_fn = jax.value_and_grad(
          prediction_and_loss_computation_fn, has_aux=True
      )
      (loss, output_features_pred), gradients = gradient_fn(state.params)
      state = state.apply_gradients(grads=gradients)
    else:
      loss, output_features_pred = prediction_and_loss_computation_fn(
          state.params
      )
      state = state.replace(step=state.step + 1)
    return state, loss, output_features_pred

  def loss_fn(
      self,
      output_features_pred: jax.Array,
      true_output_features: jax.Array,
  ) -> jax.Array:
    """Computes the loss between the predicted and actual output features.

    By default, applies MSE loss, but can be overridden to use other loss
    functions.

    Args:
      output_features_pred: The predicted output features.
      true_output_features: The true output features.

    Returns:
      The loss between the predicted and actual output features.
    """
    return jnp.mean(optax.l2_loss(output_features_pred, true_output_features))

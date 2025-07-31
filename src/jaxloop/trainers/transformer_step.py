"""A jaxloop step implementation for training or evaluating sequence models."""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
from jaxloop import types
from jaxloop.trainers import simple_step
import jaxtyping
import optax


# Assuming jaxloop.types.Batch will support nested dictionaries (PyTrees)

Array = jaxtyping.Array
PyTree = jaxtyping.PyTree


class TransformerStep(simple_step.SimpleStep):
  """A step for training or evaluating Transformer models.

  This step expects batches with a nested structure:
  {
      "input_features": {
          "input_ids": <int>[batch, seq_len],
          "attention_mask": <int>[batch, seq_len],
      },
      "output_features": {
          "labels": <int>[batch, seq_len],
      },
  }

  Where input_ids are token ids for the model input, attention_mask is a mask
  for non-padded tokens (1=unmasked, 0=masked), and labels are the target token
  ids for loss calculations.

  The model provided to this step should expect the 'input_features' dictionary
  as input to its apply method.

  This class overrides the loss function to use a masked cross-entropy loss,
  suitable for sequence modeling tasks like language modeling.

  The run() method is inherited from SimpleStep and will call the
  overridden methods. The output dictionary it returns will be:
  {
      "loss": clu_metrics.Average.from_model_output(loss),
      "output_features_pred": logits,
  }
  """

  def _get_input_features(self, batch: types.Batch | types.BatchSpec) -> PyTree:
    """Extracts the input features dictionary from the batch.

    This function is overridden to remove the isinstance check because
    a batch passed to TransformerStep is always expected to be a dictionary.

    Args:
      batch: A PyTree representing the batch, expected to have a key
        "input_features".

    Returns:
      The value associated with batch["input_features"].
    """
    return batch["input_features"]

  def _validate_input_features(self, input_features: Any) -> None:
    """Checks for required keys in input_features for TransformerStep.

    Args:
      input_features: A dictionary containing `input_ids` and `attention_mask`.
        { "input_ids": <int>[batch, seq_len], "attention_mask": <int>[batch,
        seq_len], }

    Raises:
      ValueError: If input_features is not a dictionary or is missing required
        keys.
    """
    if not isinstance(input_features, dict):
      raise ValueError("input_features must be a dictionary.")
    required_keys = {"input_ids", "attention_mask"}
    if not required_keys.issubset(input_features.keys()):
      missing = required_keys - input_features.keys()
      raise ValueError(
          f"input_features is missing required keys: {missing}. "
          "TransformerStep requires 'input_ids' and 'attention_mask'."
      )

  def _validate_output_features(self, output_features: Any) -> None:
    """Checks for required keys in output_features for TransformerStep.

    Args:
      output_features: A dictionary containing `labels`. { "labels":
        <int>[batch, seq_len], }

    Raises:
      ValueError: If output_features is not a dictionary or is missing required
        keys.
    """
    if not isinstance(output_features, dict):
      raise ValueError("output_features must be a dictionary.")
    if "labels" not in output_features:
      raise ValueError(
          "output_features is missing required key: {'labels'}. "
          "TransformerStep requires 'labels'."
      )

  def loss_fn(
      self, logits: Array, labels: Array, attention_mask: Array
  ) -> Array:
    """Computes masked softmax cross-entropy loss.

    Args:
        logits: Predicted logits from the model of shape <float>[batch, seq_len,
          vocab_size].
        labels: True labels (token ids) of shape <int>[batch, seq_len].
        attention_mask: Mask indicating non-padded tokens (1 for non-padded, 0
          for padded) of shape <int>[batch, seq_len].

    Returns:
        The mean loss per token over the non-padded tokens.
    """
    vocab_size = logits.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, num_classes=vocab_size)

    token_losses = optax.softmax_cross_entropy(
        logits=logits, labels=labels_onehot
    )

    masked_losses = token_losses * attention_mask

    total_loss = jnp.sum(masked_losses)
    num_active_tokens = jnp.sum(attention_mask)

    # Avoid division by zero if all tokens are masked
    mean_loss = total_loss / jnp.maximum(num_active_tokens, 1e-8)
    return mean_loss

  def _predict_and_compute_loss(
      self,
      state: types.TrainState,
      input_features: Dict[str, Array],
      output_features: Dict[str, Array],
  ) -> Tuple[types.TrainState, Array, Array]:
    """Applies the model and computes the masked cross-entropy loss.

    Args:
      state: The current model state.
      input_features: A dictionary containing `input_ids` and `attention_mask`.
        { "input_ids": <int>[batch, seq_len], "attention_mask": <int>[batch,
        seq_len], }
      output_features: A dictionary containing `labels`. { "labels":
        <int>[batch, seq_len], }

    Returns:
      A tuple of (updated state, loss, logits).

    Raises:
      ValueError: If input_features or output_features are incorrectly
        structured.
    """
    self._validate_input_features(input_features)
    self._validate_output_features(output_features)

    def prediction_and_loss_computation_fn(
        params: PyTree,
    ) -> Tuple[Array, Array]:  # Returns loss and logits
      # The model's apply function must be able to handle the dict of features.
      logits = state.apply_fn(
          {"params": params},
          input_features,  # Pass the whole dict
          train=self._train,
          rngs=self.prng_key(state.step),  # Use step-dependent PRNG key
      )

      labels = output_features["labels"]
      attention_mask = input_features["attention_mask"]

      loss = self.loss_fn(logits, labels, attention_mask)

      return loss, logits

    if self._train:
      grad_fn = jax.value_and_grad(
          prediction_and_loss_computation_fn, has_aux=True
      )
      (loss, logits), gradients = grad_fn(state.params)
      state = state.apply_gradients(grads=gradients)
    else:
      loss, logits = prediction_and_loss_computation_fn(state.params)
      state = state.replace(step=state.step + 1)

    return state, loss, logits

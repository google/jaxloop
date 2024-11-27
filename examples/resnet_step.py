"""Step function for ResNet."""

from typing import Optional, Tuple

from clu import metrics as clu_metrics
import jax
import jax.numpy as jnp
from jaxloop import step as step_lib
from jaxloop import types
import jaxtyping
import optax


def _resnet_training_loss_and_logits(
    state: types.TrainState,
    images: jax.Array,
    labels: jax.Array,
    rng_key: jaxtyping.PRNGKeyArray,
):
  """Apply resnet training, get loss and logits."""

  def _loss_fn(
      params: jaxtyping.PyTree,
  ) -> Tuple[jax.Array, Tuple[jax.Array, types.TrainState]]:
    # For training, add mutable batch_stats for batch norm.
    # https://flax.readthedocs.io/en/latest/guides/training_techniques/batch_norm.html
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        images,
        train=True,
        mutable=['batch_stats'],
        rngs={'dropout': rng_key},
    )

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss), (logits, new_model_state)

  gradient_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, (logits, new_model_state)), gradients = gradient_fn(state.params)
  state = state.apply_gradients(
      grads=gradients, batch_stats=new_model_state['batch_stats']
  )
  return state, loss, logits


def _resnet_eval_loss_and_logits(
    state: types.TrainState,
    images: jax.Array,
    labels: jax.Array,
):
  """Apply resnet eval, get loss and logits."""
  logits = state.apply_fn(
      {'params': state.params, 'batch_stats': state.batch_stats},
      images,
      train=False,
  )
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  return state, loss, logits


class ResnetStep(step_lib.Step):
  """A single step for Resnet."""

  def run(
      self, state: types.TrainState, batch: types.Batch
  ) -> tuple[types.TrainState, Optional[types.Output]]:
    """A single step for Resnet."""
    images, labels = batch['image'], batch['label']

    # Get loss and logits.
    if self._train:
      state, loss, logits = _resnet_training_loss_and_logits(
          state, images, labels, self.prng_key(state.step)
      )
    else:
      state, loss, logits = _resnet_eval_loss_and_logits(state, images, labels)

    loss = clu_metrics.Average.from_model_output(values=loss)
    accuracy = clu_metrics.Accuracy.from_model_output(
        logits=logits, labels=labels
    )

    return state, {'loss': loss, 'accuracy': accuracy}

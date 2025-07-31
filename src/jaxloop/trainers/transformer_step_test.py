"""Unit tests for the TransformerStep class."""

from typing import Dict

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import types
from jaxloop.trainers import transformer_step
import jaxtyping
import optax

Array = jaxtyping.Array
PyTree = jaxtyping.PyTree


class MockTransformerModel(nn.Module):
  """A minimal mock model that outputs logits."""

  vocab_size: int
  embed_size: int = 16

  @nn.compact
  def __call__(self, input_features: Dict[str, Array], train: bool = True):
    """Mock call, expects a dictionary.

    Args:
      input_features: Should contain 'input_ids'.
      train: True if training, False if eval.

    Returns:
      Logits of shape [batch, seq_len, vocab_size].
    """
    input_ids = input_features['input_ids']
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)(
        input_ids
    )
    logits = nn.Dense(features=self.vocab_size)(x)
    return logits


class TransformerStepTest(parameterized.TestCase):
  """TransformerStep unit tests."""

  def setUp(self):
    super().setUp()
    self.vocab_size = 10
    self.model = MockTransformerModel(vocab_size=self.vocab_size)
    self.prng = jax.random.PRNGKey(0)
    self.batch_size = 2
    self.seq_len = 4

    self.batch_spec: types.BatchSpec = {
        'input_features': {
            'input_ids': ((self.batch_size, self.seq_len), jnp.int32),
            'attention_mask': ((self.batch_size, self.seq_len), jnp.int32),
        },
        'output_features': {
            'labels': ((self.batch_size, self.seq_len), jnp.int32),
        },
    }

  def _get_dummy_batch(self) -> types.Batch:
    """Creates a dummy batch matching the transformer structure."""
    input_features = self._get_dummy_input_features()
    output_features = self._get_dummy_output_features()
    return {
        'input_features': input_features,
        'output_features': output_features,
    }

  def _get_dummy_input_features(self):
    return {
        'input_ids': jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=jnp.int32),
        'attention_mask': jnp.array(
            [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32
        ),
    }

  def _get_dummy_output_features(self):
    return {
        'labels': jnp.array([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=jnp.int32),
    }

  def _get_dummy_logits(self):
    return jax.random.uniform(
        self.prng, (self.batch_size, self.seq_len, self.vocab_size)
    )

  def test_instantiation(self):
    """Tests if the TransformerStep can be instantiated."""
    step = transformer_step.TransformerStep(
        base_prng=self.prng,
        model=self.model,
        optimizer=optax.adam(1e-4),
        train=True,
    )
    self.assertIsInstance(step, transformer_step.TransformerStep)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_masking',
          mask_list=[[1, 1, 1, 1], [1, 1, 1, 1]],
          expected_loss_greater_than=0.0,
      ),
      dict(
          testcase_name='some_masking',
          mask_list=[[1, 1, 0, 0], [1, 0, 0, 0]],
          expected_loss_greater_than=0.0,
      ),
      dict(
          testcase_name='all_masked',
          mask_list=[[0, 0, 0, 0], [0, 0, 0, 0]],
          expected_loss_greater_than=-1e-7,
      ),
  )
  def test_loss_fn(self, mask_list, expected_loss_greater_than):
    """Tests the loss_fn calculation directly."""
    step = transformer_step.TransformerStep(
        base_prng=self.prng, model=self.model, train=True
    )
    labels = self._get_dummy_output_features()['labels']
    dummy_logits = self._get_dummy_logits()
    attention_mask = jnp.array(mask_list, dtype=jnp.int32)

    loss = step.loss_fn(dummy_logits, labels, attention_mask)

    self.assertIsInstance(loss, Array)
    self.assertGreater(loss.item(), expected_loss_greater_than)

    if jnp.all(jnp.array(mask_list) == 0):
      self.assertAlmostEqual(loss.item(), 0.0, places=6)

  def test_loss_fn_masking_effect(self):
    """Ensures masked positions don't contribute to the loss."""
    step = transformer_step.TransformerStep(
        base_prng=self.prng, model=self.model, train=True
    )
    labels = self._get_dummy_output_features()['labels']
    dummy_logits = self._get_dummy_logits()

    mask1 = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.int32)
    mask2 = jnp.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)

    loss1 = step.loss_fn(dummy_logits, labels, mask1)

    extreme_val = 1e6
    key = jax.random.PRNGKey(42)
    modified_logits = dummy_logits.at[:, 2:, :].set(
        jax.random.uniform(key, (self.batch_size, 2, self.vocab_size))
        * extreme_val
    )

    loss2_orig = step.loss_fn(dummy_logits, labels, mask2)
    loss2_modified = step.loss_fn(modified_logits, labels, mask2)

    # The loss should be the same because the changes are in masked positions
    self.assertAlmostEqual(loss2_orig.item(), loss2_modified.item(), places=5)
    # Loss with less masking should be different
    self.assertNotAlmostEqual(loss1.item(), loss2_orig.item(), places=5)

  @parameterized.named_parameters(
      dict(testcase_name='train', training_mode=True),
      dict(testcase_name='eval', training_mode=False),
  )
  def test_predict_and_compute_loss(self, training_mode):
    """Tests the _predict_and_compute_loss method with mock inputs."""
    step = transformer_step.TransformerStep(
        base_prng={'params': self.prng, 'dropout': jax.random.PRNGKey(1)},
        model=self.model,
        optimizer=optax.adam(1e-4),
        train=training_mode,
    )

    input_features = self._get_dummy_input_features()
    output_features = self._get_dummy_output_features()

    # Initialize state
    params = self.model.init(self.prng, input_features)['params']
    state = types.TrainState.create(
        apply_fn=self.model.apply,
        params=params,
        tx=optax.adam(1e-4),
    )
    initial_step = state.step

    new_state, loss, logits = step._predict_and_compute_loss(
        state, input_features, output_features
    )

    self.assertIsInstance(new_state, types.TrainState)
    self.assertIsInstance(loss, Array)
    self.assertIsInstance(logits, Array)

    self.assertEqual(
        logits.shape, (self.batch_size, self.seq_len, self.vocab_size)
    )
    self.assertGreater(loss.item(), 0.0)
    self.assertEqual(new_state.step, initial_step + 1)

    if training_mode:
      param_changed = any(
          not jnp.array_equal(p1, p2)
          for p1, p2 in zip(
              jax.tree_util.tree_leaves(state.params),
              jax.tree_util.tree_leaves(new_state.params),
          )
      )
      self.assertTrue(param_changed)
    else:
      # Params should NOT change in eval mode
      param_equal = all(
          jnp.array_equal(p1, p2)
          for p1, p2 in zip(
              jax.tree_util.tree_leaves(state.params),
              jax.tree_util.tree_leaves(new_state.params),
          )
      )
      self.assertTrue(param_equal)

  def test_initialize_model(self):
    """Tests the initialize_model method."""
    step = transformer_step.TransformerStep(
        base_prng={'params': self.prng},
        model=self.model,
        optimizer=optax.adam(1e-4),
        train=True,
    )
    state = step.initialize_model(self.batch_spec)
    self.assertIsInstance(state, types.TrainState)
    self.assertTrue(hasattr(state, 'params'))
    self.assertTrue(hasattr(state, 'step'))

  def test_loss_decreases_over_steps(self):
    """Tests that the training loss decreases over several steps."""
    step = transformer_step.TransformerStep(
        base_prng={'params': self.prng, 'dropout': jax.random.PRNGKey(1)},
        model=self.model,
        optimizer=optax.adam(learning_rate=1e-2),
        train=True,
    )

    state = step.initialize_model(self.batch_spec)

    batch = self._get_dummy_batch()

    num_train_steps = 10
    losses = []

    for i in range(num_train_steps):
      state, output = step(state, batch, per_loop_step_number=i)
      self.assertIsNotNone(output)
      loss = output['loss'].compute()
      losses.append(loss)

    self.assertLess(
        losses[-1],
        losses[0],
        'Loss should decrease after several training steps.',
    )
    self.assertTrue(
        any(losses[i] != losses[0] for i in range(1, num_train_steps)),
        'Losses should change over steps.',
    )


if __name__ == '__main__':
  absltest.main()

"""Tests for jaxloop.metrics.metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jaxloop.metrics import metrics
import numpy as np


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_outputs = (
        dict(
            logits=jnp.array([0.4, 0.4]),
            labels=jnp.array([0, 1]),
        ),
        dict(
            logits=jnp.array([0.6, 0.4]),
            labels=jnp.array([1, 1]),
        ),
        dict(
            logits=jnp.array([1.0, 0.7]),
            labels=jnp.array([0, 1]),
        ),
        dict(
            logits=jnp.array([0.0, 0.0]),
            labels=jnp.array([1, 1]),
        ),
    )
    self.model_outputs_batch_size_one = (
        dict(
            logits=jnp.array([[0.3]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.7]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.8]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.2]]),
            labels=jnp.array([1]),
        ),
    )

  def compute_precision(self, model_outputs, threshold: float = 0.5):
    metric = None
    for model_output in model_outputs:
      update = metrics.Precision.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_recall(self, model_outputs, threshold: float = 0.5):
    metric = None
    for model_output in model_outputs:
      update = metrics.Recall.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def test_precision(self):
    """Test that new Precision Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_precision(self.model_outputs),
        [2 / 3],
    )

  def test_precision_at_0_7(self):
    """Test that new Precision Metric computes correct values at 0.7 threshold."""
    np.testing.assert_allclose(
        self.compute_precision(self.model_outputs, threshold=0.7),
        [1 / 2],
    )

  def test_precision_with_batch_size_one(self):
    """Test that new Precision Metric is correct with batch size one."""
    np.testing.assert_allclose(
        self.compute_precision(self.model_outputs_batch_size_one),
        [1.0],
    )

  def test_recall(self):
    """Test that new Recall Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_recall(self.model_outputs),
        [1 / 3],
    )

  def test_recall_at_0_7(self):
    """Test that new Recall Metric computes correct values at 0.7 threshold."""
    np.testing.assert_allclose(
        self.compute_recall(self.model_outputs, threshold=0.7),
        [1 / 6],
    )

  def test_recall_with_batch_size_one(self):
    """Test that new Recall Metric is correct with batch size one."""
    np.testing.assert_allclose(
        self.compute_recall(self.model_outputs_batch_size_one),
        [0.5],
    )


if __name__ == '__main__':
  absltest.main()

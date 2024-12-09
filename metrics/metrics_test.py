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

  def compute_aucpr(self, model_outputs):
    metric = None
    for model_output in model_outputs:
      update = metrics.AUCPR.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_aucroc(self, model_outputs):
    metric = None
    for model_output in model_outputs:
      update = metrics.AUCROC.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
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

  def test_aucpr(self):
    """Test that new AUC-PR Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs),
        jnp.array(0.5972222, dtype=jnp.float32),
    )

  def test_aucpr_with_batch_size_one(self):
    """Test that new AUC-PR Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs_batch_size_one),
        jnp.array(0.875, dtype=jnp.float32),
    )

  def test_aucroc(self):
    """Test that new AUC-ROC Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_aucroc(self.model_outputs),
        jnp.array(0.046875, dtype=jnp.float32),
    )

  def test_aucroc_with_batch_size_one(self):
    """Test that new AUC-ROC Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_aucroc(self.model_outputs_batch_size_one),
        [0],
    )


if __name__ == '__main__':
  absltest.main()

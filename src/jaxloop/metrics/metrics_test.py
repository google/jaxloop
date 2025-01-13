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
from sklearn import metrics as sklearn_metrics

np.random.seed(42)
OUTPUT_LABELS = np.random.randint(0, 2, size=(4, 10))
OUTPUT_PREDS = np.random.uniform(size=(4, 10))
OUTPUT_LABELS_BS1 = np.random.randint(0, 2, size=(4, 1))
OUTPUT_PREDS_BS1 = np.random.uniform(size=(4, 1))


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # TODO(jeffcarp): Merge these into generated fixtures.
    self.model_outputs = (
        dict(
            logits=jnp.array(
                [0.34, 0.89, 0.12, 0.67, 0.98, 0.23, 0.56, 0.71, 0.45, 0.08]
            ),
            labels=jnp.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1]),
        ),
        dict(
            logits=jnp.array(
                [0.23, 0.89, 0.57, 0.11, 0.99, 0.38, 0.76, 0.05, 0.62, 0.44]
            ),
            labels=jnp.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0]),
        ),
        dict(
            logits=jnp.array(
                [0.67, 0.21, 0.95, 0.03, 0.88, 0.51, 0.34, 0.79, 0.15, 0.42]
            ),
            labels=jnp.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1]),
        ),
        dict(
            logits=jnp.array(
                [0.91, 0.37, 0.18, 0.75, 0.59, 0.02, 0.83, 0.26, 0.64, 0.48]
            ),
            labels=jnp.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
        ),
    )
    self.model_outputs_batch_size_one = (
        dict(
            logits=jnp.array([[0.32]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.74]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.86]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.21]]),
            labels=jnp.array([1]),
        ),
    )
    self.sample_weights = jnp.array([0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0])

  def compute_aucpr(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrics.AUCPR.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_aucroc(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrics.AUCROC.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_mse(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrics.MSE.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_rmse(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrics.RMSE.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_rsquared(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrics.RSQUARED.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, 0.5),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.7),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_precision(self, y_true, y_pred, threshold):
    """Test that Precision metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    expected = sklearn_metrics.precision_score(y_true, y_pred)

    metric = None
    for logits, labels in zip(y_pred, y_true):
      update = metrics.Precision.from_model_output(
          predictions=logits,
          labels=labels,
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)

    np.testing.assert_allclose(
        metric.compute(),
        expected,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, 0.5),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.7),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_recall(self, y_true, y_pred, threshold):
    """Test that Recall metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    expected = sklearn_metrics.recall_score(y_true, y_pred)

    metric = None
    for logits, labels in zip(y_pred, y_true):
      update = metrics.Recall.from_model_output(
          predictions=logits,
          labels=labels,
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)

    np.testing.assert_allclose(
        metric.compute(),
        expected,
    )

  def test_aucpr(self):
    """Test that AUC-PR Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs),
        jnp.array(0.39081815, dtype=jnp.float32),
    )

  def test_aucpr_with_sample_weight(self):
    """Test that AUC-PR Metric computes correct values when using sample weights."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs, self.sample_weights),
        jnp.array(0.3224537, dtype=jnp.float32),
    )

  def test_aucpr_with_batch_size_one(self):
    """Test that AUC-PR Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs_batch_size_one),
        jnp.array(0.875, dtype=jnp.float32),
    )

  def test_aucroc(self):
    """Test that AUC-ROC Metric computes correct values."""
    # Concatenate logits and labels
    all_logits = jnp.concatenate(
        [model_output['logits'] for model_output in self.model_outputs]
    )
    all_labels = jnp.concatenate(
        [model_output['labels'] for model_output in self.model_outputs]
    )
    np.testing.assert_allclose(
        self.compute_aucroc(self.model_outputs),
        sklearn_metrics.roc_auc_score(all_labels, all_logits),
    )

  def test_aucroc_with_sample_weight(self):
    """Test that AUC-ROC Metric computes correct values when using sample weights."""
    # Concatenate logits and labels
    all_logits = jnp.concatenate(
        [model_output['logits'] for model_output in self.model_outputs]
    )
    all_labels = jnp.concatenate(
        [model_output['labels'] for model_output in self.model_outputs]
    )
    sample_weights = jnp.concatenate(
        [self.sample_weights] * len(self.model_outputs)
    )
    np.testing.assert_allclose(
        jnp.array(
            self.compute_aucroc(self.model_outputs, self.sample_weights),
            dtype=jnp.float16,
        ),
        jnp.array(
            sklearn_metrics.roc_auc_score(
                all_labels, all_logits, sample_weight=sample_weights
            ),
            dtype=jnp.float16,
        ),
    )

  def test_mse(self):
    """Test that MSE Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_mse(self.model_outputs),
        jnp.array(0.47074753, dtype=jnp.float32),
    )

  def test_mse_with_sample_weight(self):
    """Test that MSE Metric computes correct values when using sample weights."""
    np.testing.assert_allclose(
        self.compute_mse(self.model_outputs, self.sample_weights),
        jnp.array(0.5529917, dtype=jnp.float32),
    )

  def test_mse_with_batch_size_one(self):
    """Test that MSE Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_mse(self.model_outputs_batch_size_one),
        jnp.array(0.29342502, dtype=jnp.float32),
    )

  def test_rmse(self):
    """Test that RMSE Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_rmse(self.model_outputs),
        jnp.array(0.68611044, dtype=jnp.float32),
    )

  def test_rmse_with_sample_weight(self):
    """Test that RMSE Metric computes correct values when using sample weights."""
    np.testing.assert_allclose(
        self.compute_rmse(self.model_outputs, self.sample_weights),
        jnp.array(0.7436341, dtype=jnp.float32),
    )

  def test_rmse_with_batch_size_one(self):
    """Test that RMSE Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_rmse(self.model_outputs_batch_size_one),
        jnp.array(0.5416872, dtype=jnp.float32),
    )

  def test_rsquared(self):
    """Test that RSQUARED Metric computes correct values.

    Correct values were calculated using the sklearn library.
    """
    np.testing.assert_allclose(
        self.compute_rsquared(self.model_outputs).astype(jnp.float16),
        jnp.array(-0.887709, dtype=jnp.float16),
    )

  def test_rsquared_with_sample_weight(self):
    """Test that RSQUARED Metric computes correct values when using sample weights.

    Correct values were calculated using the sklearn library.
    """
    np.testing.assert_allclose(
        self.compute_rsquared(self.model_outputs, self.sample_weights).astype(
            jnp.float16
        ),
        jnp.array(-1.2119668, dtype=jnp.float16),
    )

  def test_rsquared_with_batch_size_one(self):
    """Test that RSQUARED Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_rsquared(self.model_outputs_batch_size_one),
        [1.0],
    )


if __name__ == '__main__':
  absltest.main()

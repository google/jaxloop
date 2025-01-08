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

"""A collection of different CLU metrics for the training library."""

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp


def _default_threshold() -> jax.Array:
  """This is the default threshold used for binary classification.

  Starts at 1.0 and goes down to 0.0 by an interval of 1/199.
  """
  return jnp.array(
      [1.0 + 1e-7] + [(198 - i) / (199) for i in range(198)] + [0.0 - 1e-7]
  )


def _divide_no_nan(x: jax.Array, y: jax.Array) -> jax.Array:
  """Computes a safe divide which returns 0 if the y is zero."""
  return jnp.where(y != 0, jnp.divide(x, y), 0.0)


@flax.struct.dataclass
class MSE(clu_metrics.Average):
  """Computes the mean squared error for regression problems given `predictions` and `labels`."""

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'MSE':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` representing the prediction
        generated from the model. The shape should be [batch_size, 1].
      labels: True value. The shape should be [batch_size, 1].
      sample_weights: An optional floating point `Tensor` representing the
        weight of each sample. The shape should be [batch_size, 1].

    Returns:
      Updated MSE metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    squared_error = jnp.square(predictions - labels)
    count = jnp.ones_like(labels, dtype=jnp.int32)
    if sample_weights is not None:
      squared_error *= sample_weights
      count *= sample_weights
    return cls(
        total=squared_error.sum(),
        count=count.sum(),
    )


@flax.struct.dataclass
class RMSE(MSE):
  """Computes the root mean squared error for regression problems given `predictions` and `labels`."""

  def compute(self) -> jax.Array:
    return jnp.sqrt(super().compute())


@flax.struct.dataclass
class RSQUARED(clu_metrics.Metric):
  """Computes the r-squared score of a scalar or a batch of tensors.

  R-squared is a measure of how well the regression model fits the data. It
  measures the proportion of the variance in the dependent variable that is
  explained by the independent variable(s). It is defined as 1 - SSE / SST,
  where SSE is the sum of squared errors and SST is the total sum of squares.
  """

  total: jax.Array
  count: jax.Array
  sum_of_squared_error: jax.Array
  sum_of_squared_label: jax.Array

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'RSQUARED':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` representing the prediction
        generated from the model. The shape should be [batch_size, 1].
      labels: True value. The shape should be [batch_size, 1].
      sample_weights: An optional floating point `Tensor` representing the
        weight of each sample. The shape should be [batch_size, 1].

    Returns:
      Updated RSQUARED metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    count = jnp.ones_like(labels, dtype=jnp.int32)
    squared_error = jnp.power(labels - predictions, 2)
    squared_label = jnp.power(labels, 2)
    if sample_weights is not None:
      labels *= sample_weights
      count *= sample_weights
      squared_error *= sample_weights
      squared_label *= sample_weights
    return cls(
        total=labels.sum(),
        count=count.sum(),
        sum_of_squared_error=squared_error.sum(),
        sum_of_squared_label=squared_label.sum(),
    )

  def merge(self, other: 'RSQUARED') -> 'RSQUARED':
    return type(self)(
        total=self.total + other.total,
        sum_of_squared_error=self.sum_of_squared_error
        + other.sum_of_squared_error,
        sum_of_squared_label=self.sum_of_squared_label
        + other.sum_of_squared_label,
        count=self.count + other.count,
    )

  def compute(self) -> jax.Array:
    """Computes the r-squared score.

    Since we don't know the mean of the labels before we aggregate all of the
    data, we will manipulate the formula to be:
    sst = \sum_i (x_i - mean)^2
        = \sum_i (x_i^2 - 2 x_i mean + mean^2)
        = \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2
        = \sum_i x_i^2 - 2 mean * N * mean + N * mean^2
        = \sum_i x_i^2 - N * mean^2

    Returns:
      The r-squared score.
    """
    mean = self.total / self.count
    sst = self.sum_of_squared_label - self.count * jnp.power(mean, 2)
    return 1 - _divide_no_nan(self.sum_of_squared_error, sst)


@flax.struct.dataclass
class Precision(clu_metrics.Metric):
  """Computes precision for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_positives: The count of false positive instances from the given data,
      label, and threshold.
  """

  true_positives: jax.Array
  false_positives: jax.Array

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      threshold: float = 0.5,
  ) -> 'Precision':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. The shape should be [batch_size, 1].
      labels: True value. The value is expected to be 0 or 1. The shape should
        be [batch_size, 1].
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Precision metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = jnp.where(predictions >= threshold, 1, 0)
    true_positives = jnp.sum((predictions == 1) & (labels == 1))
    false_positives = jnp.sum((predictions == 1) & (labels == 0))

    return cls(true_positives=true_positives, false_positives=false_positives)

  def merge(self, other: 'Precision') -> 'Precision':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
    )

  def compute(self) -> jax.Array:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_positives)
    )


@flax.struct.dataclass
class Recall(clu_metrics.Metric):
  """Computes recall for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_negatives: The count of false negative instances from the given data,
      label, and threshold.
  """

  true_positives: jax.Array
  false_negatives: jax.Array

  @classmethod
  def from_model_output(
      cls, predictions: jax.Array, labels: jax.Array, threshold: float = 0.5
  ) -> 'Recall':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. The shape should be [batch_size, 1].
      labels: True value. The value is expected to be 0 or 1. The shape should
        be [batch_size, 1].
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Recall metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = jnp.where(predictions >= threshold, 1, 0)
    true_positives = jnp.sum((predictions == 1) & (labels == 1))
    false_negatives = jnp.sum((predictions == 0) & (labels == 1))

    return cls(true_positives=true_positives, false_negatives=false_negatives)

  def merge(self, other: 'Recall') -> 'Recall':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )

  def compute(self) -> jax.Array:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_negatives)
    )


@flax.struct.dataclass
class AUCPR(clu_metrics.Metric):
  """Computes area under the precision-recall curve for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    false_negatives: The count of false negative instances from the given data
      and label at each threshold.
  """

  # shape: (threshold, 1)
  true_positives: jax.Array
  false_positives: jax.Array
  false_negatives: jax.Array

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'AUCPR':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. The shape should be [batch_size, 1].
      labels: True value. The value is expected to be 0 or 1. The shape should
        be [batch_size, 1].
      sample_weights: An optional floating point `Tensor` representing the
        weight of each sample. The shape should be [batch_size, 1].

    Returns:
      The area under the precision-recall curve. The shape should be a single
      scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(predictions, _default_threshold()[..., None])
    pred_is_neg = jnp.logical_not(pred_is_pos)
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    false_positives = pred_is_pos * label_is_neg
    false_negatives = pred_is_neg * label_is_pos

    if sample_weights is not None:
      true_positives *= sample_weights
      false_positives *= sample_weights
      false_negatives *= sample_weights

    return cls(
        true_positives=true_positives.sum(axis=-1),
        false_positives=false_positives.sum(axis=-1),
        false_negatives=false_negatives.sum(axis=-1),
    )

  def merge(self, other: 'AUCPR') -> 'AUCPR':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )

  def compute(self) -> jax.Array:
    precision = _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_positives)
    )
    recall = _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_negatives)
    )
    return jnp.trapezoid(precision, recall)


@flax.struct.dataclass
class AUCROC(clu_metrics.Metric):
  """Computes area under the receiver operation characteristic curve for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    total_count: The count of every data point.
  """

  # shape: (threshold, 1)
  true_positives: jax.Array
  false_positives: jax.Array
  # shape: (1)
  total_count: jax.Array

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'AUCROC':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. The shape should be [batch_size, 1].
      labels: True value. The value is expected to be 0 or 1. The shape should
        be [batch_size, 1].
      sample_weights: An optional floating point `Tensor` representing the
        weight of each sample. The shape should be [batch_size, 1].

    Returns:
      The area under the receiver operation characteristic curve. The shape
      should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(predictions, _default_threshold()[..., None])
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    false_positives = pred_is_pos * label_is_neg
    total = jnp.ones_like(labels)

    if sample_weights is not None:
      true_positives *= sample_weights
      false_positives *= sample_weights
      total *= sample_weights

    return cls(
        true_positives=true_positives.sum(axis=-1),
        false_positives=false_positives.sum(axis=-1),
        total_count=total.sum(axis=-1),
    )

  def merge(self, other: 'AUCROC') -> 'AUCROC':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
        total_count=self.total_count + other.total_count,
    )

  def compute(self) -> jax.Array:
    tp_rate = _divide_no_nan(self.true_positives, self.total_count)
    fp_rate = _divide_no_nan(self.false_positives, self.total_count)
    return jnp.trapezoid(tp_rate, fp_rate)

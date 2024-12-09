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
import jax.numpy as jnp


def _default_threshold() -> jnp.ndarray:
  """This is the default threshold used for binary classification.

  Starts at 1.0 and goes down to 0.0 by an interval of 1/199.
  """
  return jnp.array(
      [1.0 + 1e-7] + [(198 - i) / (199) for i in range(198)] + [0.0 - 1e-7]
  )


def _divide_no_nan(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes a safe divide which returns 0 if the y is zero."""
  return jnp.where(y != 0, jnp.divide(x, y), 0.0)


@flax.struct.dataclass
class Precision(clu_metrics.Metric):
  """Computes precision for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_positives: The count of false positive instances from the given data,
      label, and threshold.
  """

  true_positives: jnp.ndarray
  false_positives: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, predictions: jnp.ndarray, labels: jnp.ndarray, threshold: float = 0.5
  ) -> 'Precision':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. This is calculated from the output logits of the model.
      labels: True labels. These are expected to be of dtype=int32.
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Precision metric.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `logits` and
        `labels` are incompatible.
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

  def compute(self) -> jnp.ndarray:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_positives)
    )


@flax.struct.dataclass
class Recall(clu_metrics.Metric):
  """Computes recall for binary classification given `logits` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_negatives: The count of false negative instances from the given data,
      label, and threshold.
  """

  true_positives: jnp.ndarray
  false_negatives: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, predictions: jnp.ndarray, labels: jnp.ndarray, threshold: float = 0.5
  ) -> 'Recall':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. This is calculated from the output logits of the model.
      labels: True labels. These are expected to be of dtype=int32.
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Recall metric.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `logits` and
        `labels` are incompatible.
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

  def compute(self) -> jnp.ndarray:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_negatives)
    )


@flax.struct.dataclass
class AUCPR(clu_metrics.Metric):
  """Computes area under the precision-recall curve for binary classification given `logits` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    false_negatives: The count of false negative instances from the given data
      and label at each threshold.
  """

  # shape: (threshold, 1)
  true_positives: jnp.ndarray
  false_positives: jnp.ndarray
  false_negatives: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, predictions: jnp.ndarray, labels: jnp.ndarray
  ) -> 'AUCPR':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. This is calculated from the output logits of the model.
      labels: True labels. These are expected to be of dtype=int32.

    Returns:
      The area under the precision-recall curve.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `logits` and
        `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(predictions, _default_threshold()[..., None])
    pred_is_neg = jnp.logical_not(pred_is_pos)
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    false_positives = pred_is_pos * label_is_neg
    false_negatives = pred_is_neg * label_is_pos

    return cls(
        true_positives=true_positives.sum(axis=1),
        false_positives=false_positives.sum(axis=1),
        false_negatives=false_negatives.sum(axis=1),
    )

  def merge(self, other: 'AUCPR') -> 'AUCPR':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )

  def compute(self) -> jnp.ndarray:
    precision = _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_positives)
    )
    recall = _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_negatives)
    )
    return jnp.trapezoid(precision, recall)


@flax.struct.dataclass
class AUCROC(clu_metrics.Metric):
  """Computes area under the receiver operation characteristic curve for binary classification given `logits` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    total_count: The count of every data point.
  """

  # shape: (threshold, 1)
  true_positives: jnp.ndarray
  false_positives: jnp.ndarray
  # shape: (1)
  total_count: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, predictions: jnp.ndarray, labels: jnp.ndarray
  ) -> 'AUCROC':
    """Updates the metric.

    Args:
      predictions: A floating point `Tensor` whose values are in the range [0,
        1]. This is calculated from the output logits of the model.
      labels: True labels. These are expected to be of dtype=int32.

    Returns:
      The area under the receiver operation characteristic curve.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `logits` and
        `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(predictions, _default_threshold()[..., None])
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    false_positives = pred_is_pos * label_is_neg
    total_count = jnp.size(labels)

    return cls(
        true_positives=true_positives.sum(axis=-1),
        false_positives=false_positives.sum(axis=-1),
        total_count=total_count,
    )

  def merge(self, other: 'AUCROC') -> 'AUCROC':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
        total_count=self.total_count + other.total_count,
    )

  def compute(self) -> jnp.ndarray:
    tp_rate = _divide_no_nan(self.true_positives, self.total_count)
    fp_rate = _divide_no_nan(self.false_positives, self.total_count)
    return jnp.trapezoid(tp_rate, fp_rate)

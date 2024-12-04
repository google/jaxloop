"""A collection of different CLU metrics for the training library."""

from clu import metrics as clu_metrics
import flax
import jax.numpy as jnp


def _divide_no_nan(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes a safe divide which returns 0 if the y is zero."""
  dtype = jnp.result_type(x, y)
  y_is_zero = jnp.equal(y, 0.0)
  div = jnp.divide(x, jnp.where(y_is_zero, jnp.ones((), dtype=dtype), y))
  return jnp.where(y_is_zero, jnp.zeros((), dtype=dtype), div)


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

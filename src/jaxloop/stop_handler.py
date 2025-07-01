"""Early stopping handler for Jaxloop."""

from absl import logging
import jax.numpy as jnp


class StopTrainingError(Exception):
  """Exception raised to stop training."""

  def __init__(self, message, best_step=None, best_metric=None):
    super().__init__(message)
    self.best_step = best_step
    self.best_metric = best_metric

  def __str__(self):
    message = self.args[0]
    return (
        f"{message} Training stopped early. Best Step: {self.best_step}, Best"
        f" Metric: {self.best_metric:.4f})"
    )


class MetricMonitoringStopHandler:
  """Manages the state and logic for stopping training.

  Decides if stopping training should be triggered based on a monitored metric.
  """

  def __init__(
      self,
      monitored_metric: str,
      min_delta: float = 1e-8,
      patience: int = 10,
      mode: str = "auto",
      baseline: float | None = None,
  ):
    """Initializes the handler.

    Args:
      monitored_metric: The metric to be monitored.
      min_delta: The minimum change in the monitored metric to be considered an
        improvement.
      patience: The number of consecutive cycles without improvement to wait
        before triggering early stopping. This is treated as a countdown.
      mode: One of "min", "max", or "auto". In "min" mode, training will stop
        when the quantity monitored has stopped decreasing; in "max" mode it
        will stop when the quantity has stopped increasing; in "auto" mode, the
        direction is automatically inferred from the name of the monitored
        quantity.
      baseline: The minimum threshold value the monitored metric must reach
        before early stopping can be triggered. If specified, early stopping
        will not activate until the metric surpasses this baseline, regardless
        of patience. Once the baseline is reached, normal patience-based early
        stopping begins.
    """
    self.monitored_metric = monitored_metric
    self.min_delta = abs(min_delta)
    self.patience = patience  # Save to reset _patience_left on improvement
    self.baseline = baseline
    self._patience_left = self.patience
    self.best_step = 0
    self._initialized = False

    if mode not in ["min", "max", "auto"]:
      raise ValueError(f"Mode must be 'min', 'max' or 'auto', got {mode}")

    if mode == "min":
      self.monitored_op = jnp.less
    elif mode == "max":
      self.monitored_op = jnp.greater
    else:
      if (
          self.monitored_metric.endswith("acc")
          or self.monitored_metric.endswith("accuracy")
          or self.monitored_metric.endswith("auc")
      ):
        self.monitored_op = jnp.greater
      else:
        self.monitored_op = jnp.less

    self.improvement_delta = (
        -self.min_delta if self.monitored_op == jnp.less else self.min_delta
    )

    if self.baseline is not None:
      self.best = self.baseline
      self._initialized = True
      self.best_step = 0
    else:
      self.best = jnp.inf if self.monitored_op == jnp.less else -jnp.inf

    logging.debug(
        "[MetricMonitoringStopHandler] Initialized.",
    )

  def update_state(self, current_metric: float, current_step: int) -> bool:
    """Updates the state of the early stopping handler.

    Args:
      current_metric: The current value of the monitored metric.
      current_step: The current step number.

    Returns:
      True if the metric improved, False otherwise.
    """
    if not self._initialized:
      self.best = current_metric
      self.best_step = current_step
      self._initialized = True
      logging.debug(
          "[MetricMonitoringStopHandler] Initialized best metric '%s' to %.4f"
          " at step %d.",
          self.monitored_metric,
          self.best,
          self.best_step,
      )
      if self.baseline is not None:
        if self.monitored_op(self.baseline, self.best):
          self.best = self.baseline
          self.best_step = 0
          logging.debug(
              "[MetricMonitoringStopHandler] Baseline %.4f is better than first"
              " metric %.4f",
              self.baseline,
              current_metric,
          )
      return True

    improved = self.monitored_op(
        current_metric, self.best + self.improvement_delta
    )

    if improved:
      logging.debug(
          "[MetricMonitoringStopHandler] Metric '%s' improved from %.4f to %.4f"
          " at step %d.",
          self.monitored_metric,
          self.best,
          current_metric,
          current_step,
      )
      self.best = current_metric
      self.best_step = current_step
      self._patience_left = self.patience
      return True
    else:
      self._patience_left -= 1
      logging.debug(
          "[MetricMonitoringStopHandler] Metric '%s' did not improve from %.4f"
          " at step %d. Patience remaining: %d",
          self.monitored_metric,
          self.best,
          current_step,
          self._patience_left,
      )
      return False

  def should_stop(self) -> bool:
    """Returns True if early stopping should be triggered."""
    return self._initialized and self._patience_left <= 0

  def get_best_info(self) -> tuple[float, int]:
    """Returns the best metric value and step number."""
    return self.best, self.best_step

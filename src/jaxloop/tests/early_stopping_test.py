from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import action_loop as action_loop_lib
from jaxloop import actions
from jaxloop import stop_handler
from jaxloop import types
import optax


class TestModel(nn.Module):
  """A simple, fully-connected neural network model."""

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
    x = nn.Dense(features=4)(x)
    x = nn.relu(x)
    return nn.Dense(features=1)(x)


class MetricMonitoringStopHandlerTest(parameterized.TestCase):
  """Unit tests for the MetricMonitoringStopHandler class in isolation."""

  def test_initialization_and_mode_inference(self):
    """Tests correct mode selection and initialization."""
    handler_min = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="loss", mode="min"
    )
    self.assertIs(handler_min.monitored_op, jnp.less)

    handler_max = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="accuracy", mode="max"
    )
    self.assertIs(handler_max.monitored_op, jnp.greater)

    handler_auto = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="val_auc", mode="auto"
    )
    self.assertIs(handler_auto.monitored_op, jnp.greater)

    with self.assertRaises(ValueError):
      stop_handler.MetricMonitoringStopHandler(
          monitored_metric="loss", mode="invalid_mode"
      )

  @parameterized.named_parameters(
      dict(testcase_name="min_mode", mode="min", sequence=[10, 9, 9.5], best=9),
      dict(
          testcase_name="max_mode",
          mode="max",
          sequence=[0.8, 0.9, 0.85],
          best=0.9,
      ),
  )
  def test_mode_logic_and_state_updates(self, mode, sequence, best):
    """Tests metric tracking and state updates for min/max modes."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="metric", mode=mode
    )

    # First update initializes the best metric
    self.assertTrue(handler.update_state(sequence[0], 100))
    self.assertEqual(handler.best, sequence[0])
    self.assertEqual(handler._patience_left, handler.patience)

    # An improvement resets patience
    self.assertTrue(handler.update_state(sequence[1], 200))
    self.assertEqual(handler.best, best)
    self.assertEqual(handler.best_step, 200)
    self.assertEqual(handler._patience_left, handler.patience)

    # No improvement decreases patience
    self.assertFalse(handler.update_state(sequence[2], 300))
    self.assertEqual(handler.best, best)
    self.assertEqual(handler._patience_left, handler.patience - 1)

  def test_patience_and_should_stop(self):
    """Tests that stopping occurs only after patience runs out."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="loss", patience=2
    )
    handler.update_state(10.0, 100)
    self.assertFalse(handler.should_stop())

    handler.update_state(11.0, 200)  # No improvement, _patience_left=patience-1
    self.assertFalse(handler.should_stop())

    handler.update_state(12.0, 300)  # No improvement, _patience_left=patience-2
    self.assertTrue(handler.should_stop())

  def test_min_delta(self):
    """Tests that changes smaller than min_delta are ignored."""
    handler = stop_handler.MetricMonitoringStopHandler("loss", min_delta=0.1)
    handler.update_state(10.0, 100)
    self.assertFalse(handler.update_state(9.95, 200))  # Not a real improvement
    self.assertEqual(handler._patience_left, handler.patience - 1)

  def test_baseline(self):
    """Tests that a baseline is respected."""
    handler = stop_handler.MetricMonitoringStopHandler(
        "loss", baseline=5.0, mode="min"
    )
    # Handler should be considered initialized immediately with a baseline.
    self.assertTrue(handler._initialized)
    self.assertEqual(handler.best, 5.0)

    # A metric worse than the baseline should not be an improvement.
    self.assertFalse(handler.update_state(10.0, 100))
    self.assertEqual(handler.best, 5.0)
    self.assertEqual(handler.best_step, 0)
    self.assertEqual(handler._patience_left, handler.patience - 1)

    # A metric better than the baseline should be an improvement.
    handler = stop_handler.MetricMonitoringStopHandler(
        "loss", baseline=5.0, mode="min"
    )
    self.assertTrue(handler.update_state(4.0, 100))
    self.assertEqual(handler.best, 4.0)
    self.assertEqual(handler.best_step, 100)
    self.assertEqual(handler._patience_left, handler.patience)

  def test_get_best_info(self):
    """Tests that get_best_info returns the correct state."""
    handler = stop_handler.MetricMonitoringStopHandler(monitored_metric="loss")
    handler.update_state(10.0, 100)
    handler.update_state(9.0, 200)
    best_metric, best_step = handler.get_best_info()
    self.assertEqual(best_metric, 9.0)
    self.assertEqual(best_step, 200)

  def test_initial_best_value_without_baseline(self):
    """Tests the initial `best` value when no baseline is provided."""
    # For min mode, best should be initialized to +infinity.
    handler_min = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="loss", mode="min"
    )
    self.assertEqual(handler_min.best, jnp.inf)

    # For max mode, best should be initialized to -infinity.
    handler_max = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="accuracy", mode="max"
    )
    self.assertEqual(handler_max.best, -jnp.inf)


class EarlyStoppingActionTest(absltest.TestCase):
  """Tests for EarlyStoppingAction logic and integration with SimpleTrainer."""

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.spec = {"input_features": ((2, 3), jnp.float32)}
    self.batches = [
        {
            "input_features": jnp.ones((2, 3)),
            "output_features": jnp.ones((2, 1)),
        }
        for _ in range(10)
    ]
    self.key = jax.random.PRNGKey(0)

    # Mock state for unit tests of the action
    self.mock_state = types.TrainState.create(
        apply_fn=self.model.apply, params={}, tx=optax.identity()
    )

  def test_missing_metric(self):
    """Tests that the action handles when the monitored metric is missing."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="val_loss"
    )
    action = actions.EarlyStoppingAction(handler=handler)
    state = self.mock_state.replace(step=100)
    outputs = {"other_metric": 1.0}
    with self.assertLogs(level="WARNING") as logs:
      action(state, outputs)
    self.assertEqual(handler._patience_left, handler.patience)
    self.assertIn("not available in the outputs dictionary", logs.output[0])

  def test_none_outputs(self):
    """Tests that the action handles when the outputs object is None."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="val_loss"
    )
    action = actions.EarlyStoppingAction(handler=handler)
    state = self.mock_state.replace(step=100)
    outputs = None
    with self.assertLogs(level="WARNING") as logs:
      # The action should not raise an error
      action(state, outputs)
    # No state should have changed in the handler
    self.assertEqual(handler._patience_left, handler.patience)
    self.assertFalse(handler._initialized)
    # A warning should be logged
    self.assertIn("not available in the outputs dictionary", logs.output[0])

  def test_nan_metric(self):
    """Tests that a NaN metric correctly raises StopTrainingError."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="val_loss"
    )
    action = actions.EarlyStoppingAction(handler=handler)
    state = self.mock_state.replace(step=100)
    outputs = {"val_loss": jnp.nan}
    with self.assertRaises(stop_handler.StopTrainingError) as e:
      action(state, outputs)
    self.assertIn("Metric 'val_loss' was NaN", str(e.exception))

  def test_stop_handler_in_action_loop(self):
    """Tests EarlyStoppingAction's integration with an ActionLoop."""
    handler = stop_handler.MetricMonitoringStopHandler(
        monitored_metric="val_loss", patience=2
    )
    early_stop_action = actions.EarlyStoppingAction(handler=handler)

    self.eval_call_count = 0
    self.loss_sequence = [1.0, 0.5, 0.6, 0.7, 0.8]

    def mock_eval_step(
        state: types.TrainState, batch: Any, *args, **kwargs
    ) -> tuple[types.TrainState, types.Output]:
      """A mock evaluation step that returns a predefined loss."""
      current_loss = self.loss_sequence[self.eval_call_count]
      self.eval_call_count += 1
      outputs = {"val_loss": jnp.array(current_loss)}
      return state, outputs

    # Usage: create an ActionLoop and an initial state
    eval_loop = action_loop_lib.ActionLoop(
        step=mock_eval_step, end_actions=[early_stop_action]
    )

    model_state = types.TrainState.create(
        apply_fn=self.model.apply, params={}, tx=optax.identity()
    )

    mock_eval_dataset = [{"input_features": jnp.ones((2, 3))}]

    with self.assertRaises(stop_handler.StopTrainingError) as cm:
      for epoch in range(10):
        current_state = model_state.replace(step=epoch)
        eval_loop(current_state, iter(mock_eval_dataset))

    # - (step 0): loss=1.0. Best metric initialized. Best step=0.
    # - (step 1): loss=0.5. Improvement. Best metric=0.5. Best step=1.
    # - (step 2): loss=0.6. No improvement.
    # - (step 3): loss=0.7. No improvement. Patience met. Stop.
    self.assertEqual(self.eval_call_count, 4)
    self.assertEqual(cm.exception.best_metric, 0.5)
    self.assertEqual(cm.exception.best_step, 1)


if __name__ == "__main__":
  absltest.main()

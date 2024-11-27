"""Unit tests for the action loop library."""

from typing import Any, Optional, Tuple

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import action_loop
from jaxloop import actions
from jaxloop import step as step_lib
from jaxloop import types
import optax


Batch = types.Batch
Output = types.Output
State = types.TrainState
Step = step_lib.Step


class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


class TestStep(Step):

  def run(self, state: State, batch: Batch) -> Tuple[State, Optional[Output]]:
    return state.replace(step=state.step + 1), None


class TestAction(actions.Action):

  def __init__(self, interval: int = 1):
    super().__init__(interval=interval)
    self.call_count = 0

  def __call__(
      self, state: State, outputs: Optional[Output], **kwargs
  ) -> Optional[Any]:
    self.call_count += 1


class ActionLoopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.step = TestStep(
        jax.random.PRNGKey(0), self.model, optax.adam(1e-4))
    self.begin_actions = [
        TestAction(interval=1),
        TestAction(interval=2),
        TestAction(interval=3),
    ]
    self.end_actions = [
        TestAction(interval=1),
        TestAction(interval=2),
        TestAction(interval=3),
    ]
    self.loop = action_loop.ActionLoop(
        self.step,
        begin_actions=self.begin_actions,
        end_actions=self.end_actions,
    )

  def test_action_loop(self):
    shape = [1, 28, 28, 1]
    state = self.step.initialize_model(shape)

    dataset = iter([jnp.ones(shape)] * 10)
    state, _ = self.loop(state, dataset)
    self.assertEqual(self.loop.loop_count, 1)
    self.assertEqual(self.begin_actions[0].call_count, 1)
    self.assertEqual(self.begin_actions[1].call_count, 0)
    self.assertEqual(self.begin_actions[2].call_count, 0)
    self.assertEqual(self.end_actions[0].call_count, 1)
    self.assertEqual(self.end_actions[1].call_count, 0)
    self.assertEqual(self.end_actions[2].call_count, 0)

    dataset = iter([jnp.ones(shape)] * 10)
    self.loop(state, dataset)
    self.assertEqual(self.begin_actions[0].call_count, 2)
    self.assertEqual(self.begin_actions[1].call_count, 1)
    self.assertEqual(self.begin_actions[2].call_count, 0)
    self.assertEqual(self.end_actions[0].call_count, 2)
    self.assertEqual(self.end_actions[1].call_count, 1)
    self.assertEqual(self.end_actions[2].call_count, 0)

    dataset = iter([jnp.ones(shape)] * 10)
    self.loop.loop_count = 1
    self.loop(state, dataset)
    self.assertEqual(self.begin_actions[0].call_count, 3)
    self.assertEqual(self.begin_actions[1].call_count, 2)
    self.assertEqual(self.begin_actions[2].call_count, 0)
    self.assertEqual(self.end_actions[0].call_count, 3)
    self.assertEqual(self.end_actions[1].call_count, 2)
    self.assertEqual(self.end_actions[2].call_count, 0)


if __name__ == '__main__':
  absltest.main()

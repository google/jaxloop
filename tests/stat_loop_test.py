"""Unit tests for the statistic loop library."""

from typing import Optional, Tuple

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxloop import stat_loop
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


class StatLoopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = TestModel()
    self.step = TestStep(
        jax.random.PRNGKey(0), self.model, optax.adam(1e-4)
    )
    self.stat_names = [
        stat_loop.STAT_NUM_PARAMS,
        stat_loop.STAT_NUM_FLOPS,
        stat_loop.STAT_LIFE_TIME_SECS,
        stat_loop.STAT_STEPS_PER_SEC,
    ]
    self.loop = stat_loop.StatLoop(self.step, stat_names=self.stat_names)
    self.shape = [1, 28, 28, 1]
    self.dataset = iter([jnp.ones(self.shape)] * 10)

  def test_stat_loop(self):
    state = self.step.initialize_model(self.shape)
    _, outputs = self.loop(state, self.dataset)
    self.assertTrue(all(name in outputs for name in self.stat_names))


if __name__ == '__main__':
  absltest.main()

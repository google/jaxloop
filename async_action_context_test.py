"""Unit test for async_action_context."""

from concurrent.futures import thread
import time
from absl.testing import absltest
from jaxloop import async_action_context
import tensorflow as tf


class AsyncActionContextTest(absltest.TestCase):

  def test_add_futures_and_throw(self):
    def fn():
      time.sleep(2)
      raise ValueError("fn error")

    def fn_no_exception():
      time.sleep(2)

    def tf_fn():
      raise tf.errors.InvalidArgumentError(
          node_def=None, op=None, message="tf error"
      )

    c = async_action_context.get_async_action_context()
    f = thread.ThreadPoolExecutor(1, "thread").submit(fn)
    c.add_future("fn future", f)

    with self.assertRaises(ValueError):
      c.check_exceptions_and_raise()

    # Future is cleaned. This call shouldn't raise any exception.
    c.check_exceptions_and_raise()

    c = async_action_context.get_async_action_context()
    f = thread.ThreadPoolExecutor(1, "thread").submit(tf_fn)
    c.add_future("tf fn future", f)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      c.check_exceptions_and_raise()

    f = thread.ThreadPoolExecutor(1, "thread").submit(fn_no_exception)
    c.add_future("fn no exception future", f)

    # No exception raised.
    c.check_exceptions_and_raise()


if __name__ == "__main__":
  absltest.main()

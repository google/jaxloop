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

"""Context for async actions.

Contains the in-flight or materialized futures provided from the actions.
"""

from concurrent import futures
import logging
import threading
from typing import Any, Optional, Tuple

Future = futures.Future


class MutexLock:
  """RAII stype mutex lock."""

  def __init__(self, mutex: threading.Lock):
    self.mutex = mutex

  def __enter__(self):
    self.mutex.acquire()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.mutex.release()


class AsyncActionContext:
  """Context for async actions.

  Attributes:
    futures: A set of tuple that contains the name of the action that creates
      it, and the "Future" object returned by threadpool executor.
    mutex: The lock that protects accessing the `futures` set.
  """

  futures: set[Tuple[str, Future[Any]]] = set()
  mutex = threading.Lock()

  def check_exceptions_and_raise(self, raise_exception: bool = True) -> None:
    """Checks all futures for any exceptions and raises any one.

    In common use cases, we expect `Action`s are called periodically  between
    inner train loops, and usually an async action's (e.g. SavedModel action)
    `Future` should be fulfilled by the next time the action is called. If you
    observe host overhead due to waiting for futures, please consider increase
    the action interval.

    Args:
      raise_exception: Whether to raise the exception if there is one. If set as
        False, just log the warning but the training job won't fail.
    """
    with MutexLock(self.mutex):
      for future_name, future in self.futures:
        # This line will block until the future is ready.
        e = future.exception()
        if e is not None:
          self.futures = set()
          note = (
              'The exception above was caught in a previous async action for'
              f' {future_name}.'
          )
          if raise_exception:
            e.add_note(note)
            raise e
          else:
            logging.error(note)

      # If there's no exception in the futures, clear the set.
      self.futures = set()

  def add_future(self, future_name: str, future: Future[Any]) -> None:
    with MutexLock(self.mutex):
      self.futures.add((future_name, future))


# Do not access directly.
_async_action_context: Optional[AsyncActionContext] = None


def get_async_action_context() -> AsyncActionContext:
  """Obtains the global AsyncActionContext.

  Returns:
    The global AsyncActionContext.
  """
  global _async_action_context
  if _async_action_context is None:
    _async_action_context = AsyncActionContext()
  return _async_action_context

"""Jaxloop Data loader for JAX model Trainers."""

import abc

from jaxloop import step


class TrainerDataLoader(abc.ABC):
  """An abstract class for loading data for Jaxloop Trainers."""

  @abc.abstractmethod
  def __next__(self) -> step.Batch:
    """Generates a batch of data.

    This method should be overridden by the specific data loader. It should
    return a batch of data in the format expected by the step function used.
    """
    pass


class SimpleDataLoader(TrainerDataLoader):
  """A Trainers-compatible Data Loader that operates on general iterators."""

  def __init__(self, batch_size: int):
    self._batch_size = batch_size

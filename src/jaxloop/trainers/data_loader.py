"""Jaxloop Data loader for JAX model Trainers."""

import abc
from typing import Iterator
import jax
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
  """A Trainers-compatible Data Loader that operates on general iterators.

  This data loader is designed to be a simple wrapper around any Python
  iterator. It is the responsibility of the wrapped iterator to handle batching
  and shuffling of the dataset.

  Example usage:

    Sample Data Source:
      num_batches = 10
      batch_size = 8
      list_of_batches = [
          {
              'input_features': jnp.ones((batch_size, 4)),
              'output_features': jnp.ones((batch_size, 1)),
          }
          for _ in range(num_batches)
      ]

    Create a standard Python iterator from the data source:
      data_iterator = iter(list_of_batches)

    Wrap the iterator with SimpleDataLoader:
      data_loader = SimpleDataLoader(dataset=data_iterator)

    Instantiate a trainer:
      trainer = simple_trainer.SimpleTrainer(
          ... # other trainer parameters (model, epochs, etc.)
          batch_spec={
              'input_features': (batch_size, 4),
              'output_features': (batch_size, 1),
          }
      )

    Run the training process, passing the SimpleDataLoader instance:
      outputs = trainer.train(data_loader)
      print(f"Training complete. Final loss: {outputs['loss'][-1]:.4f}")
  """

  def __init__(
      self, dataset: Iterator[step.Batch], validate_batches: bool = False
  ):
    """Initializes the SimpleDataLoader.

    Args:
      dataset: An iterator that yields batches of data.
      validate_batches: Whether to validate the batches against the batch spec.
        This is useful for catching potential shape mismatches in the dataset.
    """
    self._dataset = dataset
    self._validate_batches = validate_batches

    self._batch_spec: step.BatchSpec = None
    self._first_batch: step.Batch = None

  def __iter__(self):
    """Returns the iterator object."""
    return self

  def __next__(self) -> step.Batch:
    """Returns the next batch of data.

    Raises:
      StopIteration: If the dataset has no more batches.
    """
    if self._first_batch is not None:
      batch = self._first_batch
      self._first_batch = None
      return batch

    batch = next(self._dataset)
    if self._validate_batches:
      self._validate_batch(batch)
    return batch

  def _validate_batch(self, batch: step.Batch):
    """Validates the batch structure and leaf shapes against the batch spec.

    Args:
      batch: The batch to validate.

    Raises:
      ValueError: If the batch spec is not initialized.
      TypeError: If leaves in the batch are not array-like.
    """
    try:
      current_batch_spec = jax.tree.map(lambda x: (x.shape, x.dtype), batch)
    except AttributeError as e:
      raise TypeError(
          "Failed to get shape from leaves of the current batch. "
          "Ensure all leaves are array-like objects with a .shape attribute."
      ) from e

    if current_batch_spec != self.get_batch_spec():
      raise TypeError(
          "Batch spec mismatch. Expected: {expected}, got: {got}".format(
              expected=self._batch_spec, got=current_batch_spec
          )
      )

  def get_batch_spec(self) -> step.BatchSpec:
    """Returns the batch spec by peeking at the first batch of the dataset."""
    if not self._batch_spec and not self._first_batch:
      try:
        self._first_batch = next(
            self._dataset
        )  # Save for the first __next__ call.
      except StopIteration:
        raise ValueError("Dataset is empty.") from None

      self._batch_spec = jax.tree.map(
          lambda x: (x.shape, x.dtype), self._first_batch
      )

    return self._batch_spec

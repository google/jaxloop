"""Jaxloop Data loader for JAX model Trainers."""

import abc
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import jax
import jax.numpy as jnp
from jaxloop import step

# A PyTree of batches, where each leaf is either an jax.Array or a NestedBatch.
NestedBatch = Union[jax.Array, Dict[str, "NestedBatch"]]

PADDING_TOKEN = -100


class TrainerDataLoader(abc.ABC):
  """An abstract class for loading data for Jaxloop Trainers."""

  @abc.abstractmethod
  def __next__(self) -> step.Batch | NestedBatch:
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
              "input_features": jnp.ones((batch_size, 4)),
              "output_features": jnp.ones((batch_size, 1)),
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
              "input_features": (batch_size, 4),
              "output_features": (batch_size, 1),
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


class TransformerDataLoader(TrainerDataLoader):
  """A data loader for sequence-based datasets, typical for Transformers.

  This data loader wraps an iterator that yields individual pre-tokenized
  samples (as lists of integers) and handles the batching and collation process.
  batch_size samples are grouped into a batch and collated via collate_fn,
  which pads the samples and creates attention masks. The resulting Batch
  is formatted as a dictionary.

  The default collate_fn is designed for causal language modeling and produces
  the following Batch:
    {
        "input_features": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "output_features": {
            "labels": labels,
        },
    }

  Example usage:

    Sample Data Source (an iterator yielding lists of token IDs):
      raw_dataset = [
          [101, 7592, 1010, 2026, 3899, 102],
          [101, 2023, 2003, 1037, 7455, 2039, 102],
          [101, 2054, 2024, 2017, 2031, 2001, 2183, 102],
      ]
      data_iterator = iter(raw_dataset)

    Create the data loader:
      data_loader = TransformerDataLoader(
          dataset=data_iterator,
          batch_size=2,
      )

    The first batch produced by `next(data_loader)` would be a dictionary
    containing padded 'input_ids', 'attention_mask', and 'labels'.
  """

  def __init__(
      self,
      dataset: Iterator[jax.Array],
      batch_size: int,
      collate_fn: Optional[Callable[[Any], Any]] = None,
  ):
    """Initializes the TransformerDataLoader.

    Args:
      dataset: An iterator that yields lists of token IDs.
      batch_size: The number of samples to include in each batch.
      collate_fn: A function that collates a list of samples into a single
        batch. If not provided, a default collate_fn is used for causal language
        modeling.
    """

    if dataset is None or batch_size is None:
      raise ValueError("`dataset` and `batch_size` must be provided.")

    if batch_size <= 0:
      raise ValueError("`batch_size` must be positive.")

    self._dataset = dataset
    self._batch_size = batch_size
    self._collate_fn = (
        collate_fn if collate_fn is not None else self.clm_collate_fn
    )

  def __iter__(self):
    """Returns the iterator object."""
    return self

  def __next__(self) -> NestedBatch:
    """Fetches a batch of samples from the dataset and collates them.

    This method pulls `batch_size` number of samples from the dataset,
    assembles them into a list, and passes them to the collate function to
    produce the final, trainer-ready batch. If the underlying iterator is
    exhausted, it collates any remaining samples into a final, smaller batch.

    Returns:
      A dictionary representing the collated batch of data.

    Raises:
      StopIteration: If the dataset is exhausted and no samples are left to
      form a batch.
    """
    samples = []
    try:
      for _ in range(self._batch_size):
        sample = next(self._dataset)
        samples.append(sample)
    except StopIteration:
      pass

    if not samples:
      raise StopIteration  # So we do not attempt to collate an empty list.

    return self._collate_fn(samples)

  def clm_collate_fn(self, samples: List[jax.Array]) -> NestedBatch:
    """Default collate function for causal language modeling.

    Takes a list of input sequences (samples), pads them to the length of the
    longest sequence, and stacks them into a single `input_ids` array. Creates
    a corresponding `attention_mask` and `labels` for each sample.

    Example usage:

      For an input of:
        samples = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9],
        ]

      The output of clm_collate_fn(samples) would be the following batch:
        {
            "input_features": {
                "input_ids": [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]],
                "attention_mask": [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
            },
            "output_features": {
                "labels": [[1, 2, 3, -100], [4, 5, -100, -100], [6, 7, 8, 9]],
            },
        }

    Args:
      samples: A list of samples to collate.

    Returns:
      A dictionary containing the collated `input_ids`, `attention_mask`, and
      `labels`, structured for use with a trainer.
    """

    max_len = max(len(s) for s in samples)

    padded_samples = []
    attention_masks = []
    for s in samples:
      padding_len = max_len - len(s)

      padded_sample = jnp.pad(
          s, (0, padding_len), "constant", constant_values=0
      )
      padded_samples.append(padded_sample)

      # Create `attention mask`: 1 for real tokens, 0 for padding.
      attention_mask = jnp.pad(
          jnp.ones(s.shape), (0, padding_len), "constant", constant_values=0
      )
      attention_masks.append(attention_mask)

    input_ids = jnp.stack(padded_samples)

    attention_mask = jnp.stack(attention_masks)

    # `labels` are `input_ids` with padding replaced by the padding token,
    # to be ignored by the loss function during training.
    labels = jnp.where(attention_mask == 1, input_ids, PADDING_TOKEN)

    # A batch with "input_features" and "output_features" for compatibility
    # with existing training loop solutions.
    return {
        "input_features": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "output_features": {
            "labels": labels,
        },
    }

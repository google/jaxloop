"""Jaxloop Data Preprocessor for batches."""

from typing import Any, Callable
from jaxloop import types

Batch = types.Batch


class DataPreprocessor:
  """A callable class that executes a data preprocessing pipeline.

  Example usage:

    # A batch of the format of a types.Batch.
    batch = ["JAX is a powerful library.", "This is the second sample.",
    "And a final sentence."]

    tokenizer = gm.text.Gemma3Tokenizer()

    def batch_to_lowercase(batch: Batch) -> Batch:
      return Batch([v.lower() for v in batch])

    def tokenize(batch: Batch) -> Batch:
      return Batch([tokenizer(v) for v in batch])

    builder = DataPreprocessorBuilder()
    batch_processor = (
        builder
        .add_stage(batch_to_lowercase)
        .add_stage(tokenize)
        .build()
    )

    # Create a custom step that uses the created batch processing function.
    class MyStep(Step):
      def __init__(self, batch_processor: Callable[[Batch], Batch]):
        self._batch_processing_fn = batch_processor

      def preprocess_batch(self, batch: Batch) -> Batch:
        return self._batch_processing_fn(batch)

    # Or, create an iterator of processed samples from the builder function.
    processed_batch = iter(batch_processor(batch))
  """

  def __init__(self, stages: list[Callable[[Any], Any]]):
    """Initializes the data preprocessor.

    Args:
      stages: A list of data transformation stages to be executed in order.
    """
    self._stages = stages

  @property
  def stages(self) -> list[Callable[[Any], Any]]:
    """Returns the list of data transformation stages."""
    return self._stages

  def __call__(self, batch: Batch) -> Batch:
    """Executes the data preprocessing pipeline on a batch.

    Args:
      batch: The batch to be processed.

    Returns:
      The processed batch.
    """
    for stage_fn in self._stages:
      batch = stage_fn(batch)
    return batch


class DataPreprocessorBuilder:
  """A builder that configures and creates a data preprocessing pipeline.

  The pipeline is callable via a DataPreprocessor.
  """

  def __init__(self):
    """Initializes the builder."""
    self._stages: list[Callable[[Any], Any]] = []

  def add_stage(self, stage: Callable[[Any], Any]) -> "DataPreprocessorBuilder":
    """Adds a data transformation stage to the pipeline.

    Args:
      stage: The preprocessing stage to be added.

    Returns:
      The builder instance.
    """
    self._stages.append(stage)
    return self

  def build(self) -> DataPreprocessor:
    """Constructs the final processing function.

    Returns:
      A DataPreprocessor instance that can execute the entire pipeline.
    """
    return DataPreprocessor(stages=self._stages)

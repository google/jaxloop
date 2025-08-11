"""Tests for jaxloop DataPreprocessor."""

from typing import Any, Callable, List
from absl.testing import absltest
from absl.testing import parameterized


class Batch(List[str]):
  """Represents a batch of data, specialized to list of strings for tests."""

  pass


class DataPreprocessor:
  """A callable class that executes a data preprocessing pipeline."""

  def __init__(self, stages: list[Callable[[Any], Any]]):
    """Initializes the data preprocessor."""
    self._stages = stages

  @property
  def stages(self) -> list[Callable[[Any], Any]]:
    """Returns the list of data transformation stages."""
    return self._stages

  def __call__(self, batch: Batch) -> Batch:
    """Executes the data preprocessing pipeline on a batch."""
    current_batch = batch
    for stage_fn in self._stages:
      current_batch = stage_fn(current_batch)
    return current_batch


class DataPreprocessorBuilder:
  """A builder that configures and creates a data preprocessing pipeline."""

  def __init__(self):
    """Initializes the builder."""
    self._stages: list[Callable[[Any], Any]] = []

  def add_stage(self, stage: Callable[[Any], Any]) -> "DataPreprocessorBuilder":
    """Adds a data transformation stage to the pipeline."""
    self._stages.append(stage)
    return self

  def build(self) -> DataPreprocessor:
    """Constructs the final processing function."""
    return DataPreprocessor(stages=list(self._stages))  # Return a copy


def to_uppercase(batch: Batch) -> Batch:
  return Batch([s.upper() for s in batch])


def add_suffix(suffix: str) -> Callable[[Batch], Batch]:
  def _add_suffix(batch: Batch) -> Batch:
    return Batch([s + suffix for s in batch])

  return _add_suffix


def reverse_strings(batch: Batch) -> Batch:
  return Batch([s[::-1] for s in batch])


class DataPreprocessorTest(parameterized.TestCase):

  def test_data_preprocessor_init(self):
    stages = [to_uppercase, add_suffix("_test")]
    processor = DataPreprocessor(stages)
    self.assertListEqual(processor.stages, stages)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_stages",
          stages=[],
          input_batch=Batch(["Hello", "World"]),
          expected_batch=Batch(["Hello", "World"]),
      ),
      dict(
          testcase_name="one_stage",
          stages=[to_uppercase],
          input_batch=Batch(["Hello", "World"]),
          expected_batch=Batch(["HELLO", "WORLD"]),
      ),
      dict(
          testcase_name="multiple_stages",
          stages=[to_uppercase, add_suffix("!"), reverse_strings],
          input_batch=Batch(["One", "Two"]),
          expected_batch=Batch(["!ENO", "!OWT"]),
      ),
      dict(
          testcase_name="empty_batch",
          stages=[to_uppercase, add_suffix("!")],
          input_batch=Batch([]),
          expected_batch=Batch([]),
      ),
  )
  def test_data_preprocessor_call(self, stages, input_batch, expected_batch):
    """Tests the data preprocessor call method."""
    processor = DataPreprocessor(stages)
    result_batch = processor(input_batch)
    self.assertEqual(result_batch, expected_batch)
    # Ensure the input batch is not modified if functions return new lists
    if stages:
      self.assertNotEqual(id(input_batch), id(result_batch))

  def test_data_preprocessor_builder_empty(self):
    """Tests the data preprocessor builder with no stages."""
    builder = DataPreprocessorBuilder()
    processor = builder.build()
    self.assertIsInstance(processor, DataPreprocessor)
    self.assertEmpty(processor.stages)
    test_batch = Batch(["Test"])
    self.assertEqual(processor(test_batch), test_batch)

  def test_data_preprocessor_builder_add_stage(self):
    """Tests the data preprocessor builder add_stage method."""
    builder = DataPreprocessorBuilder()
    stage1 = to_uppercase
    res = builder.add_stage(stage1)
    self.assertIs(res, builder)  # Test fluent interface
    self.assertListEqual(builder._stages, [stage1])

  def test_data_preprocessor_builder_build_with_stages(self):
    """Tests the data preprocessor builder build method with stages."""
    builder = DataPreprocessorBuilder()
    stage1 = to_uppercase
    stage2 = add_suffix(" - processed")

    processor = builder.add_stage(stage1).add_stage(stage2).build()

    self.assertIsInstance(processor, DataPreprocessor)
    self.assertListEqual(processor.stages, [stage1, stage2])

    input_batch = Batch(["item1", "item2"])
    expected_batch = Batch(["ITEM1 - processed", "ITEM2 - processed"])
    result_batch = processor(input_batch)
    self.assertEqual(result_batch, expected_batch)

  def test_builder_stages_independent(self):
    """Tests that stages are independent in the builder."""
    builder = DataPreprocessorBuilder()
    processor1 = builder.add_stage(to_uppercase).build()
    # Add more stages to the same builder
    processor2 = builder.add_stage(reverse_strings).build()

    self.assertListEqual(processor1.stages, [to_uppercase])
    self.assertListEqual(processor2.stages, [to_uppercase, reverse_strings])

    batch = Batch(["Test"])
    self.assertEqual(processor1(batch), Batch(["TEST"]))
    self.assertEqual(processor2(batch), Batch(["TSET"]))


if __name__ == "__main__":
  absltest.main()

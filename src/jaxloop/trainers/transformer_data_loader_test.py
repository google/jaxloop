from typing import Any, Dict, List
from absl.testing import absltest
import jax.numpy as jnp
from jaxloop.trainers import data_loader


class TransformerDataLoaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenized_dataset = [
        jnp.array([1, 2, 3]),
        jnp.array([4, 5]),
        jnp.array([6, 7, 8, 9]),
        jnp.array([10]),
    ]

  def test_invalid_init(self):
    """Tests ValueError for missing dataset or batch_size."""
    with self.assertRaisesRegex(
        ValueError, "`dataset` and `batch_size` must be provided."
    ):
      data_loader.TransformerDataLoader(dataset=None, batch_size=2)
    with self.assertRaisesRegex(
        ValueError, "`dataset` and `batch_size` must be provided."
    ):
      data_loader.TransformerDataLoader(
          dataset=iter(self.tokenized_dataset), batch_size=None
      )
    with self.assertRaisesRegex(ValueError, "`batch_size` must be positive."):
      data_loader.TransformerDataLoader(
          dataset=iter(self.tokenized_dataset), batch_size=0
      )
    with self.assertRaisesRegex(ValueError, "`batch_size` must be positive."):
      data_loader.TransformerDataLoader(
          dataset=iter(self.tokenized_dataset), batch_size=-1
      )

  def test_iteration_and_default_collate(self):
    """Tests batching, padding, and masks with the default collate function."""
    batch_size = 2
    loader = data_loader.TransformerDataLoader(
        dataset=iter(self.tokenized_dataset), batch_size=batch_size
    )
    batches = list(loader)

    self.assertLen(batches, 2)  # 4 samples, batch_size 2

    # Batch 1
    batch1 = batches[0]
    self.assertIn("input_features", batch1)
    self.assertIn("output_features", batch1)
    inputs1 = batch1["input_features"]
    outputs1 = batch1["output_features"]

    expected_ids1 = jnp.array([[1, 2, 3], [4, 5, 0]])
    expected_mask1 = jnp.array([[1, 1, 1], [1, 1, 0]])
    expected_labels1 = jnp.array([[1, 2, 3], [4, 5, -100]])

    self.assertTrue(jnp.array_equal(inputs1["input_ids"], expected_ids1))
    self.assertTrue(jnp.array_equal(inputs1["attention_mask"], expected_mask1))
    self.assertTrue(jnp.array_equal(outputs1["labels"], expected_labels1))

    # Batch 2
    batch2 = batches[1]
    inputs2 = batch2["input_features"]
    outputs2 = batch2["output_features"]

    expected_ids2 = jnp.array([[6, 7, 8, 9], [10, 0, 0, 0]])
    expected_mask2 = jnp.array([[1, 1, 1, 1], [1, 0, 0, 0]])
    expected_labels2 = jnp.array([[6, 7, 8, 9], [10, -100, -100, -100]])

    self.assertTrue(jnp.array_equal(inputs2["input_ids"], expected_ids2))
    self.assertTrue(jnp.array_equal(inputs2["attention_mask"], expected_mask2))
    self.assertTrue(jnp.array_equal(outputs2["labels"], expected_labels2))

  def test_iteration_incomplete_last_batch(self):
    """Tests the last batch when dataset size is not a multiple of batch_size."""
    batch_size = 3
    loader = data_loader.TransformerDataLoader(
        dataset=iter(self.tokenized_dataset), batch_size=batch_size
    )
    batches = list(loader)

    self.assertLen(batches, 2)  # ceil(4 / 3) = 2

    # Batch 1 (size 3)
    batch1 = batches[0]
    self.assertEqual(batch1["input_features"]["input_ids"].shape[0], 3)

    # Batch 2 (size 1)
    batch2 = batches[1]
    inputs2 = batch2["input_features"]
    outputs2 = batch2["output_features"]

    expected_ids2 = jnp.array([[10]])
    expected_mask2 = jnp.array([[1]])
    expected_labels2 = jnp.array([[10]])

    self.assertTrue(jnp.array_equal(inputs2["input_ids"], expected_ids2))
    self.assertTrue(jnp.array_equal(inputs2["attention_mask"], expected_mask2))
    self.assertTrue(jnp.array_equal(outputs2["labels"], expected_labels2))

  def test_custom_collate_fn(self):
    """Tests using a custom collate function."""

    def custom_collate(samples: List[List[int]]) -> Dict[str, Any]:
      return {"processed_samples": jnp.array([len(s) for s in samples])}

    batch_size = 4
    loader = data_loader.TransformerDataLoader(
        dataset=iter(self.tokenized_dataset),
        batch_size=batch_size,
        collate_fn=custom_collate,
    )
    batch = next(loader)

    expected_batch = {"processed_samples": jnp.array([3, 2, 4, 1])}
    self.assertTrue(
        jnp.array_equal(
            batch["processed_samples"], expected_batch["processed_samples"]
        )
    )

    with self.assertRaises(StopIteration):
      next(loader)

  def test_empty_dataset(self):
    """Tests behavior with an empty dataset."""
    loader = data_loader.TransformerDataLoader(dataset=iter([]), batch_size=2)
    with self.assertRaises(StopIteration):
      next(loader)
    # Check list() also works as expected
    self.assertEmpty(list(loader))

  def test_stop_iteration(self):
    """Ensures StopIteration is raised correctly after all batches."""
    loader = data_loader.TransformerDataLoader(
        dataset=iter(self.tokenized_dataset), batch_size=1
    )
    for _ in range(len(self.tokenized_dataset)):
      try:
        next(loader)
      except StopIteration:
        self.fail("StopIteration raised too early")

    with self.assertRaises(StopIteration):
      next(loader)


if __name__ == "__main__":
  absltest.main()

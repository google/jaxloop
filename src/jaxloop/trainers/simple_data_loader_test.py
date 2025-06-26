from absl.testing import absltest
import jax
import jax.numpy as jnp
from jaxloop.trainers import data_loader


class SimpleDataLoaderTest(absltest.TestCase):

  def setUp(self):
    """Set up a simple list of batches to be used as an iterator source."""
    super().setUp()
    # Manually create the batches that the underlying iterator will produce,
    # and test if our SimpleDataLoader correctly passes these through.
    # All batches have the same spec.
    self.good_batches = [
        {
            "input_features": jnp.array([[0, 1], [2, 3]]),
            "output_features": jnp.array([[0], [1]]),
        },
        {
            "input_features": jnp.array([[4, 5], [6, 7]]),
            "output_features": jnp.array([[2], [3]]),
        },
        {
            "input_features": jnp.array([[8, 9], [10, 11]]),
            "output_features": jnp.array([[4], [5]]),
        },
    ]

    self.expected_spec = {
        "input_features": ((2, 2), jnp.dtype("int32")),
        "output_features": ((2, 1), jnp.dtype("int32")),
    }

  def test_iteration_and_batch_passthrough(self):
    """Tests that the loader yields all items from the input iterator."""
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(self.good_batches)
    )

    result_batches = list(simple_loader)

    self.assertLen(
        result_batches,
        len(self.good_batches),
        "Loader did not yield the correct number of batches.",
    )

    # Check that the batches are the same
    for result_batch, expected_batch in zip(result_batches, self.good_batches):
      self.assertTrue(
          jax.tree_util.tree_all(
              jax.tree.map(jnp.array_equal, result_batch, expected_batch)
          )
      )

  def test_stop_iteration(self):
    """Tests that the loader correctly raises StopIteration at the end."""
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(self.good_batches)
    )
    for _ in range(len(self.good_batches)):
      next(simple_loader)
    with self.assertRaises(StopIteration):
      next(simple_loader)

  def test_get_batch_spec(self):
    """Tests that get_batch_spec returns the correct spec and caches the first batch."""
    dataset = iter(self.good_batches)
    simple_loader = data_loader.SimpleDataLoader(dataset=dataset)

    # Call get_batch_spec
    spec = simple_loader.get_batch_spec()
    self.assertEqual(spec, self.expected_spec)

    # Check that the first batch is still iterated on (not skipped)
    first_batch = next(simple_loader)
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.array_equal, first_batch, self.good_batches[0])
        )
    )

    # Should be able to call get_batch_spec again and get the same result.
    spec_again = simple_loader.get_batch_spec()
    self.assertEqual(spec_again, self.expected_spec)

    remaining_batches = list(simple_loader)
    self.assertLen(remaining_batches, len(self.good_batches) - 1)
    for result_batch, expected_batch in zip(
        remaining_batches, self.good_batches[1:]
    ):
      self.assertTrue(
          jax.tree_util.tree_all(
              jax.tree.map(jnp.array_equal, result_batch, expected_batch)
          )
      )

  def test_get_batch_spec_empty_dataset(self):
    """Tests that get_batch_spec raises ValueError with an empty dataset."""
    simple_loader = data_loader.SimpleDataLoader(dataset=iter([]))
    with self.assertRaises(ValueError):
      simple_loader.get_batch_spec()

  def test_validate_batches_matching(self):
    """Tests validation with matching batches."""
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(self.good_batches), validate_batches=True
    )

    spec = simple_loader.get_batch_spec()
    self.assertEqual(spec, self.expected_spec)

    list(simple_loader)

  def test_validate_batches_struct_mismatch(self):
    """Tests validation failure due to struct mismatch."""
    bad_batches = self.good_batches[:1] + [{
        "input_features": jnp.ones((2, 2)),
        # Missing "output_features"
    }]
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(bad_batches), validate_batches=True
    )
    with self.assertRaises(TypeError):
      list(simple_loader)

  def test_validate_batches_shape_mismatch(self):
    """Tests validation failure due to tensor shape mismatch."""
    bad_batches = self.good_batches[:1] + [{
        "input_features": jnp.ones((2, 2)),
        "output_features": jnp.ones((5, 1)),
    }]
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(bad_batches), validate_batches=True
    )
    with self.assertRaises(TypeError):
      list(simple_loader)

  def test_validate_batches_shape_mismatch_nested(self):
    """Tests validation failure due to shape mismatch in a nested structure."""
    nested_batches = [
        {"data": {"a": jnp.ones((2, 2)), "b": jnp.ones((2, 1))}},
        {"data": {"a": jnp.ones((2, 2)), "b": jnp.ones((9, 9))}},
    ]
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(nested_batches), validate_batches=True
    )
    with self.assertRaises(TypeError):
      list(simple_loader)

  def test_validate_batches_false(self):
    """Tests that no errors are raised when validate_batches is False."""
    mismatched_batches = [
        {"input": jnp.ones((2, 2))},
        {"something_else": jnp.ones((3, 3))},
        {"input": jnp.ones((1, 1))},
    ]
    simple_loader = data_loader.SimpleDataLoader(
        dataset=iter(mismatched_batches), validate_batches=False
    )

    spec = simple_loader.get_batch_spec()
    self.assertEqual(spec, {"input": ((2, 2), jnp.dtype("float32"))})

    results = list(simple_loader)
    self.assertLen(results, 3)


if __name__ == "__main__":
  absltest.main()

import os.path

from absl import flags
from absl.testing import absltest
from jaxloop import actions
import orbax.checkpoint as ocp

FLAGS = flags.FLAGS


class CheckpointActionTest(absltest.TestCase):

  def test_async_save(self):
    # In reality, this would be a tree of np.ndarray or jax.Array.
    pytree = {'a': 0}

    options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(
            os.path.join(FLAGS.test_tmpdir, '/tmp/ckpt/')
        ),
        options=options,
    ) as mngr:
      ckpt_action = actions.CheckpointAction(ckpt_manager=mngr)
      ckpt_action.save(step=0, state=pytree, outputs=None)

      restored_pytree = mngr.restore(0)
      self.assertEqual(restored_pytree, pytree)


if __name__ == '__main__':
  absltest.main()

import asyncio
import collections
import os.path
import threading
import time
from unittest import mock

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


class _RecordingPolicy(ocp.checkpoint_managers.SaveDecisionPolicy):
  """Records the checked steps and allows saves only on `allowed_steps`."""

  def __init__(self, allowed_steps):
    self.allowed_steps = set(allowed_steps)
    self.checked_steps = []

  def should_save(self, step, previous_steps, *, context):
    del previous_steps, context
    self.checked_steps.append(step.step)
    return step.step in self.allowed_steps


# Minimal pytree state with a `step` field.
_State = collections.namedtuple('_State', ['step'])


class ContinuousCheckpointActionTest(absltest.TestCase):

  def _manager(self, policy):
    return ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(
            os.path.join(FLAGS.test_tmpdir, self.id())
        ),
        options=ocp.CheckpointManagerOptions(
            save_decision_policy=policy, enable_async_checkpointing=False
        ),
    )

  def _run(self, action, first_step, last_step):
    for step in range(first_step, last_step + 1):
      action(_State(step), None)

  def test_does_not_ask_step_to_wait(self):
    with self._manager(_RecordingPolicy([])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self.assertFalse(action.wait_for_outputs)
      self.assertEqual(action.interval, 1)

  def test_calls_save_every_step_and_policy_decides(self):
    policy = _RecordingPolicy(allowed_steps=[3, 5])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 6)
      # Save steps are checked twice: once before the device read, once by
      # `CheckpointManager.save()`.
      self.assertEqual(policy.checked_steps, [1, 2, 3, 3, 4, 5, 5, 6])
      self.assertEqual(mngr.all_steps(), [3, 5])

  def test_logs_only_saved_steps(self):
    policy = _RecordingPolicy(allowed_steps=[2])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      with self.assertLogs(level='INFO') as logs:
        self._run(action, 1, 3)
      saved = [line for line in logs.output if 'Saved checkpoint to' in line]
      self.assertLen(saved, 1)
      self.assertIn('on step 2.', saved[0])

  def test_milestone_and_preemption_policies_are_honored(self):
    policy = ocp.checkpoint_managers.AnySavePolicy([
        ocp.checkpoint_managers.SpecificStepsPolicy(steps=[7]),
        ocp.checkpoint_managers.PreemptionCheckpointingPolicy(),
    ])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      # The manager checks for preemption on every save() call.
      with mock.patch.object(
          mngr, 'reached_preemption', side_effect=lambda s: s == 11
      ) as reached:
        self._run(action, 1, 12)
      self.assertEqual(
          sorted({c.args[0] for c in reached.call_args_list}),
          list(range(1, 13)),
      )
      self.assertEqual(mngr.all_steps(), [7, 11])

  def test_finalize_saves_last_step_once(self):
    policy = _RecordingPolicy(allowed_steps=[2])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 5)
      self.assertTrue(action.finalize(_State(5)))
      self.assertEqual(mngr.all_steps(), [2, 5])
      # The last step is already saved: no second save.
      self.assertFalse(action.finalize(_State(5)))
      self.assertEqual(mngr.all_steps(), [2, 5])

  def test_finalize_saves_when_no_checkpoint_exists(self):
    with self._manager(_RecordingPolicy([])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 3)
      self.assertIsNone(mngr.latest_step())
      self.assertTrue(action.finalize(_State(3)))
      self.assertEqual(mngr.all_steps(), [3])

  def test_finalize_waits_for_saves(self):
    with self._manager(_RecordingPolicy([])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 1)
      with mock.patch.object(
          mngr, 'wait_until_finished', wraps=mngr.wait_until_finished
      ) as wait:
        self.assertTrue(action.finalize(_State(1)))
        # Once before reading the latest step, once after the final save.
        self.assertEqual(wait.call_count, 2)

  def test_finalize_waits_before_reading_latest_step(self):
    calls = []
    with self._manager(_RecordingPolicy([])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 1)
      with mock.patch.object(
          mngr, 'wait_until_finished', side_effect=lambda: calls.append('wait')
      ), mock.patch.object(
          mngr, 'latest_step', side_effect=lambda: calls.append('latest') or 1
      ):
        self.assertFalse(action.finalize(_State(1)))
      self.assertEqual(calls, ['wait', 'latest'])

  def test_finalize_saves_nothing_if_action_never_ran(self):
    with self._manager(_RecordingPolicy([])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self.assertFalse(action.finalize(_State(0)))
      self.assertEqual(mngr.all_steps(), [])

  def test_finalize_uses_save_override(self):
    class _RecordingAction(actions.ContinuousCheckpointAction):

      def __init__(self, ckpt_manager):
        super().__init__(ckpt_manager)
        self.save_calls = []

      def save(self, step, state, outputs, force=False):
        self.save_calls.append((step, force))
        return super().save(step, state, outputs, force=force)

    with self._manager(_RecordingPolicy([])) as mngr:
      action = _RecordingAction(mngr)
      self._run(action, 4, 4)
      self.assertTrue(action.finalize(_State(4)))
      self.assertEqual(action.save_calls, [(4, True)])
      self.assertEqual(mngr.all_steps(), [4])

  def test_passes_outputs_unchanged_without_best_fn(self):
    class Unfetchable:
      """Stands in for an array that cannot be copied to this host."""

      def __array__(self, *args, **kwargs):
        raise RuntimeError('cannot fetch')

      def tolist(self):
        raise RuntimeError('cannot fetch')

    outputs = {'loss': Unfetchable()}
    with self._manager(_RecordingPolicy(allowed_steps=[2])) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      with mock.patch.object(mngr, 'save', wraps=mngr.save) as save:
        for step in range(1, 3):
          action(_State(step), outputs)
        save.assert_called_once()
        self.assertIs(save.call_args.kwargs['metrics'], outputs)
      self.assertEqual(mngr.all_steps(), [2])

  def test_saves_and_finalize_store_step_metrics_with_best_fn(self):
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=_RecordingPolicy(allowed_steps=[2]),
        enable_async_checkpointing=False,
        best_fn=lambda metrics: metrics['loss'],
        best_mode='min',
    )
    directory = ocp.test_utils.erase_and_create_empty(
        os.path.join(FLAGS.test_tmpdir, self.id())
    )
    with ocp.CheckpointManager(directory, options=options) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      for step in range(1, 4):
        action(_State(step), {'loss': 1.0 / step})
      # `finalize()` without outputs stores the last step's outputs.
      self.assertTrue(action.finalize(_State(3)))
      self.assertEqual(mngr.all_steps(), [2, 3])
      self.assertAlmostEqual(mngr.metrics(2)['loss'], 0.5, places=5)
      self.assertAlmostEqual(mngr.metrics(3)['loss'], 1.0 / 3, places=5)
      self.assertEqual(mngr.best_step(), 3)

  def test_host_step_reads_device_step_once(self):
    class CountingState:
      reads = 0

      def __init__(self, step):
        self._step = step

      @property
      def step(self):
        CountingState.reads += 1
        return self._step

    policy = _RecordingPolicy(allowed_steps=[])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      for step in range(11, 21):
        action(CountingState(step), None)
      self.assertEqual(CountingState.reads, 1)
      self.assertEqual(policy.checked_steps, list(range(11, 21)))

  def test_saves_under_device_step_after_skipped_steps(self):
    policy = _RecordingPolicy(allowed_steps=[2, 7, 8])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      action(_State(1), None)
      # The caller skipped steps: the host step becomes 2, the device step is 7.
      with self.assertLogs(level='ERROR'):
        action(_State(7), None)
      # Saved as 7, never as 2.
      self.assertEqual(mngr.all_steps(), [7])
      # The host step is back in sync.
      action(_State(8), None)
      self.assertEqual(mngr.all_steps(), [7, 8])

  def test_no_save_when_device_step_is_already_saved(self):
    policy = _RecordingPolicy(allowed_steps=[3, 4])
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 3, 3)
      self.assertEqual(mngr.all_steps(), [3])
      # The device step went back to 3: no save, and no error from Orbax.
      with self.assertLogs(level='ERROR'):
        action(_State(3), None)
      self.assertEqual(mngr.all_steps(), [3])

  def test_continuous_checkpointing_policy(self):
    policy = ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=3600
    )
    with self._manager(policy) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 5)
      # The first check saves; later checks are inside the minimum interval.
      self.assertEqual(mngr.all_steps(), [1])

  def test_continuous_checkpointing_options_async_target_steps(self):
    original_on_commit = ocp.path.atomicity.on_commit_callback
    step_1_commit_started = threading.Event()

    async def slow_step_1_commit(tmp_dir, *args, **kwargs):
      # Keeps the initial save at step 1 in flight until it is cancelled.
      if tmp_dir.get_final().name == '1':
        step_1_commit_started.set()
        await asyncio.sleep(10)
      return await original_on_commit(tmp_dir, *args, **kwargs)

    options = ocp.checkpoint_managers.continuous_checkpointing_options(
        target_interval_steps=3, minimum_interval_secs=3600
    )
    directory = ocp.test_utils.erase_and_create_empty(
        os.path.join(FLAGS.test_tmpdir, self.id())
    )
    with mock.patch.object(
        ocp.path.atomicity, 'on_commit_callback', new=slow_step_1_commit
    ), ocp.CheckpointManager(directory, options=options) as mngr:
      action = actions.ContinuousCheckpointAction(mngr)
      self._run(action, 1, 1)
      self.assertTrue(step_1_commit_started.wait(timeout=5.0))
      # Target step 3 cancels the step 1 save instead of waiting for it.
      start = time.time()
      self._run(action, 2, 7)
      self.assertTrue(action.finalize(_State(7)))
      self.assertLess(time.time() - start, 5.0)
      self.assertEqual(mngr.all_steps(), [3, 6, 7])
    self.assertFalse(os.path.exists(os.path.join(directory, '1')))


if __name__ == '__main__':
  absltest.main()

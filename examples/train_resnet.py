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

"""Train ResNet model with Jaxloop.

Test locally:

  blaze run -c opt //third_party/py/jaxloop/examples:train_resnet --
  --alsologtostderr --workdir=/tmp/jax_example
"""

from collections.abc import Sequence
import logging
from typing import Optional

from absl import flags
from clu import metric_writers
from etils import epath
import jax
from jax import sharding
from jax.experimental import mesh_utils
from jaxloop import actions
from jaxloop import eval_loop as eval_loop_lib
from jaxloop import outer_loop as outer_loop_lib
from jaxloop import partition
from jaxloop import train_loop as train_loop_lib
from jaxloop import types
from jaxloop.examples import resnet_dataset
from jaxloop.examples import resnet_model
from jaxloop.examples import resnet_pygrain_dataset
from jaxloop.examples import resnet_step
import optax
from orbax import checkpoint

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

Mesh = sharding.Mesh

_DATASOURCE = flags.DEFINE_string(
    'datasource',
    'pygrain_dataset',
    'A string flag to determine the datasource of the experiment. If not'
    ' specified, we will use Pygrain.Dataset.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Work directory to store checkpoints, metrics and other artifacts. This is'
    ' a required option.',
    required=True,
)

FLAGS = flags.FLAGS

_CHECKPOINT_INTERVAL = 100  # This is loop interval, not step interval.
_SUMMARY_INTERVAL = 10  # This is loop interval, not step interval.
_EVAL_LOOP_INTERVAL = 2  # This is loop interval, not step interval.
_INPUT_SIZE = (224, 224)
_IMAGE_SHAPE = (224, 224, 3)
_NUM_EPOCHS = 100
_PER_CORE_BATCH_SIZE = 128
_PLACER_DATA_DIR = '/readahead/200M/placer/prod/home/tensorflow-datasets-cns-storage-owner/datasets/'
_SEED = 42
_TRAIN_LOOP_STEPS = 100


def compute_metrics(outputs: types.Output) -> types.Output:
  """Computes metrics from the outputs of a step."""
  if not outputs:
    return outputs
  computed_metrics = {}
  for metric_name, metric_list in outputs.items():
    metric = metric_list[0]
    for metric_item in metric_list[1:]:
      metric = metric.merge(metric_item)
    computed_metrics[metric_name] = metric.compute()
  return computed_metrics


class ResnetTrainLoop(train_loop_lib.TrainLoop):
  """Training loop for Resnet, with a customized metric function."""

  def end(
      self, state: types.TrainState, outputs: Optional[types.Output]
  ) -> tuple[types.TrainState, Optional[types.Output]]:
    return super().end(state, compute_metrics(outputs))


class ResnetEvalLoop(eval_loop_lib.EvalLoop):
  """Eval loop for Resnet, with a customized metric function."""

  def end(
      self, state: types.TrainState, outputs: Optional[types.Output]
  ) -> tuple[types.TrainState, Optional[types.Output]]:
    return super().end(state, compute_metrics(outputs))


def _get_data(
    input_size: tuple[int, int],
    batch_size: int,
    data_dir: str,
):
  match _DATASOURCE.value:
    case 'pygrain_dataloader':
      return resnet_pygrain_dataset.imagenet_datasets(
          input_size=input_size,
          batch_size=batch_size,
          data_dir=data_dir,
      )
    case 'pygrain_dataset':
      return resnet_pygrain_dataset.imagenet_dataloaders(
          input_size=input_size,
          batch_size=batch_size,
          data_dir=data_dir,
      )
    case 'tf_dataset':
      return resnet_dataset.imagenet_datasets(
          input_size=input_size,
          batch_size=batch_size,
          data_dir=data_dir,
      )
    case _:
      raise ValueError(f'Unsupported datasource: {_DATASOURCE.value}')


def main(_: Sequence[str]) -> None:
  workdir = epath.Path(_WORKDIR.value)
  workdir.mkdir(parents=True, exist_ok=True)

  prng = jax.random.PRNGKey(_SEED)

  global_batch_size = _PER_CORE_BATCH_SIZE * jax.device_count()
  per_host_batch_size = global_batch_size // jax.process_count()

  model = resnet_model.ResNet()

  steps_per_epoch, ds_train, ds_val = _get_data(
      _INPUT_SIZE, per_host_batch_size, _PLACER_DATA_DIR
  )

  train_steps = _NUM_EPOCHS * steps_per_epoch
  schedule = optax.warmup_cosine_decay_schedule(
      warmup_steps=steps_per_epoch,
      decay_steps=_NUM_EPOCHS * steps_per_epoch,
      init_value=1.0e-3,
      peak_value=1.0e-2,
      end_value=1.0e-5,
  )
  optimizer = optax.adamw(
      learning_rate=schedule,
      nesterov=True,
  )
  num_devices = len(jax.devices())
  mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ('data',))
  partitioner = partition.DataParallelPartitioner(mesh=mesh, data_axis='data')
  train_step = resnet_step.ResnetStep(
      base_prng=prng,
      model=model,
      optimizer=optimizer,
      partitioner=partitioner,
      train=True,
  )
  eval_step = resnet_step.ResnetStep(
      base_prng=prng,
      model=model,
      partitioner=partitioner,
      train=False,
  )

  ckpt_manager = checkpoint.CheckpointManager(
      workdir / 'checkpoints',
      checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler()),
      checkpoint.CheckpointManagerOptions(max_to_keep=3),
  )
  ckpt_action = actions.CheckpointAction(ckpt_manager, _CHECKPOINT_INTERVAL)
  train_events_dir = workdir / 'train'
  train_metrics_writer = metric_writers.create_default_writer(
      train_events_dir,
      just_logging=jax.process_index() > 0,
      asynchronous=False,
  )
  train_summary_action = actions.SummaryAction(
      train_metrics_writer, _SUMMARY_INTERVAL
  )
  eval_events_dir = workdir / 'eval'
  eval_metrics_writer = metric_writers.create_default_writer(
      eval_events_dir,
      just_logging=jax.process_index() > 0,
      asynchronous=False,
  )
  eval_summary_action = actions.SummaryAction(
      eval_metrics_writer, _SUMMARY_INTERVAL
  )

  train_loop = ResnetTrainLoop(
      train_step, end_actions=[train_summary_action, ckpt_action]
  )
  eval_loops = [ResnetEvalLoop(eval_step, end_actions=[eval_summary_action])]

  outer_loop = outer_loop_lib.OuterLoop(
      train_loop=train_loop, eval_loops=eval_loops
  )
  outer_loop(
      state=train_step.initialize_model((global_batch_size, *_IMAGE_SHAPE)),
      train_dataset=ds_train,
      train_total_steps=train_steps,
      train_loop_steps=_TRAIN_LOOP_STEPS,
      eval_specs=[
          outer_loop_lib.EvalSpec(
              dataset=ds_val,
              eval_loop_interval=_EVAL_LOOP_INTERVAL,
          )
      ],
  )


if __name__ == '__main__':
  g3_multiprocessing.handle_main(main)

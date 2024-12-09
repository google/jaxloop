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

r"""An example XManager script for launching ResNet training on Borg.

Originally from: /google3/third_party/py/xmanager/examples/alphabet/codelab/xm_launch.py

EXAMPLE USAGE:
POOL=xx; ALLOC=yy  # Set to your team's resource pool and allocation.
google_xmanager launch \
third_party/py/jaxloop/examples/xm_launch.py -- \
--xm_resource_pool=$POOL \
--xm_resource_alloc=$ALLOC
"""

import datetime
from typing import Any, Dict

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import requirements_flag
from xmanager.contrib.internal import tensorboard
from xmanager.contrib.internal import xm_jax

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'Experiment name. Will use ISO timestamp if not provided.',
)
_PLATFORM = requirements_flag.DEFINE_requirements(
    'platform',
    'viperlite_4x2',
    (
        'Platform to use for the training job (e.g. v100, df_4x4). See'
        ' parse_requirements_spec() in http://shortn/_dUJCgjpijN.'
    ),
)
_CELL = flags.DEFINE_string(
    'cell',
    'rs',
    'Where to launch the job.',
)
_DATASOURCE = flags.DEFINE_string(
    'datasource',
    'pygrain_dataset',
    'A string flag to determine the datasource of the experiment. If not'
    ' specified, we will use Pygrain.Dataset.',
)
_PRIORITY = flags.DEFINE_integer(
    'priority',
    200,
    'Priority of the job. Use 200 for PROD, 115 for BATCH and 25 for FREEBIE.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Work directory where to store checkpoints, metrics and other artifacts. A'
    ' few placeholders are available, see default value.',
)

FLAGS = flags.FLAGS

_DEFAULT_WORKDIR = '/cns/{cell}-d/home/{author}/xm/{xid}'
_SRCDIR = '//third_party/py/jaxloop/examples'


def _get_requirements_and_extra_args(
    platform: Dict[str, Any],
    cell: str,
    priority: int,
) -> tuple[xm.JobRequirements, Dict[str, Any]]:
  """Gets the job requirements and args."""

  (accelerator, spec) = list(platform.items())[0]
  acc_req = xm.JobRequirements(**platform)

  # We will run this experiment on TPU only.
  is_pufferfish = acc_req.accelerator == xm.ResourceType.PUFFERFISH
  if is_pufferfish and 'twisted' not in spec:
    spec += '_untwisted'
  req_dict = {accelerator: spec}
  extra_args = {'xla_tpu_spmd_rng_bit_generator_unsafe': 'true'}

  requirements = xm.JobRequirements(
      location=cell, priority=priority, **req_dict
  )

  return requirements, extra_args


def _create_trainer_job(experiment: xm.Experiment) -> xm.Job:
  """Creates the trainer job."""

  annotations = experiment.context.annotations
  trainer_requirements, extra_flags = _get_requirements_and_extra_args(
      _PLATFORM.value, _CELL.value, _PRIORITY.value
  )

  executor = xm_abc.Borg(requirements=trainer_requirements)
  annotations.add_tags('tpu')

  workdir = _WORKDIR.value or _DEFAULT_WORKDIR.format(
      cell=executor.requirements.location,
      name=experiment.context.annotations.title,
      author=experiment.context.creator,
      xid=experiment.experiment_id,
  )

  extra_flags['workdir'] = workdir
  extra_flags['datasource'] = _DATASOURCE.value

  tensorboard.add_tensorboard_corp(
      experiment,
      workdir=workdir,
      # Stops TensorBoard Corp event exporter 10 mins after experiment stops.
      termination_delay_secs=600,
  )

  [executable] = experiment.package([
      xm.bazel_binary(
          executor_spec=executor.Spec(),
          label=f'{_SRCDIR}:train_resnet',
          bazel_args=xm_abc.bazel_args.tpu(),
          args=xm_jax.JaxFlags().flags(),
      ),
  ])

  trainer_job = xm.Job(
      executable=executable,
      executor=executor,
      name='trainer',
      args={
          'xprof_port': '%port_xprof%',
          **extra_flags,
      },
  )

  return trainer_job


def main(_) -> None:
  """Launches the experiment using the arguments from the command line."""

  experiment_name = _EXPERIMENT_NAME.value or datetime.datetime.now().strftime(
      '%Y%m%dT%H%M%S.%f'
  )
  with xm_abc.create_experiment(experiment_title=experiment_name) as experiment:
    experiment.add(_create_trainer_job(experiment))


if __name__ == '__main__':
  app.run(main)  # This block is not executed when run under XManager.

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

"""The pipeline loop library for JAX models."""

import collections
import itertools
from typing import Any, Dict, Iterator, Optional, Tuple

from jaxloop import eval_loop
from jaxloop import stat_loop
from jaxloop import step as step_lib
from jaxloop import types
from ml_metrics import aggregates
from ml_metrics import pipeline as pipeline_lib

# pylint: disable=logging-fstring-interpolation

State = types.TrainState
Step = step_lib.Step


class PipelineLoop(eval_loop.EvalLoop):
  """The loop class using ML Analysis Toolkit pipeline.

  If users don't provide pipeline, default pipeline will be used. The
  default pipeline will run loop step on dataset and aggregate the result using
  a preset aggregation logic below. Users can also override make_pipeline()
  method to create custom pipeline.
  """

  def __init__(
      self,
      step: Step,
      pipeline: Optional[pipeline_lib.Pipeline] = None,
      **kwargs,
  ):
    """Initializes a PipelineLoop instance.

    Args:
      step: Step instance used to get method to make the default pipeline.
      pipeline: A ML Analysis Toolkit pipeline instance. If passed in,
        PipelineLoop will use the instance instead of creating a default
        pipeline instance.
      **kwargs: Keyword args passed to parent initializer.
    """
    super().__init__(step, **kwargs)
    self._pipeline = pipeline

  def make_pipeline(self, state: State) -> pipeline_lib.Pipeline:
    """Makes a pipeline using ML analysis toolkit.

    Args:
      state: The model state.

    Returns:
      A pipeline instance.
    """

    class ReduceLoopOutputs(aggregates.Aggregatable):
      """Aggregation that merges output per batch into final result."""

      def __init__(self, update_func):
        self._update_func = update_func

      def create_state(self) -> dict[str, Any]:
        return collections.defaultdict(list)

      def update_state(
          self, reduced_loop_outputs: dict[str, Any], inputs: dict[str, Any]
      ) -> dict[str, Any]:
        return self._update_func(reduced_loop_outputs, inputs)

      def get_result(
          self, reduced_loop_outputs: dict[str, Any]
      ) -> dict[str, Any]:
        return reduced_loop_outputs

    log_num_flops = stat_loop.STAT_NUM_FLOPS in self._stat_names
    return (
        pipeline_lib.Pipeline.new()
        .apply(output_keys='batch')
        .assign(
            input_keys='batch',
            fn=lambda batch: self._step(
                state, batch, log_num_flops=log_num_flops
            ),
            assign_keys=('state', 'step_output'),
        )
        .aggregate(
            input_keys='step_output',
            fn=ReduceLoopOutputs(self.update_outputs),
            output_keys='loop_output',
        )
    )

  def run(
      self,
      state: State,
      dataset: Iterator[Any],
      num_steps: Optional[int] = None,
      **kwargs,
  ) -> Tuple[State, Optional[Dict[str, Any]]]:
    """Runs loop with ML Analysis Toolkit pipeline.

    Args:
      state: The model state.
      dataset: The dataset iterator.
      num_steps: The number of steps to run.
      **kwargs: Addtional keyword arguments.

    Returns:
      A tuple of the model state and output.
    """
    self._log_steps(int(state.step), num_steps)

    if num_steps is not None:
      dataset = itertools.islice(dataset, num_steps)
    data_source = pipeline_lib.Pipeline.new().data_source(dataset)
    data_pipeline = data_source.chain(
        self._pipeline or self.make_pipeline(state)
    )
    return state, next(iter(data_pipeline.make()().values()))

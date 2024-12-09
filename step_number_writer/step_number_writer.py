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

"""Step number writer interface."""

import abc


class StepNumberWriter(abc.ABC):
  """Step number writer interface.

  Usage:
  . Instantiate an implementation of the StepNumberWriter at the start of the
    workload, i.e., before the start of initialization.
  . At the conclusion of a step or a set of steps in the training loop, call
    StepNumberWriter.write(step_number) to record the timestamp and the step
    number. The timestamp is when the write() call was made and is in Unix
    milliseconds.

  Implementation:
  . Implementations of this interface should be able to handle workload
    restarts. Restarts can cause the step number to go backwards, so the
    implementation should not assume that the step number is always increasing.
  . Implementations are permitted to buffer writes and write them in batches,
    but they are not required to do so.
  . At construction, implementations should take suitable action so that the
    reader knows that the workload has started but no steps have executed. This
    will permit distinguishing initialization from training. As an example, if
    the implementation is writing to a file, it could create the file. An empty
    file will be interpreted as the workload being in initialization.
  . Implementations should not block the caller. Writing to a local file or an
    in-memory buffer in the foreground is acceptable. Writing to a remote file
    or sending an RPC should be done asynchronously.
  """

  @abc.abstractmethod
  def write(self, step_number: int) -> None:
    """Write the timestamp and the supplied step number."""

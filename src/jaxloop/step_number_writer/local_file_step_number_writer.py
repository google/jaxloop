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

"""Step number writer that writes to a local POSIX file."""

import datetime

from jaxloop.step_number_writer import step_number_writer


class LocalFileStepNumberWriter(step_number_writer.StepNumberWriter):
  r"""Step number writer that writes to a local POSIX file.

  Each entry in the file has the format:
  timestamp_ms:1234,step_number:56\n
  """

  def __init__(self, file_path: str):
    self._file_path = file_path
    # Create the file so that the reader can distinguish initialization from
    # training.
    with open(self._file_path, "w") as _:
      pass

  def write(self, step_number: int) -> None:
    timestamp_ms = (
        datetime.datetime.now(tz=datetime.UTC)
        - datetime.datetime(tzinfo=datetime.UTC, year=1970, month=1, day=1)
    ) / datetime.timedelta(milliseconds=1)
    with open(self._file_path, "a") as f:
      f.write(f"timestamp_ms:{timestamp_ms},step_number:{step_number}\n")

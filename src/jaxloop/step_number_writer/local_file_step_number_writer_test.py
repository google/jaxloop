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

"""Unit test for LocalFileStepNumberWriter."""

import os
import pathlib

from absl.testing import absltest
from jaxloop.step_number_writer import local_file_step_number_writer


class LocalFileStepNumberWriterTest(absltest.TestCase):

  def test_write(self):
    temp_dir = self.create_tempdir()
    file_path = os.path.join(temp_dir, "step_number.txt")
    local_writer = local_file_step_number_writer.LocalFileStepNumberWriter(
        file_path
    )
    local_writer.write(step_number=1)
    self.assertIn("step_number:1", pathlib.Path(file_path).read_text())


if __name__ == "__main__":
  absltest.main()

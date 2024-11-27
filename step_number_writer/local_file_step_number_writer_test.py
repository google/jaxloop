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

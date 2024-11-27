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

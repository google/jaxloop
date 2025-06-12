"""Utility functions and classes to simplify trainers usage."""

import dataclasses
from typing import Dict, Optional, Tuple, Union
from etils import epath
from jaxloop import outer_loop
from jaxloop import types

TrainerBatchSpec = Dict[
    str,
    Dict[
        str,
        Union[types.Shape, Tuple[types.Shape, Optional[types.DType]]],
    ],
]


class CheckpointingConfig:
  """Configuration for checkpointing."""

  checkpoint_spec: outer_loop.CheckpointSpec
  max_checkpoints: int = 3
  checkpoint_interval: int = 1

  def __init__(
      self,
      checkpoint_dir: str | epath.Path,
      max_checkpoints: int = 3,
      checkpoint_interval: int = 1,
  ):
    self.checkpoint_spec = self._build_checkpoint_spec(checkpoint_dir)
    self.max_checkpoints = max_checkpoints
    self.checkpoint_interval = checkpoint_interval

  def _build_checkpoint_spec(
      self, checkpoint_dir: str | epath.Path
  ) -> outer_loop.CheckpointSpec:
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = epath.Path(checkpoint_dir)
    return outer_loop.CheckpointSpec(checkpoint_dir=checkpoint_dir)


@dataclasses.dataclass(frozen=True)
class SummaryConfig:
  """Configuration for writing summaries."""

  path: str | epath.Path
  interval: int = 1
  flush_each_call: bool = False
  asynchronous: bool = True

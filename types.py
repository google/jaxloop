"""Types for JAX training loop library."""

import dataclasses
import enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import clu.metrics as clu_metrics
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np


@enum.unique
class MetricType(enum.Enum):
  """Type of metric.

  These types correspond to the types write support in CLU MetricWriter.
  """

  SCALAR = 'scalar'
  IMAGE = 'image'
  VIDEO = 'video'
  AUDIO = 'audio'
  TEXT = 'text'
  HISTOGRAM = 'histogram'
  OTHER = 'other'


Array = Union[np.ndarray, jnp.ndarray]
Scalar = Union[int, float, np.number, np.ndarray, jax.Array]


@dataclasses.dataclass(frozen=True)
class MetricWithMetadata:
  """Metric with additional metadata.

  Attributes:
    value: a metric value, can be an array or a scalar.
    type: an enum type indicating what type the metric is. If users set it to
      `OTHER`, they can include more information in the `metadata` dict so that
      specific action can handle such metric.
    metadata: a dict containing additional metadata. The metadata might be
      useful for consumer of the metric, such as a TensorBoard writer, or
      "sample_rate" for audio summaries.
  """

  value: Union[Scalar, Array]
  type: MetricType = MetricType.SCALAR
  metadata: dict[str, Any] | None = None


Batch = Union[Array, Dict[str, Array]]
Output = Dict[str, Any]
Shape = Union[int, Tuple[int], List[int]]
DType = jnp.dtype
BatchShape = Union[Shape, Dict[str, Shape]]
BatchSpec = Union[
    Shape,
    Tuple[Shape, Optional[DType]],
    Dict[str, Union[Shape, Tuple[Shape, Optional[DType]]]],
]
MetricsDict = Dict[str, Union[Scalar, Array, MetricWithMetadata]]
CluMetricType = Union[clu_metrics.Metric, clu_metrics.Collection]
IterateStopFn = Callable[[], bool]


class TrainState(train_state.TrainState):
  batch_stats: Optional[Any] = None

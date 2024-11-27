"""Minimal Imagenet dataset pipeline using PyGrain for ResNet."""

import dataclasses
from typing import Any

import cv2
import grain.python as grain
import numpy as np
import tensorflow_datasets as tfds

FlatFeatures = dict[str, Any]

_NUM_GRAIN_WORKERS = 16


@dataclasses.dataclass(frozen=True)
class ResnetPreprocessing(grain.MapTransform):
  """ImageNet data preprocessing for ResNet."""

  size: int

  def map(self, features: FlatFeatures) -> FlatFeatures:
    h, w = self.size, self.size
    image = features['image']
    image = cv2.resize(
        image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR
    )
    top = (image.shape[0] - h) // 2
    left = (image.shape[1] - w) // 2
    image = image[top : top + h, left : left + w, :]
    image = image.astype(dtype=np.float32, copy=False) / 255.0
    features['image'] = image
    del features['file_name']
    return features


def imagenet_dataloaders(
    input_size: tuple[int, int],
    batch_size: int,
    data_dir: str,
):
  """Creates a training and validation dataloader that produces batches of a given shape."""
  ds_builder = tfds.builder('imagenet2012')
  train_len = ds_builder.info.splits['train'].num_examples // batch_size
  ds_builder.download_and_prepare(
      download_dir=data_dir, file_format='array_record'
  )
  ds = ds_builder.as_data_source()
  train_loader = grain.load(
      source=ds['train'],
      shuffle=True,
      seed=45,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=[ResnetPreprocessing(size=input_size[0])],
      batch_size=batch_size,
      worker_count=_NUM_GRAIN_WORKERS,
  )
  eval_loader = grain.load(
      source=ds['validation'],
      num_epochs=1,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=[ResnetPreprocessing(size=input_size[0])],
      batch_size=batch_size,
      worker_count=_NUM_GRAIN_WORKERS,
      drop_remainder=True,
  )
  return train_len, iter(train_loader), eval_loader


def imagenet_datasets(
    input_size: tuple[int, int],
    batch_size: int,
    data_dir: str,
):
  """Creates a training and validation dataset that produces batches of a given shape."""
  ds_builder = tfds.builder('imagenet2012')
  train_len = ds_builder.info.splits['train'].num_examples // batch_size
  ds_builder.download_and_prepare(
      download_dir=data_dir, file_format='array_record'
  )
  ds = ds_builder.as_data_source()
  ds_train = (
      grain.MapDataset.source(ds['train'])
      .shuffle(seed=45)
      .repeat()
      .to_iter_dataset()
      .map(ResnetPreprocessing(size=input_size[0]))
      .batch(batch_size, drop_remainder=True)
      .mp_prefetch(grain.MultiprocessingOptions(num_workers=_NUM_GRAIN_WORKERS))
  )
  ds_eval = (
      grain.MapDataset.source(ds['validation'])
      .shuffle(seed=45)
      .to_iter_dataset()
      .map(ResnetPreprocessing(size=input_size[0]))
      .batch(batch_size, drop_remainder=True)
      .mp_prefetch(grain.MultiprocessingOptions(num_workers=_NUM_GRAIN_WORKERS))
  )
  return train_len, iter(ds_train), ds_eval
